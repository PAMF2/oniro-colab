"""Build ONIRO v35 Colab notebook.

v35 changes vs v34:
    - GQA attention (n_heads=12, n_kv_heads=3, group_size=4)
    - Flash attention via sdpa
    - Cross-cycle KV cache (refresh every 2 cycles)
    - 2 untied URM groups, weight-tied within each (group_loops=6, total=12)
    - GRID 14 -> 30 (matches ARC distribution)
    - D 128 -> 768, ffn 256 -> 3584
    - Params 2.5M -> ~20M
    - Heavy ARC-train mix (50%) + dihedral aug during training
    - TTFT 50 -> 100 steps

Pulls source from https://github.com/PAMF2/oniro-colab (public).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

OUT = Path(__file__).parent / "oniro_colab_micro.ipynb"


def _lines(text: str) -> list[str]:
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(text)}


def main() -> None:
    cells = []
    cells.append(md(textwrap.dedent("""
        # ONIRO v35 — URM 20M + GQA + KV Cache + RL on Colab

        **Setup:** Runtime → Change runtime type → **T4 GPU** → Save → Run All.

        Pulls source from https://github.com/PAMF2/oniro-colab (public).

        Architecture upgrades over v34:
        - GQA: n_kv_heads=3 vs n_heads=12 (4x KV reduction)
        - Flash attention (F.scaled_dot_product_attention)
        - Cross-cycle KV cache (refresh=2 cycles, ~50% attention savings)
        - 2 untied URM groups, weight-tied internally (6 cycles each)
        - GRID 14 -> 30 (covers ARC real distribution)
        - Params 2.5M -> 20M
        - ARC-1/2 train-set heavy (50%) + dihedral aug
        - TTFT 100 steps eval

        Phase A: supervised DIS training (~40k steps)
        Phase B: GRPO RL fine-tuning (~5k steps)
        AlphaEvolve-Godel outer mutation every 2000 steps.
        Eval on ARC-AGI-1, ARC-AGI-2, Sudoku, Math.

        Runtime ~6-8h on Colab free T4.
    """).strip()))

    cells.append(code(textwrap.dedent("""
        import os, sys, subprocess
        from pathlib import Path

        if Path('/content').exists():
            ROOT = Path('/content')
        elif Path('/kaggle/working').exists():
            ROOT = Path('/kaggle/working')
        else:
            ROOT = Path.home() / 'oniro_workspace'
            ROOT.mkdir(exist_ok=True)
        os.chdir(ROOT)
        print(f'workspace: {ROOT}')
    """).strip()))

    cells.append(code(textwrap.dedent("""
        ONIRO_REPO = ROOT / 'oniro-colab'
        if not ONIRO_REPO.exists():
            subprocess.check_call(['git', 'clone', '--depth', '1',
                                   'https://github.com/PAMF2/oniro-colab.git',
                                   str(ONIRO_REPO)])
        else:
            subprocess.check_call(['git', '-C', str(ONIRO_REPO), 'pull', '--ff-only'])
        if str(ONIRO_REPO) not in sys.path:
            sys.path.insert(0, str(ONIRO_REPO))
        print('oniro cloned ->', ONIRO_REPO)
    """).strip()))

    cells.append(code(textwrap.dedent("""
        ARC2_DIR = ROOT / 'ARC-AGI-2'
        ARC1_DIR = ROOT / 'ARC-AGI-1'
        if not ARC2_DIR.exists():
            subprocess.check_call(['git','clone','--depth','1',
                'https://github.com/arcprize/ARC-AGI-2.git', str(ARC2_DIR)])
        if not ARC1_DIR.exists():
            subprocess.check_call(['git','clone','--depth','1',
                'https://github.com/fchollet/ARC-AGI.git', str(ARC1_DIR)])
        ARC2_ROOT = str(ARC2_DIR / 'data')
        ARC1_ROOT = str(ARC1_DIR / 'data')
        print('ARC2 train/eval:',
              len(list(Path(ARC2_ROOT,'training').glob('*.json'))), '/',
              len(list(Path(ARC2_ROOT,'evaluation').glob('*.json'))))
        print('ARC1 train/eval:',
              len(list(Path(ARC1_ROOT,'training').glob('*.json'))), '/',
              len(list(Path(ARC1_ROOT,'evaluation').glob('*.json'))))
    """).strip()))

    cells.append(code("!pip -q install einops 2>&1 | tail -2"))

    cells.append(md("## Build URM v35 (~20M params, GQA + KV cache)"))
    cells.append(code(textwrap.dedent("""
        import time, json as _json, random
        import numpy as np
        import torch
        import torch.nn.functional as F

        from oniro.models.urm import URM
        from oniro.models.grid_token_encoder import GridTokenEncoder, GridTokenDecoder
        from oniro.losses.dis import make_dis_targets
        from oniro.losses.grid_ce import grid_ce_loss
        from oniro.orchestrator.alphaevolve_godel import alphaevolve_godel_round, AlphaEvolveGodelArchive
        from oniro.data.arc2_loader import _pairs_from_task
        from oniro.data.sudoku_gen import gen_sudoku_pair
        from oniro.data.math_gen import gen_math_pair
        from oniro.training.grpo import grpo_step, snapshot_policy

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('device:', device)
        if device == 'cpu':
            print('WARNING: GPU not active. Runtime → Change runtime type → T4 GPU')

        # v35 config
        GRID = 30
        D = 768
        N_HEADS = 12
        N_KV_HEADS = 3      # GQA 4x reduction
        FFN = 3584
        N_LOOPS = 12
        N_GROUPS = 2
        N_FORWARD_ONLY = 3
        KV_REFRESH = 2
        N_COLORS = 10
        BATCH = 8           # GRID 30 + D 768 = need smaller batch on T4

        encoder = GridTokenEncoder(grid_size=GRID, n_colors=N_COLORS, d_model=D).to(device)
        urm = URM(d_model=D, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
                  n_loops=N_LOOPS, n_forward_only=N_FORWARD_ONLY,
                  ffn_hidden=FFN, n_groups=N_GROUPS,
                  kv_refresh_every=KV_REFRESH, use_rima=True).to(device)
        decoder = GridTokenDecoder(d_model=D, n_colors=N_COLORS).to(device)

        all_params = list(encoder.parameters()) + list(urm.parameters()) + list(decoder.parameters())
        n_p = sum(p.numel() for p in all_params)
        print(f'URM v35: {n_p/1e6:.2f}M params  (D={D}, h={N_HEADS}/kv={N_KV_HEADS}, '
              f'loops={N_LOOPS}/{N_GROUPS}grp, ffn={FFN}, grid={GRID}, kv_refresh={KV_REFRESH})')

        # GRID 30 padding helper (no nearest-resize -> preserves real grid structure)
        def grid_to_fixed(grid_list, target_side=GRID):
            arr = np.asarray(grid_list, dtype=np.int64)
            if arr.ndim != 2:
                arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
            h, w = arr.shape
            if h > target_side or w > target_side:
                # ARC max grid is 30x30, so this only triggers on sudoku/math at 30+
                arr = arr[:target_side, :target_side]
                h, w = arr.shape
            canvas = np.zeros((target_side, target_side), dtype=np.int64)
            canvas[:h, :w] = arr
            canvas = np.clip(canvas, 0, N_COLORS - 1)
            return torch.from_numpy(canvas).long()
    """).strip()))

    cells.append(md("## Train: ARC heavy (50%) + Sudoku/Math + dihedral aug"))
    cells.append(code(textwrap.dedent("""
        rng = random.Random(0)
        ARC1_FILES = sorted((Path(ARC1_ROOT) / 'training').glob('*.json'))
        ARC2_FILES = sorted((Path(ARC2_ROOT) / 'training').glob('*.json'))
        print(f'ARC1 train tasks: {len(ARC1_FILES)}  ARC2 train tasks: {len(ARC2_FILES)}')

        # cache parsed pairs to avoid repeated JSON parse
        _ARC_CACHE = {}
        def _task_pairs(tf):
            s = str(tf)
            if s not in _ARC_CACHE:
                with tf.open() as f:
                    _ARC_CACHE[s] = _pairs_from_task(_json.load(f))
            return _ARC_CACHE[s]

        def sample_arc(files):
            tf = rng.choice(files)
            return rng.choice(_task_pairs(tf))

        def dihedral_aug(grid_np, k, flip):
            g = np.rot90(grid_np, k=k).copy()
            if flip:
                g = np.flip(g, axis=1).copy()
            return g

        def sample_batch(B, do_aug=True):
            gi_, go_ = [], []
            for _ in range(B):
                r = rng.random()
                # v35 mix: 35% ARC2, 25% ARC1, 5% ARC1-eval, 20% sudoku, 15% math
                if r < 0.35:
                    inp, out = sample_arc(ARC2_FILES)
                    inp = np.asarray(inp, dtype=np.int64)
                    out = np.asarray(out, dtype=np.int64)
                elif r < 0.65:
                    inp, out = sample_arc(ARC1_FILES)
                    inp = np.asarray(inp, dtype=np.int64)
                    out = np.asarray(out, dtype=np.int64)
                elif r < 0.85:
                    inp, out = gen_sudoku_pair(mask_rate=0.4, rng=rng)
                    inp = np.asarray(inp, dtype=np.int64)
                    out = np.asarray(out, dtype=np.int64)
                else:
                    inp, out = gen_math_pair(side=min(GRID, 16), rng=rng)
                    inp = np.asarray(inp, dtype=np.int64)
                    out = np.asarray(out, dtype=np.int64)
                if do_aug:
                    k = rng.randint(0, 3)
                    flip = rng.random() < 0.5
                    inp = dihedral_aug(inp, k, flip)
                    out = dihedral_aug(out, k, flip)
                gi_.append(grid_to_fixed(inp))
                go_.append(grid_to_fixed(out))
            return (torch.stack(gi_).to(device), torch.stack(go_).to(device))

        g_in, g_out = sample_batch(2)
        print('sample shapes:', g_in.shape, g_out.shape)
    """).strip()))

    cells.append(code(textwrap.dedent("""
        STEPS = int(os.environ.get('ONIRO_STEPS', '40000'))
        GRPO_STEPS = int(os.environ.get('ONIRO_GRPO_STEPS', '5000'))
        opt = torch.optim.AdamW(all_params, lr=2e-4, weight_decay=0.05)
        # warmup + cosine
        from torch.optim.lr_scheduler import LambdaLR
        WARMUP = 1000
        def lr_lambda(step):
            if step < WARMUP:
                return step / max(1, WARMUP)
            pct = (step - WARMUP) / max(1, STEPS - WARMUP)
            return 0.5 * (1 + np.cos(np.pi * min(1.0, pct)))
        sched = LambdaLR(opt, lr_lambda=lr_lambda)

        ae_archive = AlphaEvolveGodelArchive()
        AE_EVERY = 2000

        t0 = time.time()
        for step in range(STEPS):
            g_in, g_out = sample_batch(BATCH)
            enc_out = encoder(g_in)
            urm_out = urm(enc_out['tokens'])

            loss = torch.zeros((), device=device)
            n_states = len(urm_out['states_per_loop'])
            dis_targets = make_dis_targets(g_out, n_cycles=n_states - 1,
                                            n_colors=N_COLORS, max_corruption=0.5,
                                            seed=step)
            for t, state in enumerate(urm_out['states_per_loop'][1:]):
                logits = decoder(state, GRID)
                tgt = dis_targets[t].to(device)
                weight = 1.5 ** (-(n_states - 2 - t))
                loss = loss + weight * grid_ce_loss(logits, tgt, bg_weight=0.15)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
            sched.step()

            if step % 100 == 0:
                dt = time.time() - t0
                with torch.no_grad():
                    final_logits = decoder(urm_out['final_state'], GRID)
                    pred = final_logits.argmax(dim=1)
                    cell_acc = (pred == g_out).float().mean().item()
                lr_cur = sched.get_last_lr()[0]
                print(f'step {step:5d}  loss={float(loss.detach()):.4f}  cell_acc={cell_acc:.3f}  '
                      f'lr={lr_cur:.2e}  rate={(step+1)/max(dt,1):.1f}/s')

            if step > 0 and step % AE_EVERY == 0:
                urm.eval()
                eval_in, eval_out = sample_batch(BATCH, do_aug=False)
                def ae_score():
                    with torch.no_grad():
                        e = encoder(eval_in)
                        u = urm(e['tokens'])
                        l = decoder(u['final_state'], GRID)
                        return float((l.argmax(dim=1) == eval_out).float().mean().item())
                acc, base, best, ae_archive = alphaevolve_godel_round(
                    urm, ae_score, n_candidates=3, sigma=2e-3, archive=ae_archive,
                )
                print(f'  [AE] base={base:.4f} best={best:.4f} accept={acc} reject={ae_archive.rejected}')
                urm.train()

        print(f'\\nphase A (supervised) done in {(time.time()-t0)/60:.1f}min')

        # ============= Phase B: GRPO RL =============
        print(f'\\n=== Phase B: GRPO RL ({GRPO_STEPS} steps, group=4) ===')
        ref_enc, ref_urm, ref_dec = snapshot_policy(encoder, urm, decoder)
        for p in ref_enc.parameters(): p.requires_grad = False
        for p in ref_urm.parameters(): p.requires_grad = False
        for p in ref_dec.parameters(): p.requires_grad = False
        ref_enc.to(device); ref_urm.to(device); ref_dec.to(device)

        rl_opt = torch.optim.AdamW(all_params, lr=3e-5, weight_decay=0.05)
        for rl_step in range(GRPO_STEPS):
            g_in_b, g_out_b = sample_batch(BATCH, do_aug=False)
            r = grpo_step(encoder, urm, decoder, rl_opt, g_in_b, g_out_b,
                          ref_enc, ref_urm, ref_dec,
                          n_group=4, eps_clip=0.2, kl_beta=0.04,
                          temperature=1.0, reward_type='cell')
            if rl_step % 100 == 0:
                print(f'  rl_step {rl_step:5d}  reward={r["mean_reward"]:.3f}  '
                      f'max_r={r["max_reward"]:.3f}  kl={r["kl"]:.3f}  loss={r["loss"]:.4f}')
            if rl_step > 0 and rl_step % 500 == 0:
                del ref_enc, ref_urm, ref_dec
                ref_enc, ref_urm, ref_dec = snapshot_policy(encoder, urm, decoder)
                ref_enc.to(device); ref_urm.to(device); ref_dec.to(device)
                print(f'  refreshed reference policy at step {rl_step}')

        print(f'\\nphase B (GRPO RL) done')

        ckpt_dir = ROOT / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({'encoder': encoder.state_dict(), 'urm': urm.state_dict(),
                    'decoder': decoder.state_dict()},
                   str(ckpt_dir / 'urm_v35_final.pt'))
    """).strip()))

    cells.append(md("## DSL Hybrid Solver eval (symbolic + neural fallback)"))
    cells.append(code(textwrap.dedent("""
        from oniro.dsl.solver import solve_task as dsl_solve_task

        def neural_predict_np(grid_np):
            gi = grid_to_fixed(grid_np.tolist()).unsqueeze(0).to(device)
            with torch.no_grad():
                e = encoder(gi)
                u = urm(e['tokens'])
                l = decoder(u['final_state'], GRID)
                return l.argmax(dim=1)[0].cpu().numpy().astype(np.int8)

        def eval_arc_hybrid(root, label, max_dsl_depth=3):
            sd = Path(root) / 'evaluation'
            files = sorted(sd.glob('*.json'))
            n_pe = 0; n_t = 0; cells_c = []; ts = 0; tt = 0
            n_dsl_solved = 0
            for ti, tf in enumerate(files):
                with tf.open() as f:
                    task = _json.load(f)
                res = dsl_solve_task(task, neural_fallback=neural_predict_np, max_depth=max_dsl_depth)
                solved = []
                for i, tp in enumerate(task.get('test', [])):
                    if 'output' not in tp: continue
                    gt = np.asarray(tp['output'], dtype=np.int8)
                    pred = res['predictions'][i] if i < len(res['predictions']) else None
                    if pred is None:
                        pred = neural_predict_np(np.asarray(tp['input'], dtype=np.int8))
                    if pred.shape == gt.shape:
                        exact = bool(np.array_equal(pred, gt))
                        cell_acc = float((pred == gt).mean())
                    else:
                        exact = False; cell_acc = 0.0
                    n_t += 1
                    if exact:
                        n_pe += 1
                        if res['method'] == 'dsl':
                            n_dsl_solved += 1
                    cells_c.append(cell_acc)
                    solved.append(exact)
                if solved:
                    tt += 1
                    if all(solved): ts += 1
                if (ti + 1) % 50 == 0:
                    print(f'  {label} [{ti+1}/{len(files)}] exact={n_pe}/{n_t}  dsl_solved={n_dsl_solved}')
            return {'label': label, 'pairs_total': n_t, 'pairs_exact': n_pe,
                    'pair_exact_acc': n_pe / max(n_t, 1),
                    'mean_cell_acc': sum(cells_c)/max(len(cells_c), 1),
                    'tasks_total': tt, 'tasks_solved': ts,
                    'task_acc': ts / max(tt, 1),
                    'dsl_solved_pairs': n_dsl_solved}

        print('=== ARC-AGI-1 hybrid (DSL depth=3 + neural) ===')
        r1_hybrid = eval_arc_hybrid(ARC1_ROOT, 'ARC-1-hybrid', max_dsl_depth=3)
        print(_json.dumps(r1_hybrid, indent=2))
        print('=== ARC-AGI-2 hybrid (DSL depth=3 + neural) ===')
        r2_hybrid = eval_arc_hybrid(ARC2_ROOT, 'ARC-2-hybrid', max_dsl_depth=3)
        print(_json.dumps(r2_hybrid, indent=2))
        with open(str(ROOT / 'eval_hybrid.json'), 'w') as f:
            _json.dump({'arc1': r1_hybrid, 'arc2': r2_hybrid}, f, indent=2)
    """).strip()))

    cells.append(md("## Eval: ARC-1, ARC-2 (TTFT + AIRV), Sudoku, Math"))
    cells.append(code(textwrap.dedent("""
        from oniro.eval.ttft_urm import ttft_finetune_urm, restore_urm, airv_predict
        encoder.eval(); urm.eval(); decoder.eval()

        @torch.no_grad()
        def predict(grid_input_t):
            e = encoder(grid_input_t)
            u = urm(e['tokens'])
            l = decoder(u['final_state'], GRID)
            return l.argmax(dim=1)[0]

        TTFT_STEPS = int(os.environ.get('ONIRO_TTFT_STEPS', '100'))
        TTFT_LR = float(os.environ.get('ONIRO_TTFT_LR', '1e-4'))
        USE_AIRV = os.environ.get('ONIRO_AIRV', '1') == '1'

        def predict_with_ttft_airv(task, test_input_grid_int):
            demos = []
            for tp in task.get('train', []):
                di = grid_to_fixed(tp['input']).unsqueeze(0).to(device)
                do = grid_to_fixed(tp['output']).unsqueeze(0).to(device)
                demos.append((di, do))
            snap = ttft_finetune_urm(encoder, urm, decoder, demos,
                                     grid_size=GRID, n_steps=TTFT_STEPS,
                                     lr=TTFT_LR, device=device)
            if USE_AIRV:
                pred = airv_predict(encoder, urm, decoder,
                                     test_input_grid_int.unsqueeze(0),
                                     grid_size=GRID, n_colors=N_COLORS)
            else:
                pred = predict(test_input_grid_int.unsqueeze(0))
            restore_urm(encoder, urm, decoder, snap)
            return pred

        def eval_arc(root, label):
            sd = Path(root) / 'evaluation'
            files = sorted(sd.glob('*.json'))
            n_pe = 0; n_t = 0; cells_c = []; ts = 0; tt = 0
            for ti, tf in enumerate(files):
                with tf.open() as f:
                    task = _json.load(f)
                solved = []
                for tp in task.get('test', []):
                    if 'output' not in tp: continue
                    gi = grid_to_fixed(tp['input']).to(device)
                    gt = grid_to_fixed(tp['output']).to(device)
                    pred = predict_with_ttft_airv(task, gi)
                    exact = bool((pred == gt).all().item())
                    n_t += 1
                    if exact: n_pe += 1
                    cells_c.append(float((pred == gt).float().mean().item()))
                    solved.append(exact)
                if solved:
                    tt += 1
                    if all(solved): ts += 1
                if (ti + 1) % 30 == 0:
                    print(f'  {label} [{ti+1}/{len(files)}] exact={n_pe}/{n_t}')
            return {'label': label, 'pairs_total': n_t, 'pairs_exact': n_pe,
                    'pair_exact_acc': n_pe / max(n_t, 1),
                    'mean_cell_acc': sum(cells_c)/max(len(cells_c), 1),
                    'tasks_total': tt, 'tasks_solved': ts,
                    'task_acc': ts / max(tt, 1),
                    'ttft_steps': TTFT_STEPS, 'airv': USE_AIRV}

        results = {}
        print('=== ARC-AGI-1 ===')
        results['arc1'] = eval_arc(ARC1_ROOT, 'ARC-1'); print(_json.dumps(results['arc1'], indent=2))
        print('=== ARC-AGI-2 ===')
        results['arc2'] = eval_arc(ARC2_ROOT, 'ARC-2'); print(_json.dumps(results['arc2'], indent=2))

        def eval_proc(gen_fn, label, n=100):
            n_pe = 0; cell_c = []
            for _ in range(n):
                inp, out = gen_fn(rng=rng)
                gi = grid_to_fixed(inp.tolist() if hasattr(inp, 'tolist') else inp).unsqueeze(0).to(device)
                gt = grid_to_fixed(out.tolist() if hasattr(out, 'tolist') else out).to(device)
                pred = predict(gi)
                exact = bool((pred == gt).all().item())
                if exact: n_pe += 1
                cell_c.append(float((pred == gt).float().mean().item()))
            return {'label': label, 'n_samples': n, 'pair_exact_acc': n_pe / n,
                    'mean_cell_acc': sum(cell_c)/len(cell_c)}

        print('=== Sudoku ===')
        results['sudoku'] = eval_proc(lambda rng=rng: gen_sudoku_pair(mask_rate=0.4, rng=rng), 'Sudoku', n=100)
        print(_json.dumps(results['sudoku'], indent=2))

        print('=== Math (procedural) ===')
        results['math'] = eval_proc(lambda rng=rng: gen_math_pair(side=min(GRID, 16), rng=rng), 'Math', n=100)
        print(_json.dumps(results['math'], indent=2))

        out_path = ROOT / 'eval_all.json'
        with out_path.open('w') as f:
            _json.dump(results, f, indent=2)

        print('\\n=== FINAL v35 ===')
        for k, r in results.items():
            ea = r.get('pair_exact_acc', 0) * 100
            ca = r.get('mean_cell_acc', 0) * 100
            print(f'{k:8s}  pair_exact={ea:.2f}%  cell_acc={ca:.1f}%')
        print('AE-Godel:', _json.dumps(ae_archive.summary(), indent=2))
    """).strip()))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT} — {OUT.stat().st_size//1024} KB, {len(cells)} cells")


if __name__ == "__main__":
    main()
