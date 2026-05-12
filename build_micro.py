"""Build ONIRO v36 Colab notebook.

v36 changes vs v35:
    - KV cache invalidation at no_grad/grad boundary (correctness fix)
    - EMA (exponential moving average) on params - URM paper essential trick
    - Color permutation augmentation - NVARC SOTA pattern (10! perms)
    - Text head (byte-level) for natural-language Q/A about math problems
    - Expanded math suite: 6 generators (arith/seq/compare/parity/mod/linear)
    - 1000-sample TTA majority vote at ARC eval (TRM: +11pp vs single-pass)
    - Joint loss: grid_ce + 0.3 * text_ce

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
        # ONIRO v36 — URM 20M + Text Head + EMA + Color Perm Aug

        **Setup:** Runtime → Change runtime type → **T4 GPU** → Save → Run All.

        Pulls source from https://github.com/PAMF2/oniro-colab (public).

        Upgrades over v35:
        - **KV cache no_grad/grad boundary fix** (correctness)
        - **EMA params** (URM paper essential — stability)
        - **Color permutation aug** (NVARC SOTA pattern: 10! perms)
        - **Text head** (byte-level Q/A on math problems)
        - **Math suite expanded**: arith / seq / compare / parity / mod / linear eq
        - **1000-sample TTA majority vote** at ARC eval (TRM +11pp)
        - **Joint loss**: grid_ce + 0.3 * text_ce

        Phase A: supervised (40k) | Phase B: GRPO (5k) | TTFT eval

        Runtime ~6-8h on Colab T4.
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

    cells.append(md("## Build URM v36 (~22M params: 20M URM + 2M text head)"))
    cells.append(code(textwrap.dedent("""
        import time, json as _json, random
        import numpy as np
        import torch
        import torch.nn.functional as F

        from oniro.models.urm import URM
        from oniro.models.grid_token_encoder import GridTokenEncoder, GridTokenDecoder
        from oniro.models.text_head import TextHead, text_to_bytes, bytes_to_text, BYTE_VOCAB, PAD
        from oniro.losses.dis import make_dis_targets
        from oniro.losses.grid_ce import grid_ce_loss
        from oniro.orchestrator.alphaevolve_godel import alphaevolve_godel_round, AlphaEvolveGodelArchive
        from oniro.data.arc2_loader import _pairs_from_task
        from oniro.data.sudoku_gen import gen_sudoku_pair
        from oniro.data.math_gen import gen_math_pair
        from oniro.data.math_text import gen_math_text_pair
        from oniro.data.color_perm import random_color_perm, apply_color_perm
        from oniro.training.grpo import grpo_step, snapshot_policy
        from oniro.training.ema import EMA

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('device:', device)
        if device == 'cpu':
            print('WARNING: GPU not active. Runtime → Change runtime type → T4 GPU')

        # v36 config
        GRID = 30
        D = 768
        N_HEADS = 12
        N_KV_HEADS = 3
        FFN = 3584
        N_LOOPS = 12
        N_GROUPS = 2
        N_FORWARD_ONLY = 3
        KV_REFRESH = 2
        N_COLORS = 10
        BATCH = 8

        # text head
        TEXT_D = 192
        TEXT_LAYERS = 3
        TEXT_HEADS = 4
        TEXT_MAX_LEN = 48

        encoder = GridTokenEncoder(grid_size=GRID, n_colors=N_COLORS, d_model=D).to(device)
        urm = URM(d_model=D, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
                  n_loops=N_LOOPS, n_forward_only=N_FORWARD_ONLY,
                  ffn_hidden=FFN, n_groups=N_GROUPS,
                  kv_refresh_every=KV_REFRESH, use_rima=True).to(device)
        decoder = GridTokenDecoder(d_model=D, n_colors=N_COLORS).to(device)
        text_proj = torch.nn.Linear(D, TEXT_D).to(device)
        text_head = TextHead(d_model=TEXT_D, n_layers=TEXT_LAYERS,
                             n_heads=TEXT_HEADS, max_len=TEXT_MAX_LEN).to(device)

        all_params = (list(encoder.parameters()) + list(urm.parameters())
                      + list(decoder.parameters())
                      + list(text_proj.parameters()) + list(text_head.parameters()))
        n_p = sum(p.numel() for p in all_params)
        print(f'ONIRO v36: {n_p/1e6:.2f}M params (URM trunk + text head)')
        print(f'  URM: D={D}, h={N_HEADS}/kv={N_KV_HEADS}, loops={N_LOOPS}/{N_GROUPS}grp, '
              f'ffn={FFN}, grid={GRID}, kv_refresh={KV_REFRESH}')
        print(f'  Text: D={TEXT_D}, layers={TEXT_LAYERS}, byte-vocab=256, max_len={TEXT_MAX_LEN}')

        def grid_to_fixed(grid_list, target_side=GRID):
            arr = np.asarray(grid_list, dtype=np.int64)
            if arr.ndim != 2:
                arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
            h, w = arr.shape
            if h > target_side or w > target_side:
                arr = arr[:target_side, :target_side]
                h, w = arr.shape
            canvas = np.zeros((target_side, target_side), dtype=np.int64)
            canvas[:h, :w] = arr
            canvas = np.clip(canvas, 0, N_COLORS - 1)
            return torch.from_numpy(canvas).long()

        # EMA shadow params - URM paper essential for stability
        ema = EMA([encoder, urm, decoder, text_proj, text_head], decay=0.999)
    """).strip()))

    cells.append(md("## Train: ARC heavy + Sudoku + Math + Text Q/A"))
    cells.append(code(textwrap.dedent("""
        rng = random.Random(0)
        ARC1_FILES = sorted((Path(ARC1_ROOT) / 'training').glob('*.json'))
        ARC2_FILES = sorted((Path(ARC2_ROOT) / 'training').glob('*.json'))
        print(f'ARC1 train tasks: {len(ARC1_FILES)}  ARC2 train tasks: {len(ARC2_FILES)}')

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
            '''Sample mixed batch. Each item has grid_in, grid_out, and optionally text Q/A.'''
            gi_, go_, text_q, text_a, has_text = [], [], [], [], []
            for _ in range(B):
                r = rng.random()
                _text_q, _text_a = '', ''
                _has_text = False
                if r < 0.30:
                    inp, out = sample_arc(ARC2_FILES)
                    inp = np.asarray(inp, dtype=np.int64); out = np.asarray(out, dtype=np.int64)
                elif r < 0.55:
                    inp, out = sample_arc(ARC1_FILES)
                    inp = np.asarray(inp, dtype=np.int64); out = np.asarray(out, dtype=np.int64)
                elif r < 0.70:
                    inp, out = gen_sudoku_pair(mask_rate=0.4, rng=rng)
                    inp = np.asarray(inp, dtype=np.int64); out = np.asarray(out, dtype=np.int64)
                elif r < 0.80:
                    inp, out = gen_math_pair(side=min(GRID, 16), rng=rng)
                    inp = np.asarray(inp, dtype=np.int64); out = np.asarray(out, dtype=np.int64)
                else:
                    # math text Q/A
                    gi, go, q, a = gen_math_text_pair(rng=rng, side=min(GRID, 16))
                    inp = np.asarray(gi, dtype=np.int64); out = np.asarray(go, dtype=np.int64)
                    _text_q, _text_a = q, a
                    _has_text = True
                if do_aug:
                    k = rng.randint(0, 3)
                    flip = rng.random() < 0.5
                    inp = dihedral_aug(inp, k, flip)
                    out = dihedral_aug(out, k, flip)
                    # Color permutation aug (NVARC SOTA) - keep_bg=True so the
                    # padding/background color 0 stays semantically stable.
                    if rng.random() < 0.7:
                        cp = random_color_perm(rng, n_colors=N_COLORS, keep_bg=True)
                        inp = apply_color_perm(inp, cp)
                        out = apply_color_perm(out, cp)
                gi_.append(grid_to_fixed(inp))
                go_.append(grid_to_fixed(out))
                text_q.append(_text_q); text_a.append(_text_a); has_text.append(_has_text)
            g_in_t = torch.stack(gi_).to(device)
            g_out_t = torch.stack(go_).to(device)
            return g_in_t, g_out_t, text_q, text_a, has_text

        g_in, g_out, tq, ta, ht = sample_batch(4)
        print('sample shapes:', g_in.shape, g_out.shape, 'text items:', sum(ht))
    """).strip()))

    cells.append(code(textwrap.dedent("""
        STEPS = int(os.environ.get('ONIRO_STEPS', '40000'))
        GRPO_STEPS = int(os.environ.get('ONIRO_GRPO_STEPS', '5000'))
        TEXT_LOSS_W = float(os.environ.get('ONIRO_TEXT_LOSS_W', '0.3'))
        opt = torch.optim.AdamW(all_params, lr=2e-4, weight_decay=0.05)

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

        def text_supervision_loss(memory, q_list, a_list, has_text_list):
            '''Cross-entropy on byte-level text answer, conditioned on URM state.'''
            idxs = [i for i, h in enumerate(has_text_list) if h]
            if not idxs:
                return torch.zeros((), device=device)
            mem = text_proj(memory[idxs])
            tok_in = torch.stack([text_to_bytes(f'{q_list[i]} {a_list[i]}',
                                                max_len=TEXT_MAX_LEN) for i in idxs]).to(device)
            logits = text_head(mem, tok_in[:, :-1])
            tgt = tok_in[:, 1:]
            mask = (tgt != PAD).float()
            logp = F.log_softmax(logits, dim=-1)
            chosen = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            return -(chosen * mask).sum() / max(mask.sum(), 1.0)

        t0 = time.time()
        for step in range(STEPS):
            g_in, g_out, tq, ta, ht = sample_batch(BATCH)
            enc_out = encoder(g_in)
            urm_out = urm(enc_out['tokens'])

            # Grid DIS loss
            loss_grid = torch.zeros((), device=device)
            n_states = len(urm_out['states_per_loop'])
            dis_targets = make_dis_targets(g_out, n_cycles=n_states - 1,
                                            n_colors=N_COLORS, max_corruption=0.5,
                                            seed=step)
            for t, state in enumerate(urm_out['states_per_loop'][1:]):
                logits = decoder(state, GRID)
                tgt = dis_targets[t].to(device)
                weight = 1.5 ** (-(n_states - 2 - t))
                loss_grid = loss_grid + weight * grid_ce_loss(logits, tgt, bg_weight=0.15)

            # Text Q/A loss (conditioned on URM final state)
            loss_text = text_supervision_loss(urm_out['final_state'], tq, ta, ht)
            loss = loss_grid + TEXT_LOSS_W * loss_text

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
            sched.step()
            ema.update()

            if step % 100 == 0:
                dt = time.time() - t0
                with torch.no_grad():
                    final_logits = decoder(urm_out['final_state'], GRID)
                    pred = final_logits.argmax(dim=1)
                    cell_acc = (pred == g_out).float().mean().item()
                lr_cur = sched.get_last_lr()[0]
                print(f'step {step:5d}  loss={float(loss.detach()):.4f}  '
                      f'grid={float(loss_grid.detach()):.3f}  txt={float(loss_text):.3f}  '
                      f'cell_acc={cell_acc:.3f}  lr={lr_cur:.2e}  '
                      f'rate={(step+1)/max(dt,1):.1f}/s')

            if step > 0 and step % AE_EVERY == 0:
                # AE-Godel mutates live weights. Do NOT wrap with ema.swap_in()
                # here - swap_in restores backup on exit and would discard the
                # accepted mutation. AE runs on live, EMA shadow keeps tracking.
                encoder.eval(); urm.eval(); decoder.eval()
                eval_in, eval_out, _, _, _ = sample_batch(BATCH, do_aug=False)
                def ae_score():
                    with torch.no_grad():
                        e = encoder(eval_in); u = urm(e['tokens'])
                        l = decoder(u['final_state'], GRID)
                        return float((l.argmax(dim=1) == eval_out).float().mean().item())
                acc, base, best, ae_archive = alphaevolve_godel_round(
                    urm, ae_score, n_candidates=3, sigma=2e-3, archive=ae_archive,
                )
                print(f'  [AE] base={base:.4f} best={best:.4f} accept={acc} reject={ae_archive.rejected}')
                encoder.train(); urm.train(); decoder.train()

        print(f'\\nphase A (supervised+text) done in {(time.time()-t0)/60:.1f}min')

        # ============= Phase B: GRPO RL =============
        print(f'\\n=== Phase B: GRPO RL ({GRPO_STEPS} steps, group=4) ===')
        ref_enc, ref_urm, ref_dec = snapshot_policy(encoder, urm, decoder)
        for p in ref_enc.parameters(): p.requires_grad = False
        for p in ref_urm.parameters(): p.requires_grad = False
        for p in ref_dec.parameters(): p.requires_grad = False
        ref_enc.to(device); ref_urm.to(device); ref_dec.to(device)

        # RL phase only touches grid trunk. Excluding text_proj/text_head so
        # AdamW weight_decay does not erode them with zero gradient.
        rl_trunk_params = (list(encoder.parameters()) + list(urm.parameters())
                            + list(decoder.parameters()))
        rl_opt = torch.optim.AdamW(rl_trunk_params, lr=3e-5, weight_decay=0.05)
        for rl_step in range(GRPO_STEPS):
            g_in_b, g_out_b, _, _, _ = sample_batch(BATCH, do_aug=False)
            r = grpo_step(encoder, urm, decoder, rl_opt, g_in_b, g_out_b,
                          ref_enc, ref_urm, ref_dec,
                          n_group=4, eps_clip=0.2, kl_beta=0.04,
                          temperature=1.0, reward_type='cell')
            ema.update()
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
                    'decoder': decoder.state_dict(),
                    'text_proj': text_proj.state_dict(),
                    'text_head': text_head.state_dict()},
                   str(ckpt_dir / 'urm_v36_final.pt'))
    """).strip()))

    cells.append(md("## ARC eval — 1000-sample TTA majority vote (TRM SOTA pattern)"))
    cells.append(code(textwrap.dedent("""
        encoder.eval(); urm.eval(); decoder.eval()

        N_TTA = int(os.environ.get('ONIRO_TTA', '64'))   # T4 mem -> 64, paper uses 1000
        # Augmentation pool: 8 dihedrals × random color perms

        @torch.no_grad()
        def tta_majority_vote(grid_int_t, n_samples=N_TTA):
            '''Predict via N augmented passes, majority vote per pixel.'''
            H, W = grid_int_t.shape
            votes = torch.zeros(N_COLORS, GRID, GRID, device=device)
            gnp = grid_int_t.cpu().numpy()
            for s in range(n_samples):
                k = rng.randint(0, 3); flip = rng.random() < 0.5
                # keep_bg=True preserves color 0 (matches train-time aug + padding semantics)
                cp = random_color_perm(rng, n_colors=N_COLORS, keep_bg=True)
                inv_cp = np.argsort(cp).astype(np.int64)
                gaug = dihedral_aug(gnp, k, flip)
                gaug = apply_color_perm(gaug, cp)
                t = grid_to_fixed(gaug).unsqueeze(0).to(device)
                e = encoder(t); u = urm(e['tokens'])
                l = decoder(u['final_state'], GRID)
                pred = l.argmax(dim=1)[0]
                # invert color perm
                pred_np = pred.cpu().numpy()
                pred_np = apply_color_perm(pred_np.astype(np.int64), inv_cp)
                pred = torch.from_numpy(pred_np).to(device)
                # invert dihedral
                if flip:
                    pred = torch.flip(pred, dims=(-1,))
                pred = torch.rot90(pred, k=-k, dims=(-2, -1))
                for c in range(N_COLORS):
                    votes[c] += (pred == c).float()
            return votes.argmax(dim=0)

        def eval_arc_tta(root, label):
            sd = Path(root) / 'evaluation'
            files = sorted(sd.glob('*.json'))
            n_pe = 0; n_t = 0; cells_c = []; ts = 0; tt = 0
            with ema.swap_in():
                for ti, tf in enumerate(files):
                    with tf.open() as f:
                        task = _json.load(f)
                    solved = []
                    for tp in task.get('test', []):
                        if 'output' not in tp: continue
                        gi = grid_to_fixed(tp['input']).to(device)
                        gt = grid_to_fixed(tp['output']).to(device)
                        pred = tta_majority_vote(gi)
                        exact = bool((pred == gt).all().item())
                        n_t += 1
                        if exact: n_pe += 1
                        cells_c.append(float((pred == gt).float().mean().item()))
                        solved.append(exact)
                    if solved:
                        tt += 1
                        if all(solved): ts += 1
                    if (ti + 1) % 30 == 0:
                        print(f'  {label} TTA{N_TTA} [{ti+1}/{len(files)}] exact={n_pe}/{n_t}')
            return {'label': label, 'pairs_total': n_t, 'pairs_exact': n_pe,
                    'pair_exact_acc': n_pe / max(n_t, 1),
                    'mean_cell_acc': sum(cells_c)/max(len(cells_c), 1),
                    'tasks_total': tt, 'tasks_solved': ts,
                    'task_acc': ts / max(tt, 1),
                    'tta_samples': N_TTA}

        results = {}
        print(f'=== ARC-AGI-1 (TTA {N_TTA} samples) ===')
        results['arc1'] = eval_arc_tta(ARC1_ROOT, 'ARC-1')
        print(_json.dumps(results['arc1'], indent=2))
        print(f'=== ARC-AGI-2 (TTA {N_TTA} samples) ===')
        results['arc2'] = eval_arc_tta(ARC2_ROOT, 'ARC-2')
        print(_json.dumps(results['arc2'], indent=2))
    """).strip()))

    cells.append(md("## DSL Hybrid (symbolic + neural)"))
    cells.append(code(textwrap.dedent("""
        from oniro.dsl.solver import solve_task as dsl_solve_task

        @torch.no_grad()
        def neural_predict_np(grid_np):
            gi = grid_to_fixed(grid_np.tolist()).unsqueeze(0).to(device)
            e = encoder(gi); u = urm(e['tokens'])
            l = decoder(u['final_state'], GRID)
            return l.argmax(dim=1)[0].cpu().numpy().astype(np.int8)

        def eval_arc_hybrid(root, label, max_dsl_depth=3):
            sd = Path(root) / 'evaluation'
            files = sorted(sd.glob('*.json'))
            n_pe = 0; n_t = 0; cells_c = []; ts = 0; tt = 0; n_dsl_solved = 0
            with ema.swap_in():
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
                            if res['method'] == 'dsl': n_dsl_solved += 1
                        cells_c.append(cell_acc); solved.append(exact)
                    if solved:
                        tt += 1
                        if all(solved): ts += 1
                    if (ti + 1) % 50 == 0:
                        print(f'  {label}-hybrid [{ti+1}/{len(files)}] exact={n_pe}/{n_t}  dsl_solved={n_dsl_solved}')
            return {'label': label+'-hybrid', 'pairs_total': n_t, 'pairs_exact': n_pe,
                    'pair_exact_acc': n_pe / max(n_t, 1),
                    'mean_cell_acc': sum(cells_c)/max(len(cells_c), 1),
                    'tasks_total': tt, 'tasks_solved': ts,
                    'task_acc': ts / max(tt, 1), 'dsl_solved_pairs': n_dsl_solved}

        print('=== ARC-AGI-1 hybrid (DSL depth=3 + neural) ===')
        results['arc1_hybrid'] = eval_arc_hybrid(ARC1_ROOT, 'ARC-1', max_dsl_depth=3)
        print(_json.dumps(results['arc1_hybrid'], indent=2))
        print('=== ARC-AGI-2 hybrid ===')
        results['arc2_hybrid'] = eval_arc_hybrid(ARC2_ROOT, 'ARC-2', max_dsl_depth=3)
        print(_json.dumps(results['arc2_hybrid'], indent=2))
    """).strip()))

    cells.append(md("## Math text eval — model answers in natural language"))
    cells.append(code(textwrap.dedent("""
        @torch.no_grad()
        def answer_math(question_text, grid_in_np, max_new=12):
            '''Prime decoder with question prefix, then sample continuation.

            Training format: "BOS q_text<SPACE>a_text EOS". So at inference we
            feed [BOS, q_bytes..., SPACE] and sample the next bytes which are
            the answer. Strip the prefix from the generated bytes.
            '''
            gi = grid_to_fixed(grid_in_np.tolist()).unsqueeze(0).to(device)
            e = encoder(gi); u = urm(e['tokens'])
            mem = text_proj(u['final_state'])
            # build prefix: BOS + question bytes + ' '   (mirrors training format)
            from oniro.models.text_head import BOS, EOS
            prefix_ids = [BOS] + list(question_text.encode('utf-8')) + [ord(' ')]
            prefix_ids = prefix_ids[:TEXT_MAX_LEN - max_new]
            seq = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            for _ in range(max_new):
                logits = text_head(mem, seq[:, -TEXT_MAX_LEN:])
                nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
                seq = torch.cat([seq, nxt], dim=1)
                if (nxt == EOS).all():
                    break
            # decode only the answer suffix (after the prefix)
            answer_ids = seq[0, len(prefix_ids):]
            return bytes_to_text(answer_ids)

        def eval_math_text(n=200):
            n_correct = 0
            examples = []
            with ema.swap_in():
                encoder.eval(); urm.eval(); text_head.eval(); text_proj.eval()
                for i in range(n):
                    gi, go, q, a = gen_math_text_pair(rng=rng, side=min(GRID, 16))
                    pred_full = answer_math(q, np.asarray(gi, dtype=np.int64))
                    # normalise: strip whitespace/punctuation, take first token
                    pred = pred_full.strip().split()[0] if pred_full.strip() else ''
                    pred = pred.strip().rstrip('.,?!').lower()
                    ok = pred == a.strip().lower()
                    n_correct += int(ok)
                    if i < 6:
                        examples.append(f'  Q: {q!r}  A: {a!r}  pred: {pred_full!r}  ok={ok}')
            return n_correct / n, examples

        print('=== Math Text Q/A eval ===')
        text_acc, ex = eval_math_text(n=200)
        print(f'text answer accuracy: {text_acc*100:.1f}%')
        for line in ex: print(line)
        results['math_text'] = {'accuracy': text_acc, 'n': 200}
    """).strip()))

    cells.append(md("## Sudoku + Math grid eval (procedural)"))
    cells.append(code(textwrap.dedent("""
        @torch.no_grad()
        def predict_grid(grid_input_t):
            e = encoder(grid_input_t); u = urm(e['tokens'])
            l = decoder(u['final_state'], GRID)
            return l.argmax(dim=1)[0]

        def eval_proc(gen_fn, label, n=100):
            n_pe = 0; cell_c = []
            with ema.swap_in():
                for _ in range(n):
                    inp, out = gen_fn(rng=rng)
                    gi = grid_to_fixed(inp.tolist() if hasattr(inp, 'tolist') else inp).unsqueeze(0).to(device)
                    gt = grid_to_fixed(out.tolist() if hasattr(out, 'tolist') else out).to(device)
                    pred = predict_grid(gi)
                    exact = bool((pred == gt).all().item())
                    if exact: n_pe += 1
                    cell_c.append(float((pred == gt).float().mean().item()))
            return {'label': label, 'n_samples': n, 'pair_exact_acc': n_pe / n,
                    'mean_cell_acc': sum(cell_c)/len(cell_c)}

        print('=== Sudoku ===')
        results['sudoku'] = eval_proc(lambda rng=rng: gen_sudoku_pair(mask_rate=0.4, rng=rng), 'Sudoku', n=100)
        print(_json.dumps(results['sudoku'], indent=2))
        print('=== Math grid ===')
        results['math_grid'] = eval_proc(lambda rng=rng: gen_math_pair(side=min(GRID, 16), rng=rng), 'Math', n=100)
        print(_json.dumps(results['math_grid'], indent=2))

        with open(str(ROOT / 'eval_v36.json'), 'w') as f:
            _json.dump(results, f, indent=2)

        print('\\n=== FINAL v36 ===')
        for k, r in results.items():
            if 'accuracy' in r:
                print(f'{k:14s}  acc={r["accuracy"]*100:.1f}%')
            elif 'task_acc' in r:
                print(f'{k:14s}  task_acc={r["task_acc"]*100:.2f}%  cell={r["mean_cell_acc"]*100:.1f}%')
            elif 'pair_exact_acc' in r:
                print(f'{k:14s}  pair_exact={r["pair_exact_acc"]*100:.1f}%  cell={r["mean_cell_acc"]*100:.1f}%')
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
