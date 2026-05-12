"""Build ONIRO MICRO Colab notebook (public repo edition).

Clones https://github.com/PAMF2/oniro-colab to get source.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

OUT = Path(__file__).parent / "oniro_colab_micro.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.split("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.split("\n")}


def main() -> None:
    cells = []
    cells.append(md(textwrap.dedent("""
        # ONIRO MICRO — URM 2.5M on Colab

        **Setup:** Runtime → Change runtime type → **T4 GPU** → Save → Run All.

        Pulls source from https://github.com/PAMF2/oniro-colab (public).
        Clones ARC-1 / ARC-2. Trains URM + RIMA + DIS + AE-Gödel.
        Evaluates on ARC-AGI-1, ARC-AGI-2, Sudoku, Math.
        Runtime ~1-2h on Colab free T4.
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

    cells.append(md("## Build URM MICRO (~2.5M params)"))
    cells.append(code(textwrap.dedent("""
        import time, json as _json, random
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

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('device:', device)
        if device == 'cpu':
            print('WARNING: GPU not active. Runtime → Change runtime type → T4 GPU')

        GRID = 12
        D = 96
        N_HEADS = 4
        N_LOOPS = 6
        N_FORWARD_ONLY = 1
        N_COLORS = 10
        BATCH = 32

        encoder = GridTokenEncoder(grid_size=GRID, n_colors=N_COLORS, d_model=D).to(device)
        urm = URM(d_model=D, n_heads=N_HEADS, n_loops=N_LOOPS,
                  n_forward_only=N_FORWARD_ONLY, use_rima=True).to(device)
        decoder = GridTokenDecoder(d_model=D, n_colors=N_COLORS).to(device)

        all_params = list(encoder.parameters()) + list(urm.parameters()) + list(decoder.parameters())
        n_p = sum(p.numel() for p in all_params)
        print(f'URM MICRO: {n_p/1e6:.2f}M params  (D={D}, loops={N_LOOPS}, grid={GRID})')

        def grid_to_fixed(grid_list, target_side=GRID):
            import numpy as np
            arr = np.asarray(grid_list, dtype=np.int64)
            if arr.ndim != 2:
                arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
            h, w = arr.shape
            side = max(h, w, 1)
            canvas = np.zeros((side, side), dtype=np.int64)
            canvas[:h, :w] = arr
            canvas = np.clip(canvas, 0, N_COLORS - 1)
            t = torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0).float()
            t = F.interpolate(t, size=(target_side, target_side), mode='nearest')
            return t.squeeze(0).squeeze(0).long()
    """).strip()))

    cells.append(md("## Train: ARC-1 + ARC-2 + Sudoku + Math interleaved"))
    cells.append(code(textwrap.dedent("""
        rng = random.Random(0)
        ARC1_FILES = sorted((Path(ARC1_ROOT) / 'training').glob('*.json'))
        ARC2_FILES = sorted((Path(ARC2_ROOT) / 'training').glob('*.json'))

        def sample_arc(files):
            tf = rng.choice(files)
            with tf.open() as f:
                task = _json.load(f)
            pairs = _pairs_from_task(task)
            return rng.choice(pairs)

        def sample_batch(B):
            gi_, go_ = [], []
            for _ in range(B):
                r = rng.random()
                if r < 0.4:
                    inp, out = sample_arc(ARC2_FILES)
                elif r < 0.65:
                    inp, out = sample_arc(ARC1_FILES)
                elif r < 0.85:
                    inp, out = gen_sudoku_pair(mask_rate=0.4, rng=rng)
                    inp, out = inp.tolist(), out.tolist()
                else:
                    inp, out = gen_math_pair(side=GRID, rng=rng)
                    inp, out = inp.tolist(), out.tolist()
                gi_.append(grid_to_fixed(inp))
                go_.append(grid_to_fixed(out))
            return (torch.stack(gi_).to(device), torch.stack(go_).to(device))

        g_in, g_out = sample_batch(2)
        print('sample shapes:', g_in.shape, g_out.shape)
    """).strip()))

    cells.append(code(textwrap.dedent("""
        STEPS = int(os.environ.get('ONIRO_STEPS', '5000'))
        opt = torch.optim.AdamW(all_params, lr=3e-4, weight_decay=0.05)
        ae_archive = AlphaEvolveGodelArchive()
        AE_EVERY = 1000

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

            if step % 100 == 0:
                dt = time.time() - t0
                with torch.no_grad():
                    final_logits = decoder(urm_out['final_state'], GRID)
                    pred = final_logits.argmax(dim=1)
                    cell_acc = (pred == g_out).float().mean().item()
                print(f'step {step:5d}  loss={float(loss):.4f}  cell_acc={cell_acc:.3f}  rate={(step+1)/max(dt,1):.1f}/s')

            if step > 0 and step % AE_EVERY == 0:
                urm.eval()
                eval_in, eval_out = sample_batch(BATCH)
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

        print(f'\\ntrained in {(time.time()-t0)/60:.1f}min')
        ckpt_dir = ROOT / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({'encoder': encoder.state_dict(), 'urm': urm.state_dict(),
                    'decoder': decoder.state_dict()},
                   str(ckpt_dir / 'urm_micro_final.pt'))
    """).strip()))

    cells.append(md("## Eval: ARC-1, ARC-2, Sudoku, Math"))
    cells.append(code(textwrap.dedent("""
        encoder.eval(); urm.eval(); decoder.eval()

        @torch.no_grad()
        def predict(grid_input_t):
            e = encoder(grid_input_t)
            u = urm(e['tokens'])
            l = decoder(u['final_state'], GRID)
            return l.argmax(dim=1)[0]

        def eval_arc(root, label):
            sd = Path(root) / 'evaluation'
            files = sorted(sd.glob('*.json'))
            n_pe = 0; n_t = 0; cells_c = []; ts = 0; tt = 0
            for tf in files:
                with tf.open() as f:
                    task = _json.load(f)
                solved = []
                for tp in task.get('test', []):
                    if 'output' not in tp: continue
                    gi = grid_to_fixed(tp['input']).unsqueeze(0).to(device)
                    gt = grid_to_fixed(tp['output']).to(device)
                    pred = predict(gi)
                    exact = bool((pred == gt).all().item())
                    n_t += 1
                    if exact: n_pe += 1
                    cells_c.append(float((pred == gt).float().mean().item()))
                    solved.append(exact)
                if solved:
                    tt += 1
                    if all(solved): ts += 1
            return {'label': label, 'pairs_total': n_t, 'pairs_exact': n_pe,
                    'pair_exact_acc': n_pe / max(n_t, 1),
                    'mean_cell_acc': sum(cells_c)/max(len(cells_c), 1),
                    'tasks_total': tt, 'tasks_solved': ts,
                    'task_acc': ts / max(tt, 1)}

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
        results['math'] = eval_proc(lambda rng=rng: gen_math_pair(side=GRID, rng=rng), 'Math', n=100)
        print(_json.dumps(results['math'], indent=2))

        out_path = ROOT / 'eval_all.json'
        with out_path.open('w') as f:
            _json.dump(results, f, indent=2)

        print('\\n=== FINAL ===')
        for k, r in results.items():
            ea = r.get('pair_exact_acc', 0) * 100
            ca = r.get('mean_cell_acc', 0) * 100
            print(f'{k:8s}  pair_exact={ea:.2f}%  cell_acc={ca:.1f}%')
        print('AE-Gödel:', _json.dumps(ae_archive.summary(), indent=2))
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
