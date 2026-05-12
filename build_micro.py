"""Build ONIRO v40.0 Colab notebook.

v40.0 (math-as-code, sub-version 1 of 3):
    - Op-conditioning: OpEmbedding(32, D) injected as first token of URM input
    - Dual-encoder split: PatchEncoder (vision) for ARC, MathPatchEncoder (numeric)
      for math/sudoku/CA/compose. Both produce 100 tokens.
    - MoL (Mixture of LoRAs) on ConvSwiGLU: 4 LoRA experts top-1 routed by router
      conditioned on (mean tokens, op_embed). +280k params.
    - AlphaEvolve-Godel population_size=4 every 2k steps
    - 32-op vocabulary tagged per sample (math_gen_v2 ops, sudoku, CA, compose)

Coming in v40.1: problem-level self_simulate + weighted TTA + MCTS hybrid.
Coming in v40.2: CodeHead + CGAR PDC/HSW curriculum.

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
        # ONIRO v40.0 — Op-conditioning + MoL + Dual-encoder split

        **Setup:** Runtime → Change runtime type → **T4 GPU** → Save → Run All.

        v40.0 upgrades over v37.1:
        - **Op-conditioning**: 32-op vocabulary, OpEmbedding token prepended to URM input
        - **Dual-encoder split**: PatchEncoder (vision) for ARC tasks, MathPatchEncoder
          (per-row numeric features) for math/sudoku/CA/compose
        - **MoL (Mixture of LoRAs)**: 4 LoRA experts top-1 routed by (mean tokens, op_embed)
          on ConvSwiGLU. +280k params, paper arxiv:2512.12880.
        - **AlphaEvolve-Godel population_size=4** every 2k steps for ES expansion
        - **Math-v2 (21 ops) tagged with op_id** so model learns each operation explicitly
        - Retained from v37.1: GQA + KV cache + EMA + RIMA + Socrates Loss + 128-TTA

        Coming in v40.1: problem-level self_simulate + MCTS hybrid.
        Coming in v40.2: CodeHead + CGAR PDC/HSW curriculum.

        Runtime ~7-9h on Colab T4.
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
        # Core ARC repos
        ARC2_DIR = ROOT / 'ARC-AGI-2'
        ARC1_DIR = ROOT / 'ARC-AGI-1'
        REARC_DIR = ROOT / 're-arc'
        CONCEPT_DIR = ROOT / 'ConceptARC'
        MINI_DIR = ROOT / 'Mini-ARC'

        if not ARC2_DIR.exists():
            subprocess.check_call(['git','clone','--depth','1',
                'https://github.com/arcprize/ARC-AGI-2.git', str(ARC2_DIR)])
        if not ARC1_DIR.exists():
            subprocess.check_call(['git','clone','--depth','1',
                'https://github.com/fchollet/ARC-AGI.git', str(ARC1_DIR)])
        if not REARC_DIR.exists():
            subprocess.check_call(['git','clone','--depth','1',
                'https://github.com/michaelhodel/re-arc.git', str(REARC_DIR)])
        # ConceptARC + Mini-ARC: pulled via neoneye collection (sparse, only what we need)
        NEONEYE_DIR = ROOT / 'arc-dataset-collection'
        if not NEONEYE_DIR.exists():
            subprocess.check_call(['git','clone','--depth','1','--filter=blob:none','--sparse',
                'https://github.com/neoneye/arc-dataset-collection.git', str(NEONEYE_DIR)])
            subprocess.check_call(['git','-C',str(NEONEYE_DIR),'sparse-checkout','set',
                'dataset/ConceptARC','dataset/Mini-ARC','dataset/ARC-Heavy'])

        ARC2_ROOT = str(ARC2_DIR / 'data')
        ARC1_ROOT = str(ARC1_DIR / 'data')
        REARC_ROOT = str(REARC_DIR)
        print('ARC2 train:', len(list(Path(ARC2_ROOT,'training').glob('*.json'))))
        print('ARC1 train:', len(list(Path(ARC1_ROOT,'training').glob('*.json'))))
        # RE-ARC has tasks_train_re-arc/*.json
        rearc_glob = list(Path(REARC_ROOT).rglob('*.json'))
        print('RE-ARC json files (any):', len(rearc_glob))
    """).strip()))

    cells.append(code("!pip -q install einops 2>&1 | tail -2"))

    cells.append(md("## Build URM v40.0 (~20.6M params: OpEmbed + dual encoder + MoL URM + Socrates decoder)"))
    cells.append(code(textwrap.dedent("""
        import time, json as _json, random
        import numpy as np
        import torch
        import torch.nn.functional as F

        from oniro.models.urm import URM
        from oniro.models.grid_token_encoder import GridTokenEncoder, GridTokenDecoder
        from oniro.models.patch_encoder import PatchEncoder
        from oniro.models.math_patch_encoder import MathPatchEncoder
        from oniro.models.op_embedding import OpEmbedding, OP_ID, N_OPS
        from oniro.losses.dis import make_dis_targets
        from oniro.losses.socrates import socrates_grid_ce, socrates_argmax
        from oniro.orchestrator.alphaevolve_godel import alphaevolve_godel_round, AlphaEvolveGodelArchive
        from oniro.data.arc_json_loader import load_arc_dir, flat_pairs, _pairs_from_task
        from oniro.data.sudoku_gen import gen_sudoku_pair
        from oniro.data.math_gen import gen_math_pair
        from oniro.data.math_gen_v2 import gen_math_pair_v2, ALL_GENERATORS as MATH_V2_GENS
        from oniro.data.cellular_automata import gen_ca_pair
        from oniro.data.color_perm import random_color_perm, apply_color_perm
        from oniro.training.grpo import grpo_step, snapshot_policy
        from oniro.training.ema import EMA

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('device:', device)
        if device == 'cpu':
            print('WARNING: GPU not active. Runtime → Change runtime type → T4 GPU')

        # v40.0 config
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
        N_OUT_CLASSES = N_COLORS + 1   # +1 for UNKNOWN (Socrates)
        PATCH_SIZE = 3
        BATCH = 8

        # MoL config
        MOL_N_EXPERTS = 4
        MOL_RANK = 16

        # Dual-encoder split: vision (PatchEncoder) + math (MathPatchEncoder)
        cell_enc = GridTokenEncoder(grid_size=GRID, n_colors=N_COLORS, d_model=D).to(device)
        patch_enc_vision = PatchEncoder(grid_size=GRID, n_colors=N_COLORS,
                                        patch_size=PATCH_SIZE, d_model=D).to(device)
        patch_enc_math = MathPatchEncoder(grid_size=GRID, n_colors=N_COLORS,
                                          d_model=D, n_out_tokens=100).to(device)
        op_embedding = OpEmbedding(n_ops=N_OPS, d_model=D).to(device)

        # URM with MoL on ConvSwiGLU FFN
        urm = URM(d_model=D, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
                  n_loops=N_LOOPS, n_forward_only=N_FORWARD_ONLY,
                  ffn_hidden=FFN, n_groups=N_GROUPS,
                  kv_refresh_every=KV_REFRESH, use_rima=True,
                  use_mol=True, mol_n_experts=MOL_N_EXPERTS, mol_rank=MOL_RANK).to(device)
        decoder = GridTokenDecoder(d_model=D, n_colors=N_OUT_CLASSES).to(device)

        all_modules = [cell_enc, patch_enc_vision, patch_enc_math,
                       op_embedding, urm, decoder]
        all_params = [p for m in all_modules for p in m.parameters()]
        n_p = sum(p.numel() for p in all_params)
        print(f'ONIRO v40.0: {n_p/1e6:.2f}M params')
        print(f'  URM: D={D}, h={N_HEADS}/kv={N_KV_HEADS}, loops={N_LOOPS}/{N_GROUPS}grp, '
              f'ffn={FFN}, MoL experts={MOL_N_EXPERTS} rank={MOL_RANK}')
        print(f'  Dual encoder: vision_patch (ARC) + math_patch (non-ARC) + op token')

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

        def encode_v40(g_int: torch.Tensor,
                       op_id: torch.Tensor,
                       is_arc_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            '''Returns (urm_input_tokens, op_embed_token).

            Sequence: [op_token(1), patch_tokens(100), cell_tokens(900)] = 1001.
            Patch tokens come from vision encoder for ARC samples (mask=1) and
            from math encoder for non-ARC samples (mask=0). Vision and math
            tokens are blended per-sample using arc_mask so a single batch
            mixes the two pathways cleanly.
            '''
            cell_tok = cell_enc(g_int)['tokens']                  # (B, 900, D)
            vis_tok = patch_enc_vision(g_int)                     # (B, 100, D)
            math_tok = patch_enc_math(g_int)                      # (B, 100, D)
            m = is_arc_mask.view(-1, 1, 1).to(vis_tok.dtype).to(vis_tok.device)
            patch_tok = m * vis_tok + (1.0 - m) * math_tok        # blend per sample
            op_tok = op_embedding(op_id)                          # (B, 1, D)
            urm_input = torch.cat([op_tok, patch_tok, cell_tok], dim=1)  # (B, 1001, D)
            return urm_input, op_tok

        ema = EMA(all_modules, decay=0.999)
    """).strip()))

    cells.append(md("## Datasets: ARC train multi-source + procedural"))
    cells.append(code(textwrap.dedent("""
        rng = random.Random(0)
        ARC1_FILES = sorted((Path(ARC1_ROOT) / 'training').glob('*.json'))
        ARC2_FILES = sorted((Path(ARC2_ROOT) / 'training').glob('*.json'))

        # ConceptARC + Mini-ARC + ARC-Heavy via neoneye sparse checkout
        CONCEPT_FILES = list((NEONEYE_DIR / 'dataset' / 'ConceptARC').rglob('*.json'))
        MINI_FILES    = list((NEONEYE_DIR / 'dataset' / 'Mini-ARC').rglob('*.json'))
        HEAVY_FILES   = list((NEONEYE_DIR / 'dataset' / 'ARC-Heavy').rglob('*.json'))

        # RE-ARC: tasks_train_re-arc has 1000 pairs per ARC-1 task (or any json with pairs)
        REARC_FILES = list(Path(REARC_ROOT).rglob('*.json'))
        # Filter to those that contain ARC-format keys
        def _has_arc_format(p):
            try:
                with p.open() as f:
                    t = _json.load(f)
                return isinstance(t, dict) and ('train' in t or 'test' in t)
            except Exception:
                return False
        REARC_FILES = [p for p in REARC_FILES if _has_arc_format(p)]

        print(f'ARC1 tasks: {len(ARC1_FILES)}')
        print(f'ARC2 tasks: {len(ARC2_FILES)}')
        print(f'ConceptARC tasks: {len(CONCEPT_FILES)}')
        print(f'Mini-ARC tasks: {len(MINI_FILES)}')
        print(f'ARC-Heavy tasks: {len(HEAVY_FILES)}')
        print(f'RE-ARC tasks: {len(REARC_FILES)}')

        _CACHE = {}
        def _task_pairs_cached(tf):
            s = str(tf)
            if s not in _CACHE:
                try:
                    with tf.open() as f:
                        task = _json.load(f)
                    _CACHE[s] = _pairs_from_task(task)
                except Exception:
                    _CACHE[s] = []
            return _CACHE[s]

        def sample_arc(files):
            if not files: return None
            tf = rng.choice(files)
            pairs = _task_pairs_cached(tf)
            if not pairs: return None
            inp, out = rng.choice(pairs)
            return np.asarray(inp, dtype=np.int64), np.asarray(out, dtype=np.int64)

        def dihedral_aug(g, k, flip):
            g = np.rot90(g, k=k).copy()
            if flip:
                g = np.flip(g, axis=1).copy()
            return g

        # v40.0 mix - ARC-1 heavy + math-v2 21-op + op_id per source
        # is_arc flag routes the patch encoder pathway; op_id tags every sample.
        # math_gen_v2 op_id is sampled inside the lambda per call (5..25).
        def _math_v2_with_op():
            fn = rng.choice(MATH_V2_GENS)
            op_idx = MATH_V2_GENS.index(fn)
            pair = fn(min(GRID, 16), rng)
            return pair, OP_ID["MATH_ADD"] + op_idx  # ids 5..25

        def _ca_with_op():
            r = rng.random()
            if r < 0.5:
                pair = gen_ca_pair(rng=rng, side=min(GRID, 20))
                return pair, OP_ID["CA_CONWAY"]
            elif r < 0.8:
                pair = gen_ca_pair(rng=rng, side=min(GRID, 20))
                return pair, OP_ID["CA_BS"]
            else:
                pair = gen_ca_pair(rng=rng, side=min(GRID, 20))
                return pair, OP_ID["CA_RULE110"]

        MIX_WEIGHTS = [
            # (name, weight, is_arc, op_id, fn_returning (pair_or_None, op_id_override_or_None))
            ('ARC-1',   0.25, True,  OP_ID["ARC_GENERIC"], lambda: (sample_arc(ARC1_FILES), None)),
            ('RE-ARC',  0.20, True,  OP_ID["ARC_RE"],      lambda: (sample_arc(REARC_FILES), None)),
            ('ARC-2',   0.15, True,  OP_ID["ARC_GENERIC"], lambda: (sample_arc(ARC2_FILES), None)),
            ('Concept', 0.05, True,  OP_ID["ARC_CONCEPT"], lambda: (sample_arc(CONCEPT_FILES), None)),
            ('Mini',    0.03, True,  OP_ID["ARC_MINI"],    lambda: (sample_arc(MINI_FILES), None)),
            ('Heavy',   0.05, True,  OP_ID["ARC_HEAVY"],   lambda: (sample_arc(HEAVY_FILES), None)),
            ('Math-v2', 0.15, False, None,                  lambda: _math_v2_with_op()),
            ('Sudoku',  0.05, False, OP_ID["SUDOKU"],       lambda: (gen_sudoku_pair(mask_rate=0.4, rng=rng), None)),
            ('CA',      0.05, False, None,                  lambda: _ca_with_op()),
            ('Compose', 0.02, False, OP_ID["DSL_COMPOSE"],  lambda: (_gen_dsl_compose(), None)),
        ]
        _cum = []
        s = 0.0
        for nm, w, is_arc, op_default, fn in MIX_WEIGHTS:
            s += w
            _cum.append((s, nm, is_arc, op_default, fn))
        TOTAL_W = s

        def _gen_dsl_compose():
            '''Random self-exec DSL composition: input + applied transform = output.'''
            # synthetic small grid -> random rot+flip+recolor
            side = rng.randint(5, min(GRID, 16))
            g = np.random.randint(0, N_COLORS, size=(side, side), dtype=np.int64)
            out = g.copy()
            ops = rng.randint(1, 3)
            for _ in range(ops):
                op = rng.choice(['rot', 'flip', 'recolor'])
                if op == 'rot':
                    out = np.rot90(out, k=rng.randint(1, 3)).copy()
                elif op == 'flip':
                    out = np.flip(out, axis=rng.randint(0, 1)).copy()
                else:
                    s_ = rng.randint(0, N_COLORS-1); d_ = rng.randint(0, N_COLORS-1)
                    out = np.where(out == s_, d_, out)
            return g, out

        def sample_one():
            r = rng.random() * TOTAL_W
            for thresh, name, is_arc, op_default, fn in _cum:
                if r < thresh:
                    pair, op_override = fn()
                    if pair is None:
                        return sample_one()
                    op_id = op_override if op_override is not None else op_default
                    return pair, is_arc, op_id
            # fallback (should not trigger)
            pair, op_override = _cum[-1][4]()
            return pair, _cum[-1][2], _cum[-1][3]

        def sample_batch(B, do_aug=True):
            gi_, go_, arc_flags, op_ids = [], [], [], []
            for _ in range(B):
                (inp, out), is_arc, op_id = sample_one()
                inp = np.asarray(inp, dtype=np.int64)
                out = np.asarray(out, dtype=np.int64)
                if do_aug:
                    k = rng.randint(0, 3)
                    flip = rng.random() < 0.5
                    inp = dihedral_aug(inp, k, flip)
                    out = dihedral_aug(out, k, flip)
                    if rng.random() < 0.7:
                        cp = random_color_perm(rng, n_colors=N_COLORS, keep_bg=True)
                        inp = apply_color_perm(inp, cp)
                        out = apply_color_perm(out, cp)
                gi_.append(grid_to_fixed(inp))
                go_.append(grid_to_fixed(out))
                arc_flags.append(1.0 if is_arc else 0.0)
                op_ids.append(int(op_id))
            return (torch.stack(gi_).to(device),
                    torch.stack(go_).to(device),
                    torch.tensor(arc_flags, dtype=torch.float, device=device),
                    torch.tensor(op_ids, dtype=torch.long, device=device))

        g_in, g_out, arc_mask, op_id = sample_batch(4)
        print('sample shapes:', g_in.shape, g_out.shape,
              'arc_mask:', arc_mask.tolist(), 'op_ids:', op_id.tolist())
    """).strip()))

    cells.append(md("## Train: Socrates + dual encoder + AE-Godel"))
    cells.append(code(textwrap.dedent("""
        STEPS = int(os.environ.get('ONIRO_STEPS', '40000'))
        GRPO_STEPS = int(os.environ.get('ONIRO_GRPO_STEPS', '5000'))
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

        # MoL load-balance loss weight (small to avoid dominating)
        MOL_LB_WEIGHT = 0.01

        t0 = time.time()
        for step in range(STEPS):
            g_in, g_out, arc_mask, op_id = sample_batch(BATCH)
            urm_input, op_tok = encode_v40(g_in, op_id, arc_mask)   # (B, 1001, D), (B, 1, D)
            urm_out = urm(urm_input, op_embed=op_tok)

            loss = torch.zeros((), device=device)
            n_states = len(urm_out['states_per_loop'])
            dis_targets = make_dis_targets(g_out, n_cycles=n_states - 1,
                                            n_colors=N_COLORS, max_corruption=0.5,
                                            seed=step)
            for t, state in enumerate(urm_out['states_per_loop'][1:]):
                logits = decoder(state, GRID)
                tgt = dis_targets[t].to(device)
                weight = 1.5 ** (-(n_states - 2 - t))
                loss = loss + weight * socrates_grid_ce(
                    logits, tgt, n_colors=N_COLORS,
                    unknown_class=N_COLORS, gamma=0.05, bg_weight=0.15
                )

            # MoL load-balance aux loss (sum over all blocks)
            lb_loss = torch.zeros((), device=device)
            for blk in urm.blocks:
                if blk.use_mol:
                    lb_loss = lb_loss + blk.ffn.load_balance_loss().to(device)
            loss = loss + MOL_LB_WEIGHT * lb_loss

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
                    pred = socrates_argmax(final_logits, n_colors=N_COLORS,
                                            unknown_class=N_COLORS)
                    cell_acc = (pred == g_out).float().mean().item()
                lr_cur = sched.get_last_lr()[0]
                # show MoL expert usage histogram for the last block
                last_usage = urm.blocks[-1].ffn.last_expert_usage if urm.blocks[-1].use_mol else None
                usage_str = f'  mol={[round(float(u), 2) for u in last_usage.tolist()]}' if last_usage is not None else ''
                print(f'step {step:5d}  loss={float(loss.detach()):.4f}  '
                      f'lb={float(lb_loss.detach()):.4f}  cell_acc={cell_acc:.3f}  '
                      f'lr={lr_cur:.2e}  rate={(step+1)/max(dt,1):.1f}/s{usage_str}')

            if step > 0 and step % AE_EVERY == 0:
                for m in all_modules: m.eval()
                eval_in, eval_out, eval_arc, eval_op = sample_batch(BATCH, do_aug=False)
                def ae_score():
                    with torch.no_grad():
                        t_, op_t = encode_v40(eval_in, eval_op, eval_arc)
                        u = urm(t_, op_embed=op_t)
                        l = decoder(u['final_state'], GRID)
                        p = socrates_argmax(l, n_colors=N_COLORS, unknown_class=N_COLORS)
                        return float((p == eval_out).float().mean().item())
                acc, base, best, ae_archive = alphaevolve_godel_round(
                    urm, ae_score, n_candidates=3, sigma=2e-3,
                    population_size=4,  # v40.0: ES expansion
                    archive=ae_archive,
                )
                print(f'  [AE pop=4] base={base:.4f} best={best:.4f} accept={acc} reject={ae_archive.rejected}')
                for m in all_modules: m.train()

        print(f'\\nphase A (supervised) done in {(time.time()-t0)/60:.1f}min')

        # ============= Phase B: GRPO RL =============
        print(f'\\n=== Phase B: GRPO RL ({GRPO_STEPS} steps, group=4) ===')

        # Adapter for GRPO API. For RL phase, the GRPO step does not have a
        # per-sample op_id signal, so we default everything to ARC_GENERIC (id 0)
        # and assume vision pathway. URM still uses MoL but with a constant op_embed.
        class V40EncoderAdapter(torch.nn.Module):
            def __init__(self, cell, vis_patch, math_patch, op_emb):
                super().__init__()
                self.cell = cell
                self.vis_patch = vis_patch
                self.math_patch = math_patch
                self.op_emb = op_emb
            def forward(self, g):
                co = self.cell(g)['tokens']
                pt = self.vis_patch(g)
                B = g.shape[0]
                op_id = torch.zeros(B, dtype=torch.long, device=g.device)
                op_tok = self.op_emb(op_id)
                return {'tokens': torch.cat([op_tok, pt, co], dim=1),
                        'op_embed': op_tok}

        enc_adapter = V40EncoderAdapter(cell_enc, patch_enc_vision,
                                         patch_enc_math, op_embedding).to(device)
        ref_enc, ref_urm, ref_dec = snapshot_policy(enc_adapter, urm, decoder)
        for p in ref_enc.parameters(): p.requires_grad = False
        for p in ref_urm.parameters(): p.requires_grad = False
        for p in ref_dec.parameters(): p.requires_grad = False
        ref_enc.to(device); ref_urm.to(device); ref_dec.to(device)

        # RL phase only touches grid trunk parameters
        rl_trunk_params = (list(cell_enc.parameters())
                            + list(patch_enc_vision.parameters())
                            + list(patch_enc_math.parameters())
                            + list(op_embedding.parameters())
                            + list(urm.parameters())
                            + list(decoder.parameters()))
        rl_opt = torch.optim.AdamW(rl_trunk_params, lr=3e-5, weight_decay=0.05)
        for rl_step in range(GRPO_STEPS):
            g_in_b, g_out_b, _rl_arc, _rl_op = sample_batch(BATCH, do_aug=False)
            r = grpo_step(enc_adapter, urm, decoder, rl_opt, g_in_b, g_out_b,
                          ref_enc, ref_urm, ref_dec,
                          n_group=4, eps_clip=0.2, kl_beta=0.04,
                          temperature=1.0, reward_type='cell')
            ema.update()
            if rl_step % 100 == 0:
                print(f'  rl_step {rl_step:5d}  reward={r["mean_reward"]:.3f}  '
                      f'max_r={r["max_reward"]:.3f}  kl={r["kl"]:.3f}  loss={r["loss"]:.4f}')
            if rl_step > 0 and rl_step % 500 == 0:
                del ref_enc, ref_urm, ref_dec
                ref_enc, ref_urm, ref_dec = snapshot_policy(enc_adapter, urm, decoder)
                ref_enc.to(device); ref_urm.to(device); ref_dec.to(device)
                print(f'  refreshed reference policy at step {rl_step}')

        print(f'\\nphase B (GRPO RL) done')

        ckpt_dir = ROOT / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({'cell_enc': cell_enc.state_dict(),
                    'patch_enc_vision': patch_enc_vision.state_dict(),
                    'patch_enc_math': patch_enc_math.state_dict(),
                    'op_embedding': op_embedding.state_dict(),
                    'urm': urm.state_dict(),
                    'decoder': decoder.state_dict()},
                   str(ckpt_dir / 'urm_v40_0_final.pt'))
    """).strip()))

    cells.append(md("## ARC eval — 128-sample TTA majority vote (Socrates argmax + op_id=ARC_GENERIC)"))
    cells.append(code(textwrap.dedent("""
        for m in all_modules: m.eval()

        N_TTA = int(os.environ.get('ONIRO_TTA', '128'))

        # Eval defaults: op_id=ARC_GENERIC (id 0), is_arc=True (vision pathway).
        EVAL_OP_ID = torch.tensor([OP_ID["ARC_GENERIC"]], dtype=torch.long, device=device)
        EVAL_ARC_MASK = torch.tensor([1.0], dtype=torch.float, device=device)

        @torch.no_grad()
        def tta_majority_vote(grid_int_t, n_samples=N_TTA):
            votes = torch.zeros(N_COLORS, GRID, GRID, device=device)
            gnp = grid_int_t.cpu().numpy()
            for s in range(n_samples):
                k = rng.randint(0, 3); flip = rng.random() < 0.5
                cp = random_color_perm(rng, n_colors=N_COLORS, keep_bg=True)
                inv_cp = np.argsort(cp).astype(np.int64)
                gaug = dihedral_aug(gnp, k, flip)
                gaug = apply_color_perm(gaug, cp)
                t = grid_to_fixed(gaug).unsqueeze(0).to(device)
                urm_input, op_tok = encode_v40(t, EVAL_OP_ID, EVAL_ARC_MASK)
                u = urm(urm_input, op_embed=op_tok)
                l = decoder(u['final_state'], GRID)
                pred = socrates_argmax(l, n_colors=N_COLORS, unknown_class=N_COLORS)[0]
                pred_np = pred.cpu().numpy()
                pred_np = apply_color_perm(pred_np.astype(np.int64), inv_cp)
                pred = torch.from_numpy(pred_np).to(device)
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
                    'task_acc': ts / max(tt, 1), 'tta_samples': N_TTA}

        results = {}
        print(f'=== ARC-AGI-1 (TTA {N_TTA}) ===')
        results['arc1'] = eval_arc_tta(ARC1_ROOT, 'ARC-1')
        print(_json.dumps(results['arc1'], indent=2))
        print(f'=== ARC-AGI-2 (TTA {N_TTA}) ===')
        results['arc2'] = eval_arc_tta(ARC2_ROOT, 'ARC-2')
        print(_json.dumps(results['arc2'], indent=2))
    """).strip()))

    cells.append(md("## DSL Hybrid + procedural eval"))
    cells.append(code(textwrap.dedent("""
        from oniro.dsl.solver import solve_task as dsl_solve_task

        @torch.no_grad()
        def neural_predict_np(grid_np):
            gi = grid_to_fixed(grid_np.tolist()).unsqueeze(0).to(device)
            urm_input, op_tok = encode_v40(gi, EVAL_OP_ID, EVAL_ARC_MASK)
            u = urm(urm_input, op_embed=op_tok)
            l = decoder(u['final_state'], GRID)
            return socrates_argmax(l, n_colors=N_COLORS, unknown_class=N_COLORS)[0].cpu().numpy().astype(np.int8)

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

        print('=== ARC-AGI-1 hybrid ===')
        results['arc1_hybrid'] = eval_arc_hybrid(ARC1_ROOT, 'ARC-1', max_dsl_depth=3)
        print(_json.dumps(results['arc1_hybrid'], indent=2))
        print('=== ARC-AGI-2 hybrid ===')
        results['arc2_hybrid'] = eval_arc_hybrid(ARC2_ROOT, 'ARC-2', max_dsl_depth=3)
        print(_json.dumps(results['arc2_hybrid'], indent=2))

        # Procedural quick eval (uses op_id=ARC_GENERIC and vision pathway for now;
        # v40.1 will route procedural eval through op_id-aware path).
        @torch.no_grad()
        def predict_grid_t(g_t):
            B = g_t.shape[0]
            op_id_eval = torch.full((B,), OP_ID["ARC_GENERIC"], dtype=torch.long, device=device)
            arc_mask_eval = torch.ones(B, dtype=torch.float, device=device)
            urm_input, op_tok = encode_v40(g_t, op_id_eval, arc_mask_eval)
            u = urm(urm_input, op_embed=op_tok)
            l = decoder(u['final_state'], GRID)
            return socrates_argmax(l, n_colors=N_COLORS, unknown_class=N_COLORS)[0]

        def eval_proc(gen_fn, label, n=100):
            n_pe = 0; cell_c = []
            with ema.swap_in():
                for _ in range(n):
                    inp, out = gen_fn(rng=rng)
                    gi = grid_to_fixed(inp.tolist() if hasattr(inp, 'tolist') else inp).unsqueeze(0).to(device)
                    gt = grid_to_fixed(out.tolist() if hasattr(out, 'tolist') else out).to(device)
                    pred = predict_grid_t(gi)
                    exact = bool((pred == gt).all().item())
                    if exact: n_pe += 1
                    cell_c.append(float((pred == gt).float().mean().item()))
            return {'label': label, 'n_samples': n, 'pair_exact_acc': n_pe / n,
                    'mean_cell_acc': sum(cell_c)/len(cell_c)}

        print('=== Sudoku ===')
        results['sudoku'] = eval_proc(lambda rng=rng: gen_sudoku_pair(mask_rate=0.4, rng=rng), 'Sudoku', n=100)
        print(_json.dumps(results['sudoku'], indent=2))
        print('=== Math grid ===')
        results['math'] = eval_proc(lambda rng=rng: gen_math_pair(side=min(GRID, 16), rng=rng), 'Math', n=100)
        print(_json.dumps(results['math'], indent=2))
        print('=== Cellular Automata ===')
        results['ca'] = eval_proc(lambda rng=rng: gen_ca_pair(rng=rng, side=min(GRID, 20)), 'CA', n=100)
        print(_json.dumps(results['ca'], indent=2))

        with open(str(ROOT / 'eval_v37.json'), 'w') as f:
            _json.dump(results, f, indent=2)

        print('\\n=== FINAL v37 ===')
        for k, r in results.items():
            if 'task_acc' in r:
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
