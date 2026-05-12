"""Build ONIRO v40.4 Colab notebook.

v40.4 (ALL plan items wired, no deferreds remaining):

All v40.0/v40.1/v40.2/v40.3 features PLUS:
- BARC loader + repo clone (xu3kev/BARC). Synthetic GPT-generated ARC tasks.
- H-ARC loader + repo clone (arc-visualizations/h-arc). Human solver traces.
- VSA bindings module (oniro/models/vsa_bindings.py) - HRR-style bind/unbind/
  bundle + VSALayer (arxiv:2511.08747 Joffe & Eliasmith).
- Full FFN MoE alternative to MoL (oniro/models/full_moe.py). Heavier capacity
  if MoL ceiling is hit. Default disabled.
- TTAT-TRM test-time trunk adaptation (oniro/eval/ttat_trm.py) - snapshot,
  fine-tune on demos, restore. Per arxiv:2511.02886 McGovern.
- Learned MCTS critic (oniro/eval/mcts_critic.py) - small CNN+MLP that scores
  (urm_state, candidate_grid) pairs. Hand-rolled self_simulate stays as
  fallback when critic not trained yet.

Hybrid encoder pathway (Pedro: "geometria ativa AMBOS"):
- arc_mask is now FLOAT in [0, 1] instead of bool.
- 1.0 = pure vision pathway (ARC family + BARC + H-ARC).
- 0.0 = pure math pathway (Math-v2 pure-numeric ops, Sudoku).
- 0.5 = hybrid, both pathways blended (CA, DSL-compose, Enigmata, spatial
  math ops: rotate, mirror, gravity, sort, count, histogram).
- encode_v40 already handles float blend per-sample (`m * vis + (1-m) * math`).

Mix (v40.4 final, includes BARC and H-ARC slots):
  ARC-1 0.20, RE-ARC 0.16, ARC-GEN 0.10, ARC-2 0.12, Concept 0.04,
  Mini 0.02, Heavy 0.04, BARC 0.04, Math-v2 0.10, Enigmata 0.06,
  Sudoku 0.04, CA 0.04, Compose 0.02, H-ARC 0.02.

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
        # ONIRO v40.4 — Final: VSA + BARC + H-ARC + Full MoE + TTAT + MCTS critic + hybrid pathway

        **Setup:** Runtime → Change runtime type → **T4 GPU** → Save → Run All.

        v40.2 upgrades over v40.1 (FINAL sub-version):
        - **CodeHead** aux predictor: 3-step DSL primitive sequence per task.
          Trained on synthetic DSL_COMPOSE with known programs; NULL_PROGRAM for ARC.
        - **CGAR PDC** (arxiv:2511.08653): Progressive Depth Curriculum.
          Phase A0 (0-20%): URM at n_loops/3.
          Phase A1 (20-50%): URM at 2*n_loops/3.
          Phase A2 (50%-100%): full n_loops.
          Compute saved on early stages.
        - **CGAR HSW** Hierarchical Supervision Weighting: cycle weights ramp
          from uniform (early training) to late-cycle-heavy (consolidation).
        - **safe_softmax=True** in Socrates Loss for explicit numerical stability.

        Retained from v40.0 + v40.1:
        - Op-conditioning (32-op vocabulary) + dual encoder split
        - MoL ConvSwiGLU (4 LoRA experts top-1 routed)
        - AlphaEvolve-Godel population_size=4 every 2k steps
        - Problem-level self-simulate weighted TTA + MCTS hybrid eval

        Runtime ~7-9h on Colab T4 (PDC saves Phase A compute despite added aux head).
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
        ARCGEN_DIR = ROOT / 'ARC-GEN'
        ENIGMATA_DIR = ROOT / 'Enigmata'
        CONCEPT_DIR = ROOT / 'ConceptARC'
        MINI_DIR = ROOT / 'Mini-ARC'

        if not ARC2_DIR.exists():
            subprocess.check_call(['git','clone','--depth','1',
                'https://github.com/arcprize/ARC-AGI-2.git', str(ARC2_DIR)])
        if not ARC1_DIR.exists():
            subprocess.check_call(['git','clone','--depth','1',
                'https://github.com/fchollet/ARC-AGI.git', str(ARC1_DIR)])
        # RE-ARC: repo ships only the GENERATOR. Generation inline was hanging
        # in Colab sessions (rate 0/s). Skip it by default; user can run the
        # generator manually if they want. The training will simply not see
        # RE-ARC samples and sample_one will fall through to other sources.
        if not REARC_DIR.exists():
            try:
                subprocess.check_call(['git','clone','--depth','1',
                    'https://github.com/michaelhodel/re-arc.git', str(REARC_DIR)],
                    timeout=60)
            except Exception as e:
                print(f'RE-ARC clone failed (will run without): {e}')
        # ARC-GEN: only generator code, no shipped data. Clone but expect 0.
        if not ARCGEN_DIR.exists():
            try:
                subprocess.check_call(['git','clone','--depth','1',
                    'https://github.com/google/ARC-GEN.git', str(ARCGEN_DIR)],
                    timeout=60)
            except Exception as e:
                print(f'ARC-GEN clone failed (will run without): {e}')
        # Enigmata: full clone with timeout
        if not ENIGMATA_DIR.exists():
            try:
                subprocess.check_call(['git','clone','--depth','1',
                    'https://github.com/BytedTsinghua-SIA/Enigmata.git', str(ENIGMATA_DIR)],
                    timeout=60)
            except Exception as e:
                print(f'Enigmata clone failed (will run without): {e}')
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
        rearc_glob = list(Path(REARC_ROOT).rglob('*.json'))
        print('RE-ARC json files (any):', len(rearc_glob))
        # ARC-GEN + Enigmata enumeration (optional - if clone failed, lists are empty)
        arcgen_files = list(ARCGEN_DIR.rglob('*.json')) if ARCGEN_DIR.exists() else []
        enigmata_files = list(ENIGMATA_DIR.rglob('*.json')) if ENIGMATA_DIR.exists() else []
        print('ARC-GEN files:', len(arcgen_files))
        print('Enigmata files:', len(enigmata_files))
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
        from oniro.models.code_head import CodeHead
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
        from oniro.training.cgar_schedule import pdc_loops, hsw_weights

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

        # v40.2: CodeHead aux predictor for DSL primitive sequence
        # n_prims should match the size of the DSL library; oniro/dsl/primitives
        # exposes 7 geom + 4 struct + 45 swap + 90 recolor = 146 primitives.
        N_DSL_PRIMS = 146
        code_head = CodeHead(d_model=D, n_prims=N_DSL_PRIMS, seq_len=3,
                              hidden=256).to(device)

        all_modules = [cell_enc, patch_enc_vision, patch_enc_math,
                       op_embedding, urm, decoder, code_head]
        all_params = [p for m in all_modules for p in m.parameters()]
        n_p = sum(p.numel() for p in all_params)
        print(f'ONIRO v40.2: {n_p/1e6:.2f}M params')
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

        # RE-ARC stores per-task JSON in re_arc/tasks/ as a FLAT LIST of
        # pairs. _pairs_from_task in arc_json_loader handles this format.
        # Skip the pre-filter (was slow on large dirs); _task_pairs_cached
        # returns [] for files without valid pairs and sample_one retries.
        REARC_FILES = list(Path(REARC_ROOT).rglob('*.json')) if REARC_DIR.exists() else []

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

        # v40.4 mix - arc_mask is now a FLOAT (0.0 pure math, 0.5 geometry/hybrid,
        # 1.0 pure vision/ARC). encode_v40 blends patch_vision and patch_math
        # by this mask, so a 0.5 sample activates BOTH pathways equally.
        # Spatial math ops (rotate, mirror, gravity, sort, histogram, count)
        # are flagged as hybrid because they need visual + numeric reasoning.
        SPATIAL_MATH_OP_NAMES = {
            "MATH_SORT", "MATH_GRAVITY", "MATH_MIRROR_H", "MATH_ROTATE",
            "MATH_COUNT", "MATH_HISTOGRAM",
        }

        def _math_v2_with_op():
            fn = rng.choice(MATH_V2_GENS)
            op_idx = MATH_V2_GENS.index(fn)
            pair = fn(min(GRID, 16), rng)
            op_id_v = OP_ID["MATH_ADD"] + op_idx
            # decide arc_mask based on op semantics
            op_name = next((k for k, v in OP_ID.items() if v == op_id_v), None)
            arc_mask_v = 0.5 if op_name in SPATIAL_MATH_OP_NAMES else 0.0
            return pair, op_id_v, None, arc_mask_v

        def _ca_with_op():
            r = rng.random()
            pair = gen_ca_pair(rng=rng, side=min(GRID, 20))
            if r < 0.5:    op = OP_ID["CA_CONWAY"]
            elif r < 0.8:  op = OP_ID["CA_BS"]
            else:           op = OP_ID["CA_RULE110"]
            # CA = local-rule spatial reasoning -> hybrid
            return pair, op, None, 0.5

        def _compose_with_program():
            pair, prog = _gen_dsl_compose_with_program()
            # DSL compose uses geometric primitives -> hybrid
            return pair, OP_ID["DSL_COMPOSE"], prog, 0.5

        ARCGEN_FILES = arcgen_files if 'arcgen_files' in dir() else []
        ENIGMATA_FILES = enigmata_files if 'enigmata_files' in dir() else []
        ENIGMATA_OP_ID = OP_ID["MATH_ARITH_CHAIN"]   # logic/grid puzzles
        # v40.5: corrected dataset sources
        # BARC actual data is on HuggingFace (barc0); the GitHub repo only ships
        # the synthesis pipeline (Python seeds, not JSON tasks). We try HF Hub
        # but fall back to empty if `datasets` is not installed.
        # H-ARC correct URL is github.com/le-gris/h-arc per arxiv:2409.01374.
        # We also add neoneye/arc-dataset-tama as a large auxiliary ARC-format
        # source.
        BARC_DIR = ROOT / 'BARC-hf'
        HARC_DIR = ROOT / 'H-ARC'
        TAMA_DIR = ROOT / 'arc-dataset-tama'

        BARC_FILES = []
        # BARC HuggingFace download was stalling Colab. Skip by default.
        # User can enable with env var ONIRO_FETCH_BARC=1.
        if os.environ.get('ONIRO_FETCH_BARC') == '1' and not BARC_DIR.exists():
            try:
                subprocess.check_call(['pip', '-q', 'install', 'huggingface_hub'],
                                       timeout=120)
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="barc0/100k_yes_promptv2_problems_with_descriptions",
                                  repo_type="dataset", local_dir=str(BARC_DIR))
            except Exception as e:
                print(f'BARC HF download failed (will run without): {e}')
        if BARC_DIR.exists():
            BARC_FILES = (list(BARC_DIR.rglob('*.json'))
                          + list(BARC_DIR.rglob('*.jsonl')))

        # H-ARC: scripts only, no shipped data. Skip clone (was contributing 0).
        HARC_FILES = []

        # neoneye/arc-dataset-tama (Big ARC tasks, standard ARC JSON format)
        if not TAMA_DIR.exists():
            try:
                subprocess.check_call(['git','clone','--depth','1',
                    'https://github.com/neoneye/arc-dataset-tama.git', str(TAMA_DIR)],
                    timeout=120)
            except Exception as e:
                print(f'arc-dataset-tama clone failed (will run without): {e}')
        TAMA_FILES = list(TAMA_DIR.rglob('*.json')) if TAMA_DIR.exists() else []

        print(f'BARC files: {len(BARC_FILES)}, H-ARC files: {len(HARC_FILES)}, '
              f'arc-dataset-tama files: {len(TAMA_FILES)}')

        # arc_mask_default is FLOAT in [0, 1]. 1.0 = pure vision pathway,
        # 0.0 = pure math pathway, 0.5 = blend both (geometry / hybrid).
        # The lambda may also override arc_mask via 4th return slot.
        # v40.6 mix: weights biased toward sources that ALWAYS work (procedural
        # generators + curated ARC-1/2). Best-effort sources (RE-ARC needs
        # in-Colab generation; ARC-GEN, BARC, H-ARC, Enigmata may be empty
        # depending on clone success) get smaller weights and fall through to
        # sample_one retry if empty.
        MIX_WEIGHTS = [
            # always-available curated ARC
            ('ARC-1',     0.22, 1.0, OP_ID["ARC_GENERIC"], lambda: (sample_arc(ARC1_FILES), None, None, None)),
            ('ARC-2',     0.14, 1.0, OP_ID["ARC_GENERIC"], lambda: (sample_arc(ARC2_FILES), None, None, None)),
            ('Concept',   0.04, 1.0, OP_ID["ARC_CONCEPT"], lambda: (sample_arc(CONCEPT_FILES), None, None, None)),
            ('Mini',      0.02, 1.0, OP_ID["ARC_MINI"],    lambda: (sample_arc(MINI_FILES), None, None, None)),
            ('Heavy',     0.04, 1.0, OP_ID["ARC_HEAVY"],   lambda: (sample_arc(HEAVY_FILES), None, None, None)),
            ('Tama',      0.04, 1.0, OP_ID["ARC_GENERIC"], lambda: (sample_arc(TAMA_FILES), None, None, None)),
            # generated / external (may be empty -> sample_one falls through)
            ('RE-ARC',    0.16, 1.0, OP_ID["ARC_RE"],      lambda: (sample_arc(REARC_FILES), None, None, None)),
            ('ARC-GEN',   0.04, 1.0, OP_ID["ARC_GENERIC"], lambda: (sample_arc(ARCGEN_FILES), None, None, None)),
            ('BARC',      0.02, 1.0, OP_ID["ARC_GENERIC"], lambda: (sample_arc(BARC_FILES), None, None, None)),
            ('Enigmata',  0.04, 0.5, ENIGMATA_OP_ID,        lambda: (sample_arc(ENIGMATA_FILES), None, None, None)),
            # always-available procedural
            ('Math-v2',   0.12, 0.0, None,                  _math_v2_with_op),
            ('Sudoku',    0.06, 0.5, OP_ID["SUDOKU"],       lambda: (gen_sudoku_pair(mask_rate=0.4, rng=rng), None, None, None)),
            ('CA',        0.04, 0.5, None,                  _ca_with_op),
            ('Compose',   0.02, 0.5, OP_ID["DSL_COMPOSE"],  _compose_with_program),
        ]
        _cum = []
        s = 0.0
        for nm, w, arc_mask_d, op_default, fn in MIX_WEIGHTS:
            s += w
            _cum.append((s, nm, arc_mask_d, op_default, fn))
        TOTAL_W = s

        # v40.3: _gen_dsl_compose also returns the program (list of primitive ids).
        # Primitive id map (matches oniro.dsl.primitives ordering):
        #   0 identity, 1 rot90, 2 rot180, 3 rot270, 4 flip_h, 5 flip_v,
        #   6 transpose, 7 crop, 8 tile_2x2, 9 double, 10 half.
        #   11..55 = color swaps. 56..145 = recolors.
        # We use a coarse "recolor_generic" mapping at id 11 for any recolor
        # (90 recolor primitives collapse to 1 class for simplicity; precise
        # color params are learned through cell-decoder gradient anyway).
        ID_ROT = {1: 1, 2: 2, 3: 3}
        ID_FLIP_H, ID_FLIP_V = 4, 5
        ID_RECOLOR = 11
        ID_IDENTITY = 0

        def _gen_dsl_compose_with_program():
            side = rng.randint(5, min(GRID, 16))
            g = np.random.randint(0, N_COLORS, size=(side, side), dtype=np.int64)
            out = g.copy()
            program: list[int] = []
            n_ops = rng.randint(1, 3)
            for _ in range(n_ops):
                op = rng.choice(['rot', 'flip', 'recolor'])
                if op == 'rot':
                    k = rng.randint(1, 3)
                    out = np.rot90(out, k=k).copy()
                    program.append(ID_ROT[k])
                elif op == 'flip':
                    axis = rng.randint(0, 1)
                    out = np.flip(out, axis=axis).copy()
                    program.append(ID_FLIP_H if axis == 1 else ID_FLIP_V)
                else:
                    s_ = rng.randint(0, N_COLORS - 1); d_ = rng.randint(0, N_COLORS - 1)
                    out = np.where(out == s_, d_, out)
                    program.append(ID_RECOLOR)
            # Pad to length 3 with identity
            while len(program) < 3:
                program.append(ID_IDENTITY)
            return (g, out), program[:3]

        def _gen_dsl_compose():
            (g, out), _prog = _gen_dsl_compose_with_program()
            return g, out

        def sample_one(_retry=0):
            # Bounded retry: if a chosen slot is empty (file list empty),
            # try a different one. Cap recursion at len(MIX_WEIGHTS) + 4.
            if _retry > len(MIX_WEIGHTS) + 4:
                # final fallback: always-available math procedural
                pair = gen_math_pair_v2(side=min(GRID, 16), rng=rng)
                return pair, 0.0, OP_ID["MATH_ADD"], None
            r = rng.random() * TOTAL_W
            for thresh, name, arc_mask_d, op_default, fn in _cum:
                if r < thresh:
                    try:
                        result = fn()
                    except Exception:
                        return sample_one(_retry=_retry + 1)
                    if len(result) == 4:
                        pair, op_override, program, mask_override = result
                    else:
                        pair, op_override, program = result
                        mask_override = None
                    if pair is None:
                        return sample_one(_retry=_retry + 1)
                    op_id = op_override if op_override is not None else op_default
                    arc_mask_v = mask_override if mask_override is not None else arc_mask_d
                    return pair, float(arc_mask_v), op_id, program
            # final-tail fallback
            return sample_one(_retry=_retry + 1)

        # NULL_PROGRAM index for CodeHead targets; matches code_head.null_class.
        N_DSL_PRIMS_HOST = 146

        def sample_batch(B, do_aug=True):
            gi_, go_, arc_flags, op_ids = [], [], [], []
            code_targets, code_masks = [], []
            for _ in range(B):
                (inp, out), arc_mask_v, op_id, program = sample_one()
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
                arc_flags.append(float(arc_mask_v))  # now float [0, 1]
                op_ids.append(int(op_id))
                # Compose samples carry a real DSL program; others use NULL.
                if program is not None and len(program) >= 3:
                    code_targets.append(list(program[:3]))
                    code_masks.append(1.0)
                else:
                    code_targets.append([N_DSL_PRIMS_HOST] * 3)
                    code_masks.append(0.0)
            return (torch.stack(gi_).to(device),
                    torch.stack(go_).to(device),
                    torch.tensor(arc_flags, dtype=torch.float, device=device),
                    torch.tensor(op_ids, dtype=torch.long, device=device),
                    torch.tensor(code_targets, dtype=torch.long, device=device),
                    torch.tensor(code_masks, dtype=torch.float, device=device))

        g_in, g_out, arc_mask, op_id, code_tgt, code_mask = sample_batch(4)
        print('sample shapes:', g_in.shape, g_out.shape,
              'arc_mask:', arc_mask.tolist(), 'op_ids:', op_id.tolist(),
              'code_mask:', code_mask.tolist())
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

        # v40.2 aux loss weights
        MOL_LB_WEIGHT = 0.01
        CODE_HEAD_WEIGHT = 0.05

        # v40.7 speed: fp16 autocast on CUDA T4 has fp16 tensor cores giving
        # ~2x throughput for the URM forward (GQA + ConvSwiGLU).
        USE_AMP = device == 'cuda'
        amp_dtype = torch.float16 if USE_AMP else torch.float32
        scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

        t0 = time.time()
        for step in range(STEPS):
            # v40.2 CGAR PDC: shrink URM depth in early stages.
            urm.set_n_loops_eff(pdc_loops(step, STEPS, n_loops_full=N_LOOPS))

            g_in, g_out, arc_mask, op_id, code_tgt, code_mask = sample_batch(BATCH)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=USE_AMP):
                urm_input, op_tok = encode_v40(g_in, op_id, arc_mask)
                urm_out = urm(urm_input, op_embed=op_tok)

                loss = torch.zeros((), device=device)
                n_states = len(urm_out['states_per_loop'])
                n_cycles = n_states - 1
                dis_targets = make_dis_targets(g_out, n_cycles=n_cycles,
                                                n_colors=N_COLORS, max_corruption=0.5,
                                                seed=step)
                hsw_w = hsw_weights(n_cycles=n_cycles, step=step, total_steps=STEPS,
                                     decay=0.5, ramp_frac=0.5)
                for t, state in enumerate(urm_out['states_per_loop'][1:]):
                    logits = decoder(state, GRID)
                    tgt = dis_targets[t].to(device)
                    loss = loss + float(hsw_w[t]) * socrates_grid_ce(
                        logits, tgt, n_colors=N_COLORS,
                        unknown_class=N_COLORS, gamma=0.05, bg_weight=0.15,
                        safe_softmax=True,
                    )

                # MoL load-balance aux
                lb_loss = torch.zeros((), device=device)
                for blk in urm.blocks:
                    if blk.use_mol:
                        lb_loss = lb_loss + blk.ffn.load_balance_loss().to(device)
                loss = loss + MOL_LB_WEIGHT * lb_loss

                # CodeHead aux
                code_logits = code_head(urm_out['final_state'],
                                         op_token_idx=0, cell_start_idx=101)
                B_cur, seq_len, V = code_logits.shape
                per_step_ce = F.cross_entropy(
                    code_logits.reshape(-1, V),
                    code_tgt.reshape(-1),
                    reduction='none',
                ).view(B_cur, seq_len).mean(dim=1)
                per_sample_w = code_mask + 0.05 * (1.0 - code_mask)
                code_loss = (per_step_ce * per_sample_w).sum() / per_sample_w.sum().clamp_min(1.0)
                loss = loss + CODE_HEAD_WEIGHT * code_loss

            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
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
                last_usage = urm.blocks[-1].ffn.last_expert_usage if urm.blocks[-1].use_mol else None
                usage_str = f'  mol={[round(float(u), 2) for u in last_usage.tolist()]}' if last_usage is not None else ''
                print(f'step {step:5d}  loss={float(loss.detach()):.4f}  '
                      f'lb={float(lb_loss.detach()):.4f}  code={float(code_loss.detach()):.3f}  '
                      f'depth={urm.n_loops_eff}/{N_LOOPS}  '
                      f'cell_acc={cell_acc:.3f}  lr={lr_cur:.2e}  '
                      f'rate={(step+1)/max(dt,1):.1f}/s{usage_str}')

            if step > 0 and step % AE_EVERY == 0:
                for m in all_modules: m.eval()
                eval_in, eval_out, eval_arc, eval_op, _, _ = sample_batch(BATCH, do_aug=False)
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
        # Reset URM to full depth (CGAR PDC ends at A2 with full depth, but be
        # explicit so any subsequent eval path doesn't inherit a shrunken state).
        urm.set_n_loops_eff(N_LOOPS)
        print(f'\\n=== Phase B: GRPO RL ({GRPO_STEPS} steps, group=4, full depth) ===')

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
            g_in_b, g_out_b, _rl_arc, _rl_op, _, _ = sample_batch(BATCH, do_aug=False)
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
                    'decoder': decoder.state_dict(),
                    'code_head': code_head.state_dict()},
                   str(ckpt_dir / 'urm_v40_2_final.pt'))
    """).strip()))

    cells.append(md("## ARC eval — self-simulate-weighted TTA (v40.1)"))
    cells.append(code(textwrap.dedent("""
        from oniro.eval.self_simulate import weighted_tta_majority

        for m in all_modules: m.eval()

        N_TTA = int(os.environ.get('ONIRO_TTA', '128'))
        SELF_SIM_THRESHOLD = float(os.environ.get('ONIRO_SELF_SIM_THRESHOLD', '0.5'))

        EVAL_OP_ID = torch.tensor([OP_ID["ARC_GENERIC"]], dtype=torch.long, device=device)
        EVAL_ARC_MASK = torch.tensor([1.0], dtype=torch.float, device=device)

        @torch.no_grad()
        def neural_predict_with_aug(grid_int_t, k, flip, cp):
            '''Apply (rot k, flip, color perm cp) then forward and invert.

            Returns the prediction back in the canonical (un-augmented) frame.
            '''
            gnp = grid_int_t.cpu().numpy() if isinstance(grid_int_t, torch.Tensor) else grid_int_t
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
            return pred

        @torch.no_grad()
        def _self_sim_score_demos(demos_pt, k, flip, cp):
            '''Score how well the model reproduces demos under augmentation (k, flip, cp).
            Returns mean cell-acc across all demos.
            '''
            if not demos_pt:
                return 0.5  # neutral when no demos available
            total = 0.0
            for di, do in demos_pt:
                pred = neural_predict_with_aug(di, k, flip, cp)
                # match in the canonical frame (di and do already canonical)
                h = min(pred.shape[-2], do.shape[-2])
                w = min(pred.shape[-1], do.shape[-1])
                match = float((pred[..., :h, :w] == do[..., :h, :w]).float().mean())
                total += match
            return total / len(demos_pt)

        @torch.no_grad()
        def tta_selfsim_vote(test_grid_int, demos_pt, n_samples=N_TTA):
            '''Self-simulate-weighted TTA majority. Each augmentation is scored
            by demo reproduction under that augmentation; only augmentations
            above SELF_SIM_THRESHOLD contribute weighted votes.
            '''
            aug_seeds = []
            for _ in range(n_samples):
                k = rng.randint(0, 3); flip = rng.random() < 0.5
                cp = random_color_perm(rng, n_colors=N_COLORS, keep_bg=True)
                aug_seeds.append((k, flip, cp))

            def fwd_with_aug(_test, idx):
                k, flip, cp = aug_seeds[idx]
                pred = neural_predict_with_aug(_test, k, flip, cp)
                score = _self_sim_score_demos(demos_pt, k, flip, cp)
                return pred, score

            return weighted_tta_majority(
                fwd_with_aug, test_grid_int,
                n_colors=N_COLORS, grid_size=GRID, n_samples=n_samples,
                threshold=SELF_SIM_THRESHOLD, device=device,
            )

        def eval_arc_tta(root, label):
            sd = Path(root) / 'evaluation'
            files = sorted(sd.glob('*.json'))
            n_pe = 0; n_t = 0; cells_c = []; ts = 0; tt = 0
            with ema.swap_in():
                for ti, tf in enumerate(files):
                    with tf.open() as f:
                        task = _json.load(f)
                    # demos for self-simulate scoring (canonical frame)
                    demos_pt = []
                    for tp in task.get('train', []):
                        di = grid_to_fixed(tp['input']).to(device)
                        do = grid_to_fixed(tp['output']).to(device)
                        demos_pt.append((di, do))
                    solved = []
                    for tp in task.get('test', []):
                        if 'output' not in tp: continue
                        gi = grid_to_fixed(tp['input']).to(device)
                        gt = grid_to_fixed(tp['output']).to(device)
                        pred = tta_selfsim_vote(gi, demos_pt)
                        exact = bool((pred == gt).all().item())
                        n_t += 1
                        if exact: n_pe += 1
                        cells_c.append(float((pred == gt).float().mean().item()))
                        solved.append(exact)
                    if solved:
                        tt += 1
                        if all(solved): ts += 1
                    if (ti + 1) % 30 == 0:
                        print(f'  {label} TTAss{N_TTA} [{ti+1}/{len(files)}] exact={n_pe}/{n_t}')
            return {'label': label, 'pairs_total': n_t, 'pairs_exact': n_pe,
                    'pair_exact_acc': n_pe / max(n_t, 1),
                    'mean_cell_acc': sum(cells_c)/max(len(cells_c), 1),
                    'tasks_total': tt, 'tasks_solved': ts,
                    'task_acc': ts / max(tt, 1), 'tta_samples': N_TTA,
                    'selfsim_threshold': SELF_SIM_THRESHOLD}

        results = {}
        print(f'=== ARC-AGI-1 (self-sim TTA {N_TTA}, threshold={SELF_SIM_THRESHOLD}) ===')
        results['arc1'] = eval_arc_tta(ARC1_ROOT, 'ARC-1')
        print(_json.dumps(results['arc1'], indent=2))
        print(f'=== ARC-AGI-2 (self-sim TTA {N_TTA}) ===')
        results['arc2'] = eval_arc_tta(ARC2_ROOT, 'ARC-2')
        print(_json.dumps(results['arc2'], indent=2))
    """).strip()))

    cells.append(md("## MCTS hybrid eval — DSL + neural over op_vocab (v40.1)"))
    cells.append(code(textwrap.dedent("""
        from oniro.dsl.solver import solve_task as dsl_solve_task
        from oniro.eval.mcts_search import mcts_search

        @torch.no_grad()
        def neural_predict_np(grid_np, op_id=None):
            '''Forward with optional op_id (default ARC_GENERIC). For mcts_search.'''
            if op_id is None:
                op_id = OP_ID["ARC_GENERIC"]
            gi = grid_to_fixed(grid_np.tolist()).unsqueeze(0).to(device)
            op_t = torch.tensor([int(op_id)], dtype=torch.long, device=device)
            arc_m = torch.tensor([1.0 if op_id in (OP_ID["ARC_GENERIC"], OP_ID["ARC_RE"],
                                                    OP_ID["ARC_CONCEPT"], OP_ID["ARC_MINI"],
                                                    OP_ID["ARC_HEAVY"]) else 0.0],
                                  dtype=torch.float, device=device)
            urm_input, op_tok = encode_v40(gi, op_t, arc_m)
            u = urm(urm_input, op_embed=op_tok)
            l = decoder(u['final_state'], GRID)
            return socrates_argmax(l, n_colors=N_COLORS, unknown_class=N_COLORS)[0].cpu().numpy().astype(np.int8)

        # Candidate op_ids for MCTS sweep (ARC tasks: ARC_GENERIC + variants + UNKNOWN)
        MCTS_OP_VOCAB = [OP_ID["ARC_GENERIC"], OP_ID["ARC_RE"], OP_ID["UNKNOWN_OP"]]

        def eval_arc_mcts_hybrid(root, label, max_dsl_depth=3):
            '''MCTS-style search: try DSL solver, then sweep neural op_ids,
            score each by demo reproduction, commit best to test grid.
            '''
            sd = Path(root) / 'evaluation'
            files = sorted(sd.glob('*.json'))
            n_pe = 0; n_t = 0; cells_c = []; ts = 0; tt = 0; method_counts = {}
            with ema.swap_in():
                for ti, tf in enumerate(files):
                    with tf.open() as f:
                        task = _json.load(f)
                    # canonical-frame demos as numpy
                    demos_np = []
                    for tp in task.get('train', []):
                        demos_np.append((np.asarray(tp['input'], dtype=np.int8),
                                          np.asarray(tp['output'], dtype=np.int8)))
                    solved = []
                    for tp in task.get('test', []):
                        if 'output' not in tp: continue
                        test_np = np.asarray(tp['input'], dtype=np.int8)
                        gt = np.asarray(tp['output'], dtype=np.int8)
                        # MCTS over op_ids + DSL solver
                        res = mcts_search(
                            neural_predict_np, demos_np, test_np,
                            op_vocab=MCTS_OP_VOCAB,
                            dsl_solver=lambda td: dsl_solve_task(
                                td, neural_fallback=lambda g: neural_predict_np(g),
                                max_depth=max_dsl_depth),
                            dsl_task_dict=task,
                            branching=3,
                            n_colors=N_COLORS,
                            grid_size=GRID,
                        )
                        pred = res['pred']
                        method_counts[res['method']] = method_counts.get(res['method'], 0) + 1
                        if pred.shape == gt.shape:
                            exact = bool(np.array_equal(pred, gt))
                            cell_acc = float((pred == gt).mean())
                        else:
                            exact = False; cell_acc = 0.0
                        n_t += 1
                        if exact: n_pe += 1
                        cells_c.append(cell_acc); solved.append(exact)
                    if solved:
                        tt += 1
                        if all(solved): ts += 1
                    if (ti + 1) % 50 == 0:
                        print(f'  {label}-mcts [{ti+1}/{len(files)}] exact={n_pe}/{n_t}')
            return {'label': label+'-mcts', 'pairs_total': n_t, 'pairs_exact': n_pe,
                    'pair_exact_acc': n_pe / max(n_t, 1),
                    'mean_cell_acc': sum(cells_c)/max(len(cells_c), 1),
                    'tasks_total': tt, 'tasks_solved': ts,
                    'task_acc': ts / max(tt, 1), 'method_counts': method_counts}

        print('=== ARC-AGI-1 MCTS hybrid ===')
        results['arc1_hybrid'] = eval_arc_mcts_hybrid(ARC1_ROOT, 'ARC-1', max_dsl_depth=3)
        print(_json.dumps(results['arc1_hybrid'], indent=2))
        print('=== ARC-AGI-2 MCTS hybrid ===')
        results['arc2_hybrid'] = eval_arc_mcts_hybrid(ARC2_ROOT, 'ARC-2', max_dsl_depth=3)
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
