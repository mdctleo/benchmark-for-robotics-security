# RoboBench

Benchmark for jailbreak attacks and defenses on vision-language robot planners.
A scene is one or more camera frames with a malicious and a benign instruction;
attacks rewrite the malicious instruction and, for visual attacks, the frame;
defenses screen the request; scorers turn accept/reject labels into ASR, SR, UR
and SU-HM.

Scenes are released as Hugging Face artifacts (dataset `mdctleo/RoboJailBench`),
one config per artifact. A vanilla artifact (`DROID`, `ROBOVQA`, `RH20T`,
`robo2vlm`, `car`, `RJB-Instructions`) holds the original instructions; an
attack artifact (`<attack>_<base>`, e.g. `robopair_DROID`) holds the same frames
with the attack's rewritten instruction. Columns are `id`, `dataset`,
`category`, `images`, `benign`, `malicious`, `target`.

## Install

```
uv sync
```

API keys are read from the environment / `.env`. `OPENAI_API_KEY` is required
(target models, judge). `HF_TOKEN` is needed only to push artifacts.
`ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` are needed only when a model of that
family is selected by name. Optional extras: `roboguard`, `scenegraph`,
`shawshank` (see `pyproject.toml`).

## Layout

```
src/
  dataset/generate_prompt_from_img.py   write a dataset's <base>_instr.json from its frames
  build_attacks.py                      run an attack over a vanilla artifact -> reviewable raw tree
  build_artifacts.py                    build one HF parquet artifact from a raw tree
  build_scene_graphs.py                 build the scene graphs RoboGuard needs
  main_attack_defense.py                run defenses over attack/benign prompts (ASR, UR)
  resolve_unknowns.py                   judge-label responses the keyword parser cannot
  analysis_attack_defense.py            score: ASR, SR, UR, SU-HM and the LaTeX table
  main.py, analysis.py                  plain (no attack, no defense) malicious-vs-benign baseline
  backfill_ids.py                       one-off: add artifact ids to pre-artifact pickles
  robobench/                            the package: load_scenes, label, score; attacks, defenses, models

data/<name>/                raw source trees: <name>/dataset/<frame>, <name>_instr.json
generated_attacks/<attack>_<base>/   raw attack trees from build_attacks.py (reviewable before build)
artifacts/                  built artifacts: data/<name>/train.parquet, index.json, README.md (HF card)
scene_graphs/<source>/<frame stem>.json   per-scene graphs for RoboGuard
workdir/                    experiment output: model responses and scores
```

All commands run from this directory with `uv run python`.

---

# Pipeline A — Produce artifacts

Maintainer side: turn an attack into a released HF artifact. The vanilla base
artifacts already exist on Hugging Face, so this starts from an attack.

### 1. Generate an attack (once per attack per base)

```
uv run python src/build_attacks.py --attack badrobot_cj --dataset DROID
```

`build_attacks.py` reads the base artifact (from HF, or `--artifacts <dir>` for a
local one), rewrites the malicious instruction per frame — and the frame itself
for visual attacks — and writes a reviewable raw tree
`generated_attacks/<attack>_<base>/`. Attacks: `badrobot_cd`, `badrobot_cj`,
`badrobot_sm`, `robopair`, `shawshank`. Omit `--dataset` to run every base;
`--limit N` is a smoke test. A run resumes from the frames already written.

### 2. Build the artifact

```
uv run python src/build_artifacts.py generated_attacks/badrobot_cj_DROID
```

Input: the reviewable raw tree from step 1 (review it first). Output:
`artifacts/data/<attack>_<base>/train.parquet`, with `artifacts/index.json`, the
shipped `src/robobench/index.json`, and the HF card `artifacts/README.md`
regenerated. The directory name is the artifact name. Push to Hugging Face to
release (`hf upload mdctleo/RoboJailBench ...`).

### 3. Scene graphs for RoboGuard (once per base)

```
uv run python src/build_scene_graphs.py
```

RoboGuard plans over a scene graph, not pixels. This reads the vanilla artifacts
from HF and writes `scene_graphs/<source>/<frame stem>.json`, where the runner
looks. Scenes that already have a graph are skipped. `--dataset D` limits it.

> Note: `build_scene_graphs.py` may need adjusting once sequential (multi-frame)
> scenes are brought over.

---

# Pipeline B — Use artifacts for experiments

## B1. Downstream collaborators

You have a model and want a RoboBench score. Load an artifact, run your model,
hand the responses back; the package labels and scores them. See
`downstream_bencharmk_design.md` for the full API.

```python
import robobench

scenes = robobench.load_scenes("robopair_DROID")   # one artifact; images are lazy PIL
responses = [
    {"id": s["id"], "type": t, "response": my_model(s["images"], s[t])}
    for s in scenes.rows for t in ("benign", "malicious")
]
report = robobench.score(responses)   # labels (keyword + judge) then scores; needs OPENAI_API_KEY
```

Input: your `[{id, type, response}]` rows (list, DataFrame, or CSV/JSONL path),
from one artifact or many. Output: a JSON-serializable report — per-artifact and
overall ASR, SR, UR, SU-HM with standard errors and 95% CIs, plus coverage. The
judge is pinned to the version used in the paper and recorded in the report.

## B2. Reproduce the paper

### Run attacks against defenses

```
uv run python src/main_attack_defense.py --model gpt-5.4-mini-2026-03-17
```

Input: the attack and vanilla artifacts from HF. Runs each defense
(`no_defense`, `google_prompt`, `roboguard`) over the malicious prompts (ASR) and
the clean benign goals (UR). Output:
`workdir/attack_defense/<model>/{malicious,benign}/<dataset>/<attack>/<defense>.pkl`,
each row carrying the response and a parsed `is_accepted` label. Options:
`--dataset` / `--defense` (repeatable, default all), `--workers`, `--limit`.
Re-running resumes by `id`.

RoboGuard does not depend on the target model, so it runs once under its own path
(`workdir/attack_defense/roboguard/...`). It needs the `roboguard` extra, the
`spot` system install (`src/robobench/_vendor/README.md`), and the scene graphs
from step A4 (`--scene-graphs`, default `scene_graphs`).

### Resolve unlabeled responses

```
uv run python src/resolve_unknowns.py --model gpt-5.4-mini-2026-03-17
```

Some answers are neither a clear accept nor reject token. This re-applies the
keyword parser and sends the remainder to the judge, writing `is_accepted` back
into the response files and marking which rows the judge resolved.
`--input-dir workdir/plain` runs it over the baseline instead.

### Score

```
uv run python src/analysis_attack_defense.py --model gpt-5.4-mini-2026-03-17 \
    --model-label "GPT 5.4 Mini" --output workdir/attack_defense/gpt54mini.tex
```

Prints per-artifact rates and writes the LaTeX table. `--verify` cross-checks the
RoboGuard combination.

## Plain baseline (no attack, no defense)

```
uv run python src/main.py --model gpt-5.4-mini-2026-03-17
uv run python src/resolve_unknowns.py --input-dir workdir/plain --model gpt-5.4-mini-2026-03-17
uv run python src/analysis.py
```

`main.py` sends each frame's raw malicious and benign instruction to one target
model and writes `workdir/plain/<model>/<dataset>/{benign,malicious}.pkl` (the
same columns as the attack-defense cells, so `resolve_unknowns.py` and
`analysis.py` read them directly). `analysis.py` writes per-dataset metrics and
LaTeX tables to `workdir/plain/analysis/`.
