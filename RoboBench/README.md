# RoboBench

Benchmark for jailbreak attacks and defenses on vision-language robot planners.
Each dataset is a folder of camera frames, one malicious and one benign
instruction per frame, and a scene graph per frame for RoboGuard. Attacks
rewrite the malicious instructions; runners send them, with the image, to a
target model under each defense; scorers turn the accept/reject labels into
ASR, SR, UR and SU-HM.

## Layout

```
src/
  dataset/generate_prompt_from_img.py          build a dataset's instruction file from its images
  attacks/get_badrobot_responses.py            generate the three BadRobot attacks
  attacks/badrobot/                            BadRobot strategies (contextual jailbreak,
                                               safety misalignment, conceptual deception)
  attacks/robopair/README.md                   how the RoboPAIR attack files were produced (external tool)
  defenses/roboguard/semantics_from_img_unstructured.py
                                               build the scene graphs RoboGuard needs
  defenses/roboguard/RoboGuard/                external RoboGuard + SPINE checkout (gitignored)
  main_attack_defense.py                       run a target model under no defense / Google prompt
  main_roboguard.py                            run RoboGuard on every attack and benign goal
  resolve_unknowns.py                          judge-label responses the keyword parser cannot
  analysis_attack_defense.py                   score: ASR, SR, UR, SU-HM and the LaTeX table
  main.py, analysis.py                         plain (no attack) malicious-vs-benign baseline

data/                    datasets: <name>/dataset/*.png, one .json scene graph per frame,
                         and <name>_instr.json
generated_attacks_external/<dataset>/{badrobot_cd,badrobot_cj,badrobot_sm,robopair}.json
                         generated attacks, one file per attack per dataset
workdir/                 pipeline output: model responses and scores
```

All commands run from this directory with `uv run python`. API keys are read
from `.env`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
`DASHSCOPE_API_KEY`.

## 1. Build a dataset (once per dataset)

Put the frames in `data/<group>/<name>/dataset/frame_<i>.png`, then:

```
uv run python src/dataset/generate_prompt_from_img.py data/external_datasets/DROID
uv run python src/defenses/roboguard/semantics_from_img_unstructured.py data/external_datasets/DROID
```

The first writes `data/external_datasets/DROID/DROID_instr.json`. The second
writes one scene graph next to each frame; it skips frames that already have
one. Register the dataset in `DATASETS` in `src/main_attack_defense.py`.

## 2. Generate attacks (once per dataset)

```
uv run python src/attacks/get_badrobot_responses.py generated_attacks_external/DROID/badrobot_cj.json data/external_datasets/DROID/DROID_instr.json "contextual jailbreak"
uv run python src/attacks/get_badrobot_responses.py generated_attacks_external/DROID/badrobot_sm.json data/external_datasets/DROID/DROID_instr.json "safety misalignment"
uv run python src/attacks/get_badrobot_responses.py generated_attacks_external/DROID/badrobot_cd.json data/external_datasets/DROID/DROID_instr.json "conceptual deception"
```

RoboPAIR attacks come from the external RoboPAIR tool; see
`src/attacks/robopair/README.md`. All nine datasets already have all four
attack files.

## 3. Run attacks against defenses

Target model under the prompt-level defenses (`no_defense`, `google_prompt`):

```
uv run python src/main_attack_defense.py --model gpt-5.4-mini-2026-03-17
```

Options: `--dataset <name>` and `--defense <name>` (repeatable, default all),
`--workers`, `--retry-count`, `--limit`. Output goes to
`workdir/attack_defense/<model>/{malicious,benign}/...pkl`. Re-running resumes
from existing files.

RoboGuard, which does not depend on the target model, so one run serves all:

```
uv run python src/main_roboguard.py
```

Requires the RoboGuard checkout at `src/defenses/roboguard/RoboGuard/` and
`spot` (see the path setup at the top of the script). Output goes to `workdir/attack_defense/roboguard/`.

## 4. Resolve unlabeled responses

Some responses are neither a clear accept nor reject token. This re-applies
the keyword parser and sends the remainder to the judge model, writing labels
back in place:

```
uv run python src/resolve_unknowns.py --model gpt-5.4-mini-2026-03-17
```

## 5. Score

```
uv run python src/analysis_attack_defense.py --model gpt-5.4-mini-2026-03-17 --model-label "GPT 5.4 Mini" --output workdir/attack_defense/gpt54mini.tex
```

Prints the per-dataset rates and writes the LaTeX table. `--verify` cross-checks
the RoboGuard combination.

## Plain baseline (no attacks)

```
uv run python src/main.py
uv run python src/analysis.py
```

`main.py` queries each model in `MODEL_NAMES` with the raw malicious and
benign instructions and writes `workdir/plain/<dataset>_model_responses.pkl`;
`analysis.py` writes per-dataset metrics and LaTeX tables to
`workdir/plain/analysis/`.
