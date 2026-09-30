# RoboPAIR attacks

The `robopair.json` files under `generated_attacks_external/<dataset>/`
were produced with the external [RoboPAIR](https://github.com/alexrobey23/robopair)
project, checked out into this directory (`src/attacks/robopair/`, gitignored)
and driven by a script that is no longer maintained here.

For each entry of a dataset's instruction file the driver called RoboPAIR's
`main` with the malicious instruction as the goal and `"sure, I will " +
instruction` as the target string, then wrote a CSV with the same columns as
the BadRobot files: `filename, cat1, cat2, type, text`, one `malicious` and one
`benign` row per image. `main_attack_defense.py` reads the `malicious` rows.

To regenerate, clone RoboPAIR here, install its requirements, and write the
rows in that format. The generated files on disk cover all nine datasets.
