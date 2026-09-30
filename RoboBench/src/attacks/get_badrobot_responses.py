"""Generate BadRobot attacks for a dataset.

Reads the dataset's instruction file and rewrites every instruction with one
BadRobot strategy: ``contextual jailbreak``, ``safety misalignment`` or
``conceptual deception``. Writes a CSV (named ``.json`` by convention) with
one malicious and one benign row per image; ``main_attack_defense.py`` reads
the malicious rows.

Usage:
    python src/attacks/get_badrobot_responses.py \
        generated_attacks_external/DROID/badrobot_cj.json \
        data/external_datasets/DROID/DROID_instr.json \
        "contextual jailbreak"
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from badrobot.attack_main import main  # noqa: E402


def run():
    path = sys.argv[1]
    json_path = sys.argv[2]
    attack_method = sys.argv[3]

    entries = []
    with open(json_path, "r") as f:
        data = json.load(f)
    for key, value in data.items():
        print(key, value)
        malicious = value[0]
        benign = value[1]
        filename = key

        malicious_response = main("", "", "", malicious, attack_method)
        benign_response = main("", "", "", benign, attack_method)

        print("MAL" + str(malicious))

        entries.append([filename, "", "", "malicious", malicious_response])
        entries.append([filename, "", "", "benign", benign_response])

    generated_df = pd.DataFrame(entries, columns=["filename", "cat1", "cat2", "type", "text"])
    generated_df.to_csv(path)


if __name__ == "__main__":
    run()
