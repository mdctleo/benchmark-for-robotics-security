# read "cases.txt" and print the content
import json
import os
import unittest

tc = unittest.TestCase()

with open("cases.txt", "r") as f:
    content = f.read()

blocks = content.split("\n\n")

current_dataset = ""
temp = {}

for block in blocks:
    if "_" in block:
        if len(temp) > 0:
            assert current_dataset, "current_dataset is empty while temp has entries"
            # save temp to {current_dataset}/{current_dataset}_instr.json
            with open(
                f"{current_dataset}/{current_dataset}_instr.json",
                "w",
            ) as f:
                json.dump(temp, f, indent=4)
            temp = {}
        current_dataset = block.strip()
    else:
        assert current_dataset, f"current_dataset is empty for block: {block!r}"

        lines = block.splitlines()
        if len(lines) == 0:
            continue

        assert lines[0].startswith("$"), lines[0]
        if not lines[1].strip('"').startswith("Explanation:"):
            print(
                f"Warning: Explanation line does not start with 'Explanation:' {lines[1]}"
            )
        assert lines[2].strip('"').startswith("Violates"), lines[2]

        name_begin = lines[0]

        assert (len(lines) - 3) % 3 == 0, lines

        for i in range(3, len(lines), 3):
            name = lines[i].strip()
            malicious_with_title = lines[i + 1].strip().strip('"')
            assert malicious_with_title.startswith("Malicious:"), malicious_with_title
            malicious = malicious_with_title[len("Malicious:") :].strip()

            benign_with_title = lines[i + 2].strip().strip('"')
            assert benign_with_title.startswith("Benign: "), benign_with_title
            benign = benign_with_title[len("Benign:") :].strip()

            key = f"./data/main_dataset_5/{current_dataset}/dataset/{name_begin}{name}.png"
            # assert the file exists
            # assert os.path.exists(key), f"{key} does not exist"
            temp[key] = [malicious, benign]

# save the last temp to {current_dataset}/{current_dataset}_instr.json
if len(temp) > 0:
    assert current_dataset, "current_dataset is empty while temp has entries"
    with open(f"{current_dataset}/{current_dataset}_instr.json", "w") as f:
        json.dump(temp, f, indent=4)
