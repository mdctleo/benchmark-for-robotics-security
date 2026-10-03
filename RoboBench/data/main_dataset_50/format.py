# read "expanded_prompts.txt" and write one <category>_instr.json per category
import json
import os

DATASET = "main_dataset_50"
SOURCE = "expanded_prompts.txt"

with open(SOURCE, "r") as f:
    content = f.read()

blocks = content.split("\n\n")

current_dataset = ""
temp = {}


def flush(current_dataset, temp):
    assert current_dataset, "current_dataset is empty while temp has entries"
    os.makedirs(current_dataset, exist_ok=True)
    with open(f"{current_dataset}/{current_dataset}_instr.json", "w") as f:
        json.dump(temp, f, indent=4)
    print(f"{current_dataset}: {len(temp)} pairs")


for block in blocks:
    lines = block.splitlines()
    if len(lines) == 0:
        continue

    # a one-line block that is not a subcategory heading names the category
    if len(lines) == 1 and not lines[0].startswith("$"):
        if len(temp) > 0:
            flush(current_dataset, temp)
            temp = {}
        current_dataset = lines[0].strip()
        continue

    assert current_dataset, f"current_dataset is empty for block: {block!r}"

    assert lines[0].startswith("$"), lines[0]
    assert lines[1].strip('"').startswith("Explanation:"), lines[1]
    assert lines[2].strip('"').startswith("Violates"), lines[2]

    name_begin = lines[0]

    assert (len(lines) - 3) % 3 == 0, lines

    for i in range(3, len(lines), 3):
        name = lines[i].strip()
        malicious_with_title = lines[i + 1].strip().strip('"')
        assert malicious_with_title.startswith("Malicious:"), malicious_with_title
        malicious = malicious_with_title[len("Malicious:") :].strip()

        benign_with_title = lines[i + 2].strip().strip('"')
        assert benign_with_title.startswith("Benign:"), benign_with_title
        benign = benign_with_title[len("Benign:") :].strip()

        key = f"./data/{DATASET}/{current_dataset}/dataset/{name_begin}{name}.png"
        # the images are not generated yet, so existence is not asserted
        # assert os.path.exists(key), f"{key} does not exist"
        assert key not in temp, f"duplicate key {key}"
        temp[key] = [malicious, benign]

if len(temp) > 0:
    flush(current_dataset, temp)
