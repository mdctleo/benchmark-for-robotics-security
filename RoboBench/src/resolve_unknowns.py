"""Resolve responses that the acceptance/rejection vocabulary cannot label.

Target models sometimes answer in their own words -- "OK, this is a benign
navigation request", "I can't help with that", or a block of action code --
instead of the requested verdict token. Those responses carry a clear
accept/reject meaning that keyword matching does not capture.

This pass re-applies the keyword parser and sends whatever remains to the judge
model. It writes ``is_accepted`` back into the response files and records which
rows the judge resolved, so keyword- and judge-derived labels stay
distinguishable.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from robobench.labeling import judge_response, parse_label


def resolve_file(client: OpenAI, path: Path, workers: int,
                 retry_count: int) -> tuple[int, int]:
    frame = pd.read_pickle(path)
    if "raw_response" not in frame.columns:
        return 0, 0

    frame["is_accepted"] = frame["raw_response"].apply(parse_label)
    if "judge_resolved" not in frame.columns:
        frame["judge_resolved"] = False
    frame["judge_resolved"] = False

    todo = [
        index
        for index in frame.index
        if frame.at[index, "is_accepted"] is None
        and not pd.isna(frame.at[index, "raw_response"])
    ]
    if not todo:
        frame.to_pickle(path)
        return 0, 0

    lock = Lock()
    unresolved = 0

    def work(index: int):
        nonlocal unresolved
        verdict = judge_response(
            client=client,
            response_text=frame.at[index, "raw_response"],
            retry_count=retry_count,
        )
        with lock:
            if verdict is None:
                unresolved += 1
            else:
                frame.at[index, "is_accepted"] = verdict
                frame.at[index, "judge_resolved"] = True

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, todo))

    frame.to_pickle(path)
    return len(todo), unresolved


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", dest="models", required=True)
    parser.add_argument("--input-dir", default="workdir/attack_defense")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retry-count", type=int, default=5)
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    client = OpenAI()

    for model in args.models:
        model_dir = Path(args.input_dir) / model.replace("/", "_")
        paths = sorted(model_dir.rglob("*.pkl"))
        judged = 0
        unresolved = 0
        for path in tqdm(paths, desc=model, leave=False):
            count, still_unknown = resolve_file(
                client=client,
                path=path,
                workers=args.workers,
                retry_count=args.retry_count,
            )
            judged += count
            unresolved += still_unknown
        print(f"{model}: judged {judged}, still unresolved {unresolved}")


if __name__ == "__main__":
    main()
