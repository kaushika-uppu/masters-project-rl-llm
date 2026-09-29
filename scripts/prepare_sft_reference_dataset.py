"""Build the local SFT format from phase-2 survivor records.

The survivor file supplies the subset of original questions to retain.  The
reference dataset supplies the gold ``pos``/``neg`` question-response pairs.
Generated ``rollouts`` and ``sft_rollouts`` are deliberately not used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path} line {line_number}")
            yield row


def unwrap_variant(value: Any) -> dict[str, Any] | None:
    """Accept the dict/list/string forms used by local/Hugging Face exports."""
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            result = unwrap_variant(item)
            if result is not None:
                return result
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return unwrap_variant(parsed)
    return None


def reference_variant(value: Any, name: str) -> dict[str, str]:
    variant = unwrap_variant(value)
    if variant is None:
        raise ValueError(f"Reference record has no usable {name} variant")

    question = variant.get("question", variant.get("statement"))
    response = variant.get("response", variant.get("proof", variant.get("solution")))
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Reference {name} variant has no question")
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"Reference {name} variant has no response")
    return {"question": question, "response": response}


def load_reference_map(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(Path(args.reference_file)) if args.reference_file else None
    if rows is None:
        from datasets import load_dataset

        rows = load_dataset(args.reference_dataset, split="train")

    reference: dict[str, dict[str, Any]] = {}
    for row in rows:
        original_question = row.get("ori_question")
        if not isinstance(original_question, str) or not original_question:
            continue
        reference.setdefault(original_question, row)
    return reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/sft_combined_phase2_survivors.jsonl",
        help="Phase-2 survivor JSONL file",
    )
    parser.add_argument(
        "--output",
        default="data/sft_reference_training.jsonl",
        help="Output JSONL consumed by the local SFT formatter",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--reference-dataset",
        default="Jiahao004/DeepTheorem",
        help="Hugging Face reference dataset name",
    )
    source.add_argument(
        "--reference-file",
        help="Local JSONL export of the original reference dataset",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Survivor file not found: {input_path}")

    reference = load_reference_map(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    matched = 0
    missing = 0
    duplicate = 0

    with output_path.open("w", encoding="utf-8") as output:
        for survivor in read_jsonl(input_path):
            original_question = survivor.get("ori_question")
            if not isinstance(original_question, str) or not original_question:
                missing += 1
                continue
            if original_question in seen:
                duplicate += 1
                continue
            seen.add(original_question)

            original = reference.get(original_question)
            if original is None:
                missing += 1
                continue

            try:
                converted = {
                    "pos": reference_variant(original.get("pos"), "pos"),
                    "neg": reference_variant(original.get("neg"), "neg"),
                }
            except ValueError:
                missing += 1
                continue

            output.write(json.dumps(converted, ensure_ascii=False) + "\n")
            matched += 1

    print(f"Reference records loaded: {len(reference)}")
    print(f"Matched training records: {matched}")
    print(f"Unmatched/invalid survivor records: {missing}")
    print(f"Duplicate survivor rows skipped: {duplicate}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()