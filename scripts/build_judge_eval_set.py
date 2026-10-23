#!/usr/bin/env python3
"""Attach reference proofs from DeepTheorem to the held-out eval file for the judge eval.

The eval file (data/deeptheorem_eval.jsonl) has only {id, statement, label, ...}; the judge
eval needs a reference proof per row. This writes a NEW file (the eval file is never
modified) with the original rows plus "reference" and "reference_check".

  python scripts/build_judge_eval_set.py \
      --eval-file data/deeptheorem_eval.jsonl --output data/deeptheorem_judge_eval.jsonl

Reference per variant (same mapping as scripts/create_complete_dt_eval.py):
  original      -> theorem's `proof` column   (statement: informal_theorem_qa / informal_theorem)
  entailing/pos -> row["pos"]["response"]     (statement: row["pos"]["question"])
  contradictory/neg -> row["neg"]["response"] (statement: row["neg"]["question"])
Variant comes from row["variant_type"], else the id suffix; never inferred from the label.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.deeptheorem import (  # noqa: E402
    _PROOF_KEYS,
    _STATEMENT_KEYS,
    _coerce_variant_item,
)

_SUFFIX_RE = re.compile(r"_(original|entailing|contradictory|ori|pos|neg)$")
_VARIANT_ALIASES = {
    "original": "original", "ori": "original",
    "entailing": "pos", "pos": "pos",
    "contradictory": "neg", "neg": "neg",
}
# columns holding the original theorem statement (create_complete_dt_eval uses informal_theorem_qa)
_ORIGINAL_STATEMENT_KEYS = ("informal_theorem_qa",) + tuple(_STATEMENT_KEYS)


def normalize_variant(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return _VARIANT_ALIASES.get(value.strip().lower())


def parse_id(row: dict) -> tuple[str, Optional[str]]:
    """Return (base_id, variant) with variant in {original,pos,neg,None}."""
    rid = str(row.get("id", ""))
    m = _SUFFIX_RE.search(rid)
    suffix_variant = _VARIANT_ALIASES[m.group(1)] if m else None
    if row.get("base_id") not in (None, ""):
        base = str(row["base_id"])
    else:
        base = _SUFFIX_RE.sub("", rid)
    variant = normalize_variant(row.get("variant_type")) or suffix_variant
    return base, variant


def parse_variant_cell(value: Any) -> Optional[dict]:
    """pos/neg cell may be a dict or a stringified dict (python repr or JSON)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            obj = ast.literal_eval(value)
            if isinstance(obj, dict):
                return obj
        except (ValueError, SyntaxError):
            pass
        return _coerce_variant_item(value)
    return None


def _norm(s: Any) -> str:
    return " ".join(str(s).split()) if s is not None else ""


def _first_nonempty(row: dict, keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _lookup(ds_row: dict, variant: str) -> tuple[Optional[str], list[str]]:
    """Return (reference, candidate_statements) for this variant."""
    if variant == "original":
        cands = [ds_row[k] for k in _ORIGINAL_STATEMENT_KEYS if isinstance(ds_row.get(k), str)]
        return _first_nonempty(ds_row, _PROOF_KEYS), cands
    cell = parse_variant_cell(ds_row.get(variant))
    if not cell:
        return None, []
    resp = cell.get("response")
    ref = resp if isinstance(resp, str) and resp.strip() else None
    q = cell.get("question")
    return ref, [q] if isinstance(q, str) else []


def attach_references(eval_rows: Iterable[dict], ds_rows_by_id: dict) -> tuple[list[dict], dict]:
    """Pure function: returns (output rows, stats). No I/O."""
    out: list[dict] = []
    skipped: Counter = Counter()
    total = mismatches = 0
    by_id = {str(k): v for k, v in ds_rows_by_id.items()}
    for row in eval_rows:
        total += 1
        base, variant = parse_id(row)
        if variant is None:
            skipped["unresolved_variant"] += 1
            continue
        ds_row = by_id.get(base)
        if ds_row is None:
            skipped["missing_base_id"] += 1
            continue
        ref, cands = _lookup(ds_row, variant)
        if not ref:
            skipped["missing_response"] += 1
            continue
        target = _norm(row.get("statement"))
        ok = any(_norm(c) == target for c in cands)
        new = dict(row)
        new["reference"] = ref
        new["reference_check"] = "ok" if ok else "statement_mismatch"
        mismatches += 0 if ok else 1
        out.append(new)
    stats = {"total": total, "written": len(out), "skipped": dict(skipped),
             "skipped_total": sum(skipped.values()), "statement_mismatch": mismatches}
    return out, stats


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_ds_rows_by_id(name: str, split: str, wanted: Optional[set] = None) -> dict:
    """Lazy HF load (needs `datasets` + network/cache). Keeps only wanted ids if given."""
    from src.data.deeptheorem import load_deeptheorem
    ds = load_deeptheorem(split=split, name=name)
    out = {}
    for r in ds:
        rid = str(r.get("id"))
        if wanted is None or rid in wanted:
            out[rid] = dict(r)
    return out


def main(argv=None, row_source=None) -> int:
    """row_source: optional callable(name, split) -> iterable of dataset rows (for tests)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-file", default="data/deeptheorem_eval.jsonl")
    ap.add_argument("--output", default="data/deeptheorem_judge_eval.jsonl")
    ap.add_argument("--hf-name", default="Jiahao004/DeepTheorem")
    ap.add_argument("--split", default="train")
    args = ap.parse_args(argv)

    if os.path.abspath(args.output) == os.path.abspath(args.eval_file):
        print("ERROR: --output must differ from --eval-file (refusing to overwrite the eval file).",
              file=sys.stderr)
        return 2
    if not os.path.exists(args.eval_file):
        print(f"ERROR: eval file not found: {args.eval_file}", file=sys.stderr)
        return 2

    eval_rows = read_jsonl(args.eval_file)
    wanted = {parse_id(r)[0] for r in eval_rows}
    if row_source is not None:
        ds_by_id = {str(r.get("id")): r for r in row_source(args.hf_name, args.split)
                    if str(r.get("id")) in wanted}
    else:
        ds_by_id = load_ds_rows_by_id(args.hf_name, args.split, wanted)

    out, stats = attach_references(eval_rows, ds_by_id)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[build_judge_eval_set] total={stats['total']} written={stats['written']} "
          f"skipped={stats['skipped_total']} {stats['skipped']} "
          f"statement_mismatch={stats['statement_mismatch']}")
    if out:
        print(f"[build_judge_eval_set] wrote {args.output}")
    if not out:
        print("ERROR: 0 rows written; nothing to evaluate.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
