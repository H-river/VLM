"""Build the Stage 1D 25-record base LLM probe subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


NORMAL_CATEGORIES = {"normal_edit"}
PROBE_COMPOSITION = {
    "normal": 10,
    "constraint": 5,
    "invalid": 5,
    "ambiguous_multi_intent": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Stage 1D base LLM probe subset.")
    parser.add_argument("--data", required=True, help="Stage 1 benchmark JSONL")
    parser.add_argument("--labels", required=True, help="Stage 1 labels JSONL")
    parser.add_argument("--out", required=True, help="Output 25-row probe JSONL")
    parser.add_argument("--labels-out", required=True, help="Output 25-row probe labels JSONL")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(obj)
    return rows


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _pick_labels(labels: list[dict]) -> list[dict]:
    selected: list[dict] = []

    normal = [row for row in labels if row.get("category") in NORMAL_CATEGORIES]
    if len(normal) < PROBE_COMPOSITION["normal"]:
        raise ValueError(f"need {PROBE_COMPOSITION['normal']} normal labels, found {len(normal)}")
    selected.extend(normal[: PROBE_COMPOSITION["normal"]])

    for category in ("constraint", "invalid", "ambiguous_multi_intent"):
        rows = [row for row in labels if row.get("category") == category]
        need = PROBE_COMPOSITION[category]
        if len(rows) < need:
            raise ValueError(f"need {need} {category} labels, found {len(rows)}")
        selected.extend(rows[:need])

    return selected


def main() -> None:
    args = parse_args()
    data_rows = _load_jsonl(Path(args.data))
    label_rows = _load_jsonl(Path(args.labels))
    data_by_id = {row["id"]: row for row in data_rows}

    selected_labels = _pick_labels(label_rows)
    selected_records = []
    missing = []
    for label in selected_labels:
        record_id = label["record_id"]
        record = data_by_id.get(record_id)
        if record is None:
            missing.append(record_id)
        else:
            selected_records.append(record)
    if missing:
        raise ValueError(f"labels reference missing benchmark records: {missing}")

    _write_jsonl(selected_records, Path(args.out))
    _write_jsonl(selected_labels, Path(args.labels_out))
    print(
        json.dumps(
            {
                "out": args.out,
                "labels_out": args.labels_out,
                "composition": {
                    "normal": sum(row.get("category") in NORMAL_CATEGORIES for row in selected_labels),
                    "constraint": sum(row.get("category") == "constraint" for row in selected_labels),
                    "invalid": sum(row.get("category") == "invalid" for row in selected_labels),
                    "ambiguous_multi_intent": sum(
                        row.get("category") == "ambiguous_multi_intent" for row in selected_labels
                    ),
                },
                "records": len(selected_records),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
