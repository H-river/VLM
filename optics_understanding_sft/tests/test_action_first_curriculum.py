from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from optics_understanding_sft.build_action_first_curriculum import (
    ANCHOR_TASKS,
    prepare,
    stratified_anchor_sample,
)


def row(task: str, status: str, index: int) -> dict:
    target = json.dumps({"status": status, "answer": {}})
    return {
        "example_id": f"{task}_{status}_{index}",
        "task_type": task,
        "completion": [{"content": [{"type": "text", "text": target}]}],
    }


class ActionFirstCurriculumTests(unittest.TestCase):
    def test_prepare_preserves_earliest_source_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare(
                {
                    "example_id": "record__older_curriculum",
                    "source_example_id": "record",
                    "images": [],
                },
                Path(directory),
                "new_curriculum",
            )
        self.assertEqual(prepared["source_example_id"], "record")
        self.assertEqual(
            prepared["example_id"], "record__older_curriculum__new_curriculum"
        )

    def test_anchor_sampling_is_exact_balanced_unique_and_deterministic(self) -> None:
        rows = []
        for task in ANCHOR_TASKS:
            rows.extend(row(task, "alpha", index) for index in range(10))
            rows.extend(row(task, "beta", index) for index in range(10))
        first = stratified_anchor_sample(rows, per_task=8, seed=19)
        second = stratified_anchor_sample(rows, per_task=8, seed=19)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8 * len(ANCHOR_TASKS))
        self.assertEqual(len({item["example_id"] for item in first}), len(first))
        for task in ANCHOR_TASKS:
            task_rows = [item for item in first if item["task_type"] == task]
            self.assertEqual(len(task_rows), 8)
            statuses = [
                json.loads(item["completion"][0]["content"][0]["text"])["status"]
                for item in task_rows
            ]
            self.assertEqual(statuses.count("alpha"), 4)
            self.assertEqual(statuses.count("beta"), 4)


if __name__ == "__main__":
    unittest.main()
