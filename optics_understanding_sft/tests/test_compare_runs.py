from __future__ import annotations

import unittest

from optics_understanding_sft.compare_runs import compare, percentile, task_macro
from optics_understanding_sft.evaluate import TASKS


def rows(offset: float) -> list[dict[str, object]]:
    return [
        {
            "example_id": f"g{group_index}_{task}",
            "group_id": f"g{group_index}",
            "task_type": task,
            "task_score": min(1.0, 0.2 + group_index * 0.1 + offset),
        }
        for group_index in range(3)
        for task in TASKS
    ]


class CompareRunsTests(unittest.TestCase):
    def test_task_macro_is_equal_task_average(self) -> None:
        self.assertAlmostEqual(task_macro(rows(0.0)), 0.3)

    def test_paired_bootstrap_is_deterministic_and_detects_uniform_gain(self) -> None:
        result = compare(
            {"base": rows(0.0), "better": rows(0.1)}, reference="base", replicates=200, seed=7
        )
        comparison = result["comparisons"]["better"]
        self.assertAlmostEqual(comparison["point_difference"], 0.1)
        self.assertAlmostEqual(comparison["ci95"][0], 0.1)
        self.assertAlmostEqual(comparison["ci95"][1], 0.1)
        self.assertEqual(comparison["probability_candidate_better"], 1.0)

    def test_percentile_interpolates(self) -> None:
        self.assertAlmostEqual(percentile([0.0, 10.0], 0.25), 2.5)


if __name__ == "__main__":
    unittest.main()
