from __future__ import annotations

import unittest

from optics_understanding_sft.build_corrective_curriculum import grouped_order


class CorrectiveCurriculumTests(unittest.TestCase):
    def test_matched_units_remain_adjacent_and_order_is_deterministic(self) -> None:
        rows = [
            {"example_id": "a1", "match_group_id": "a"},
            {"example_id": "single"},
            {"example_id": "a0", "match_group_id": "a"},
            {"example_id": "b1", "match_group_id": "b"},
            {"example_id": "b0", "match_group_id": "b"},
        ]
        first, units, maximum = grouped_order(rows, 7)
        second, _, _ = grouped_order(rows, 7)
        self.assertEqual(first, second)
        self.assertEqual(units, 3)
        self.assertEqual(maximum, 2)
        positions = {
            match_id: [index for index, row in enumerate(first) if row.get("match_group_id") == match_id]
            for match_id in ("a", "b")
        }
        self.assertTrue(all(indices[1] - indices[0] == 1 for indices in positions.values()))


if __name__ == "__main__":
    unittest.main()
