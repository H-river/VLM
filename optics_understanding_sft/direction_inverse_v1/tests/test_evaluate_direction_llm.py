from __future__ import annotations

import unittest

from optics_understanding_sft.direction_inverse_v1.evaluate_direction_llm import parse_directions


class EvaluateDirectionLlmTest(unittest.TestCase):
    def test_parse_requires_all_five_fields(self) -> None:
        prediction = {
            "parsed_json": {
                "answer": {
                    "directions": {
                        "centroid_x": "increase",
                        "centroid_y": "decrease",
                        "width_x": "no_change",
                        "width_y": "no_change",
                        "peak_intensity": "increase",
                    }
                }
            }
        }
        self.assertIsNotNone(parse_directions(prediction))
        del prediction["parsed_json"]["answer"]["directions"]["width_y"]
        self.assertIsNone(parse_directions(prediction))


if __name__ == "__main__":
    unittest.main()
