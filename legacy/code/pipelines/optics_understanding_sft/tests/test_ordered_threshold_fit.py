from optics_understanding_sft.visual_state_tool_v10_1 import fit_ordered_thresholds


def _brute(rows, labels):
    values = sorted({value for value, _ in rows})
    candidates = [(a + b) / 2 for a, b in zip(values, values[1:])]
    best = (-1.0, 0.0, 0.0)
    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            correct = sum(
                target == (labels[0] if value < first else labels[2] if value > second else labels[1])
                for value, target in rows
            )
            score = correct / len(rows)
            if score > best[0]:
                best = (score, first, second)
    return best


def test_cumulative_threshold_search_matches_brute_force():
    labels = ["low", "middle", "high"]
    rows = [
        (0.0, "low"),
        (1.0, "low"),
        (1.0, "middle"),
        (2.0, "middle"),
        (3.0, "high"),
        (4.0, "middle"),
    ]
    assert fit_ordered_thresholds(rows, labels) == _brute(rows, labels)


def test_macro_f1_threshold_objective_reports_balanced_score():
    labels = ["decrease", "no_change", "increase"]
    rows = (
        [(0.0, "decrease")]
        + [(1.0, "no_change")] * 20
        + [(2.0, "increase")]
        + [(3.0, "no_change")] * 20
    )
    accuracy = fit_ordered_thresholds(rows, labels, objective="accuracy")
    macro = fit_ordered_thresholds(rows, labels, objective="macro_f1")
    assert macro[0] > 0.0
    assert macro[0] != accuracy[0]
    assert macro[1] < macro[2]
