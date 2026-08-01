from optics_understanding_sft.build_balanced_dev_panel import exclude_physical_groups


def test_exclusion_removes_entire_physical_group():
    source = [
        {"example_id": "a", "group_id": "shared"},
        {"example_id": "b", "group_id": "shared"},
        {"example_id": "c", "group_id": "kept"},
    ]
    kept, excluded = exclude_physical_groups(
        source, [{"example_id": "prior", "group_id": "shared"}]
    )
    assert excluded == {"shared"}
    assert [row["example_id"] for row in kept] == ["c"]
