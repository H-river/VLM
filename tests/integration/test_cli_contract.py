from __future__ import annotations

from optics_vla.cli import config_check, qwen_contract
from optics_vla.common.config import load_controller_config


def test_config_and_qwen_schema_smokes() -> None:
    assert config_check(load_controller_config())["status"] == "PASS"
    assert qwen_contract()["status"] == "PASS"

