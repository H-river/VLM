#!/usr/bin/env python3
"""Audit documented reproduction commands in clean, non-executing processes.

The audit extracts shell commands from fenced Markdown blocks, resolves each
``python -m`` target, invokes only its help path, and checks that every
documented long option is advertised.  It never executes an experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


VERSION = "active_diagnosis_v13_reproduction_command_audit_v1"


def extract_commands(markdown: str) -> list[str]:
    commands: list[str] = []
    in_shell_fence = False
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            if in_shell_fence:
                in_shell_fence = False
            else:
                language = line[3:].strip().lower()
                in_shell_fence = language in {"bash", "sh", "shell"}
            continue
        if in_shell_fence and line and not line.startswith("#"):
            commands.append(line)
    return commands


def _strip_assignments(tokens: list[str]) -> list[str]:
    index = 0
    while index < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[index]):
        index += 1
    return tokens[index:]


def parse_module_command(command: str) -> dict[str, Any]:
    try:
        tokens = _strip_assignments(shlex.split(command))
    except ValueError as error:
        return {"command": command, "status": "parse_error", "error": str(error)}
    if len(tokens) < 3 or tokens[1] != "-m":
        return {"command": command, "status": "not_python_module"}
    module = tokens[2]
    remainder = tokens[3:]
    subcommand = remainder[0] if remainder and not remainder[0].startswith("-") else None
    options = sorted({token.split("=", 1)[0] for token in remainder if token.startswith("--")})
    return {
        "command": command,
        "executable": tokens[0],
        "module": module,
        "subcommand": subcommand,
        "options": options,
        "status": "parsed",
    }


def _clean_environment(cwd: Path) -> dict[str, str]:
    allowed = ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "LANG", "LC_ALL")
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment["PYTHONPATH"] = str(cwd)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def audit_command(parsed: dict[str, Any], cwd: Path, timeout: float) -> dict[str, Any]:
    if parsed["status"] != "parsed":
        return parsed
    help_commands = [[parsed["executable"], "-m", parsed["module"], "--help"]]
    if parsed["subcommand"]:
        help_commands.append(
            [parsed["executable"], "-m", parsed["module"], parsed["subcommand"], "--help"]
        )
    help_texts: list[str] = []
    invocations: list[dict[str, Any]] = []
    for help_command in help_commands:
        try:
            completed = subprocess.run(
                help_command,
                cwd=cwd,
                env=_clean_environment(cwd),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            output = completed.stdout or ""
            help_texts.append(output)
            invocations.append(
                {
                    "argv": help_command,
                    "returncode": completed.returncode,
                    "output_tail": output[-1000:],
                }
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            invocations.append({"argv": help_command, "error": str(error)})
    combined_help = "\n".join(help_texts)
    missing_options = [option for option in parsed["options"] if option not in combined_help]
    help_ok = bool(invocations) and all(item.get("returncode") == 0 for item in invocations)
    return {
        **parsed,
        "help_invocations": invocations,
        "missing_options": missing_options,
        "passes": help_ok and not missing_options,
        "status": "pass" if help_ok and not missing_options else "fail",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands-markdown", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    markdown_path = args.commands_markdown.resolve()
    cwd = Path.cwd().resolve()
    commands = extract_commands(markdown_path.read_text(encoding="utf-8"))
    results = []
    for index, command in enumerate(commands, start=1):
        result = audit_command(
            parse_module_command(command), cwd=cwd, timeout=args.timeout
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "event": "command_audit_complete",
                    "index": index,
                    "commands": len(commands),
                    "module": result.get("module"),
                    "status": result["status"],
                    "missing_options": result.get("missing_options", []),
                }
            ),
            flush=True,
        )
    module_results = [result for result in results if result["status"] != "not_python_module"]
    report = {
        "version": VERSION,
        "role": "clean_process_help_and_option_audit_no_experiment_execution",
        "commands_markdown": str(markdown_path),
        "cwd": str(cwd),
        "protected_set_used": False,
        "command_count": len(commands),
        "python_module_command_count": len(module_results),
        "pass_count": sum(result.get("passes", False) for result in module_results),
        "failure_count": sum(not result.get("passes", False) for result in module_results),
        "passes": bool(module_results) and all(result.get("passes", False) for result in module_results),
        "results": results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    if not report["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
