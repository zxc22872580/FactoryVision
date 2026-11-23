#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


STAGE_CONFIG = {
    1: {
        "script": "stage1_inference.py",
        "model": "stage1_model.pth",
        "report": "stage1_report.pdf",
        "output_flag": "--output_csv",
        "output_name": "predictions.csv",
    },
    2: {
        "script": "stage2_inference.py",
        "model": "stage2_model.pth",
        "report": "stage2_report.pdf",
        "output_flag": "--output_csv",
        "output_name": "predictions.csv",
    },
    3: {
        "script": "stage3_detect.py",
        "model": "stage3_model.pt",
        "report": "stage3_report.pdf",
        "output_flag": "--output_json",
        "output_name": "detections.json",
    },
}


class ValidationResult:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        self.passed.append(message)

    def fail(self, message: str) -> None:
        self.failed.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def is_pass(self) -> bool:
        return not self.failed


def _find_add_argument_strings(py_source: str) -> set[str]:
    flags: set[str] = set()
    try:
        tree = ast.parse(py_source)
    except SyntaxError:
        return flags

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                flags.add(arg.value)
            elif isinstance(arg, ast.Tuple):
                for item in arg.elts:
                    if isinstance(item, ast.Constant) and isinstance(item.value, str):
                        flags.add(item.value)
    return flags


def _contains_regex(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None


def _validate_nonempty_file(path: Path, label: str, result: ValidationResult) -> None:
    if not path.exists():
        result.fail(f"Missing required file: {label}")
        return
    if not path.is_file():
        result.fail(f"Required path is not a file: {label}")
        return
    if path.stat().st_size <= 0:
        result.fail(f"Required file is empty: {label}")
        return
    result.ok(f"Found required file: {label}")


def validate_submission(stage: int, submission_dir: Path) -> ValidationResult:
    result = ValidationResult()
    cfg = STAGE_CONFIG[stage]

    if not submission_dir.exists():
        result.fail(f"Submission path does not exist: {submission_dir}")
        return result
    if not submission_dir.is_dir():
        result.fail(f"Submission path is not a directory: {submission_dir}")
        return result
    result.ok(f"Submission directory exists: {submission_dir}")

    script_path = submission_dir / cfg["script"]
    if not script_path.exists():
        result.fail(f"Missing required script: {cfg['script']}")
    else:
        result.ok(f"Found required script: {cfg['script']}")

    report_path = submission_dir / cfg["report"]
    _validate_nonempty_file(report_path, cfg["report"], result)

    model_path = submission_dir / cfg["model"]
    _validate_nonempty_file(model_path, cfg["model"], result)

    config_path = submission_dir / "config.json"
    _validate_nonempty_file(config_path, "config.json", result)
    if config_path.exists() and config_path.is_file() and config_path.stat().st_size > 0:
        try:
            import json

            json.loads(config_path.read_text(encoding="utf-8", errors="ignore"))
            result.ok("config.json is valid JSON.")
        except Exception as exc:  # pragma: no cover - defensive branch
            result.fail(f"config.json is not valid JSON: {exc}")

    req_path = submission_dir / "requirements.txt"
    _validate_nonempty_file(req_path, "requirements.txt", result)

    readme_path = submission_dir / "README.md"
    _validate_nonempty_file(readme_path, "README.md", result)

    src_dir = submission_dir / "src"
    if not src_dir.exists() or not src_dir.is_dir():
        result.fail("Missing required directory: src")
    else:
        result.ok("Found required directory: src")

    src_files = ["model.py", "train.py", "evaluate.py"]
    for name in src_files:
        src_file = src_dir / name
        _validate_nonempty_file(src_file, f"src/{name}", result)
        if src_file.exists() and src_file.is_file():
            try:
                ast.parse(src_file.read_text(encoding="utf-8", errors="ignore"))
                result.ok(f"src/{name} has valid Python syntax.")
            except SyntaxError as exc:
                result.fail(
                    f"src/{name} has syntax error at line {exc.lineno}, col {exc.offset}: {exc.msg}"
                )

    if script_path.exists():
        source = script_path.read_text(encoding="utf-8", errors="ignore")
        try:
            ast.parse(source)
            result.ok(f"{cfg['script']} has valid Python syntax.")
        except SyntaxError as exc:
            result.fail(
                f"{cfg['script']} has syntax error at line {exc.lineno}, col {exc.offset}: {exc.msg}"
            )

        flags = _find_add_argument_strings(source)
        if "--input_dir" in flags:
            result.ok("Found CLI argument '--input_dir'.")
        else:
            result.fail("Missing CLI argument '--input_dir' in script.")

        if cfg["output_flag"] in flags:
            result.ok(f"Found CLI argument '{cfg['output_flag']}'.")
        else:
            result.fail(f"Missing CLI argument '{cfg['output_flag']}' in script.")

        if _contains_regex(source, r"__name__\s*==\s*['\"]__main__['\"]"):
            result.ok("Found '__main__' entrypoint.")
        else:
            result.warn("No explicit '__main__' entrypoint found. Ensure script can be run directly.")

        output_name = cfg["output_name"]
        if output_name not in source:
            result.warn(
                f"Script source does not contain '{output_name}'. Ensure output path writes expected format."
            )

        if cfg["model"] not in source:
            result.warn(
                f"Script source does not explicitly reference '{cfg['model']}'. Ensure benchmark uses this checkpoint."
            )

    return result


def print_result(stage: int, submission_dir: Path, result: ValidationResult) -> None:
    print(f"Stage: {stage}")
    print(f"Submission: {submission_dir}")
    print()

    for message in result.passed:
        print(f"[PASS] {message}")
    for message in result.warnings:
        print(f"[WARN] {message}")
    for message in result.failed:
        print(f"[FAIL] {message}")

    print()
    if result.is_pass:
        print("RESULT: PASS")
        print("This submission passed structural validation and is ready for benchmark run.")
    else:
        print("RESULT: FAIL")
        print("Please fix failed items before uploading or running benchmark.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Stage 1/2/3 submission structure before benchmark."
    )
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Stage number: 1, 2, or 3",
    )
    parser.add_argument(
        "--submission",
        required=True,
        help="Path to submission folder, e.g. TeamXX_Stage1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    submission_dir = Path(args.submission).resolve()
    result = validate_submission(args.stage, submission_dir)
    print_result(args.stage, submission_dir, result)
    return 0 if result.is_pass else 1


if __name__ == "__main__":
    sys.exit(main())
