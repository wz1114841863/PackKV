"""Validate and summarize one BRISK-KV Design Compiler baseline run.

The validator deliberately separates hard validity checks from best-effort
metric extraction.  A run with missing SRAM black boxes is invalid even when
DC happened to emit area or timing reports.  Conversely, an unfamiliar report
label is reported as a warning instead of being silently converted to zero.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping


REQUIRED_REPORTS = (
    "memory_blackboxes_precompile.rpt",
    "memory_blackboxes_postcompile.rpt",
    "check_design.rpt",
    "check_timing.rpt",
    "units.rpt",
    "clocks.rpt",
    "qor.rpt",
    "area_hier.rpt",
    "timing_setup.rpt",
    "timing_hold.rpt",
    "constraints.rpt",
    "power.rpt",
    "references.rpt",
)


@dataclass(frozen=True)
class AuditReport:
    counts: Mapping[str, int]
    declared_total: int | None


def load_memory_modules(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"set\s+BRISKKV_MEMORY_MODULES\s+\{(?P<body>.*?)\}",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"cannot find BRISKKV_MEMORY_MODULES in {path}")
    modules = tuple(match.group("body").split())
    if not modules:
        raise ValueError(f"BRISKKV_MEMORY_MODULES is empty in {path}")
    if len(set(modules)) != len(modules):
        raise ValueError(f"duplicate memory module in {path}")
    return modules


def parse_audit(path: Path) -> AuditReport:
    counts: Dict[str, int] = {}
    declared_total = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"\s*(\S+)\s+instances=(\d+)(?:\s|$)", line)
        if match:
            module, raw_count = match.groups()
            if module in counts:
                raise ValueError(f"duplicate audit entry {module} in {path}")
            counts[module] = int(raw_count)
        total_match = re.match(r"\s*total_instances=(\d+)\s*$", line)
        if total_match:
            declared_total = int(total_match.group(1))
    return AuditReport(counts=counts, declared_total=declared_total)


def _first_number(text: str, patterns: Iterable[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return float(match.group(1))
    return None


def extract_metrics(report_dir: Path) -> Dict[str, float | None]:
    def read(name: str) -> str:
        path = report_dir / name
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    area = read("area_hier.rpt") + "\n" + read("qor.rpt")
    qor = read("qor.rpt")
    setup = read("timing_setup.rpt")
    hold = read("timing_hold.rpt")
    power = read("power.rpt")
    return {
        "cell_area": _first_number(
            area,
            (
                r"^\s*Total\s+cell\s+area\s*:\s*([-+0-9.eE]+)",
                r"^\s*Cell\s+Area\s*:\s*([-+0-9.eE]+)",
                r"^\s*Design\s+Area\s*:\s*([-+0-9.eE]+)",
            ),
        ),
        "qor_wns": _first_number(
            qor,
            (
                r"^\s*(?:Critical\s+Path\s+Slack|WNS)\s*:\s*([-+0-9.eE]+)",
                r"^\s*Worst\s+slack\s*:\s*([-+0-9.eE]+)",
            ),
        ),
        "qor_tns": _first_number(
            qor,
            (
                r"^\s*(?:Total\s+Negative\s+Slack|TNS)\s*:\s*([-+0-9.eE]+)",
            ),
        ),
        "setup_slack": _first_number(
            setup,
            (r"^\s*slack\s+\((?:VIOLATED|MET)\)\s+([-+0-9.eE]+)",),
        ),
        "hold_slack": _first_number(
            hold,
            (r"^\s*slack\s+\((?:VIOLATED|MET)\)\s+([-+0-9.eE]+)",),
        ),
        # Kept in the unit printed by DC.  units.rpt is retained beside this
        # JSON and must be consulted before combining with CACTI values.
        "total_dynamic_power_report_units": _first_number(
            power,
            (r"^\s*Total\s+Dynamic\s+Power\s*=\s*([-+0-9.eE]+)",),
        ),
        "cell_leakage_power_report_units": _first_number(
            power,
            (r"^\s*Cell\s+Leakage\s+Power\s*=\s*([-+0-9.eE]+)",),
        ),
    }


def validate_run(
    report_dir: Path,
    dc_log: Path,
    memory_modules_file: Path,
) -> Dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    expected_modules: tuple[str, ...] = ()

    if not report_dir.is_dir():
        errors.append(f"report directory does not exist: {report_dir}")
    missing_reports = [name for name in REQUIRED_REPORTS if not (report_dir / name).is_file()]
    if missing_reports:
        errors.append("missing reports: " + ", ".join(missing_reports))

    if not memory_modules_file.is_file():
        errors.append(f"memory module list does not exist: {memory_modules_file}")
    else:
        try:
            expected_modules = load_memory_modules(memory_modules_file)
        except ValueError as exc:
            errors.append(str(exc))

    audits: Dict[str, AuditReport] = {}
    for phase, name in (
        ("precompile", "memory_blackboxes_precompile.rpt"),
        ("postcompile", "memory_blackboxes_postcompile.rpt"),
    ):
        path = report_dir / name
        if path.is_file():
            try:
                audits[phase] = parse_audit(path)
            except ValueError as exc:
                errors.append(str(exc))

    for phase, audit in audits.items():
        missing = sorted(set(expected_modules) - set(audit.counts))
        extra = sorted(set(audit.counts) - set(expected_modules))
        zero = sorted(module for module, count in audit.counts.items() if count <= 0)
        if missing:
            errors.append(f"{phase} audit missing modules: {', '.join(missing)}")
        if extra:
            errors.append(f"{phase} audit has unexpected modules: {', '.join(extra)}")
        if zero:
            errors.append(f"{phase} audit has zero-count modules: {', '.join(zero)}")
        actual_total = sum(audit.counts.values())
        if audit.declared_total is not None and audit.declared_total != actual_total:
            errors.append(
                f"{phase} total mismatch: declared={audit.declared_total}, actual={actual_total}"
            )

    if "precompile" in audits and "postcompile" in audits:
        if dict(audits["precompile"].counts) != dict(audits["postcompile"].counts):
            errors.append("memory instance counts changed between precompile and postcompile")

    if not dc_log.is_file():
        errors.append(f"DC log does not exist: {dc_log}")
    else:
        log_text = dc_log.read_text(encoding="utf-8", errors="replace")
        success_marker = re.search(
            r"(?m)^BRISK-KV DC completed successfully\s*$", log_text
        )
        failure_marker = re.search(r"(?m)^BRISK-KV DC FAILED:", log_text)
        if success_marker is None:
            errors.append("DC success marker is absent from the log")
        if failure_marker is not None:
            errors.append("DC failure marker is present in the log")

    metrics = extract_metrics(report_dir)
    missing_metrics = [name for name, value in metrics.items() if value is None]
    if missing_metrics:
        warnings.append("unparsed metrics: " + ", ".join(missing_metrics))

    counts = dict(audits.get("postcompile", AuditReport({}, None)).counts)
    return {
        "schema": "briskkv-dc-baseline-v0",
        "valid": not errors,
        "inputs": {
            "report_dir": str(report_dir.resolve()),
            "dc_log": str(dc_log.resolve()),
            "memory_modules_file": str(memory_modules_file.resolve()),
        },
        "expected_memory_modules": list(expected_modules),
        "postcompile_memory_instances": counts,
        "metrics": metrics,
        "errors": errors,
        "warnings": warnings,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--dc-log", type=Path, required=True)
    parser.add_argument("--memory-modules", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_run(args.report_dir, args.dc_log, args.memory_modules)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
