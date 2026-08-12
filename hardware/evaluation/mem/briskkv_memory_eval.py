"""Evaluate BRISK-KV architectural SRAMs from the generated RTL inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    from .cacti_simulation import CactiSimulation
except ImportError:
    from cacti_simulation import CactiSimulation


@dataclass(frozen=True)
class MemorySpec:
    module: str
    purpose: str
    depth: int
    width_bits: int
    instances: int = 1
    read_ports: int = 1
    write_ports: int = 1
    readwrite_ports: int = 0
    access_mode: str = "parallel_width"

    @property
    def logical_bits_per_instance(self) -> int:
        return self.depth * self.width_bits

    @property
    def total_bits(self) -> int:
        return self.logical_bits_per_instance * self.instances


def _positive_int(row: Dict[str, str], key: str, default: int | None = None) -> int:
    raw = row.get(key, "")
    if raw == "" and default is not None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer {key}={raw!r} in row {row}") from exc
    if value <= 0:
        raise ValueError(f"{key} must be positive in row {row}")
    return value


def _port_count(row: Dict[str, str], key: str, default: int) -> int:
    raw = row.get(key, "")
    if raw == "":
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{key} must be non-negative in row {row}")
    return value


def load_inventory(path: Path) -> List[MemorySpec]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"memory inventory is empty: {path}")

    required = {"module", "purpose", "depth", "width_bits"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"memory inventory is missing columns {sorted(missing)}: {path}")

    specs: List[MemorySpec] = []
    seen = set()
    for row in rows:
        module = row["module"].strip()
        purpose = row["purpose"].strip()
        if not module or not purpose:
            raise ValueError(f"module and purpose must be non-empty: {row}")
        if module in seen:
            raise ValueError(f"duplicate memory module in inventory: {module}")
        seen.add(module)
        spec = MemorySpec(
            module=module,
            purpose=purpose,
            depth=_positive_int(row, "depth"),
            width_bits=_positive_int(row, "width_bits"),
            instances=_positive_int(row, "instances", 1),
            read_ports=_port_count(row, "read_ports", 1),
            write_ports=_port_count(row, "write_ports", 1),
            readwrite_ports=_port_count(row, "readwrite_ports", 0),
            access_mode=(row.get("access_mode") or "parallel_width").strip(),
        )
        if spec.access_mode != "parallel_width":
            raise ValueError(
                f"unsupported access_mode={spec.access_mode!r} for {module}; "
                "BRISK-KV currently expects every width slice to fire in parallel"
            )
        declared_bits = row.get("total_bits", "").strip()
        if declared_bits and int(declared_bits) != spec.total_bits:
            raise ValueError(
                f"total_bits mismatch for {module}: CSV={declared_bits}, computed={spec.total_bits}"
            )
        specs.append(spec)
    return specs


def split_width(width_bits: int, maximum_bank_width: int) -> List[int]:
    if width_bits <= 0 or maximum_bank_width <= 0:
        raise ValueError("width_bits and maximum_bank_width must be positive")
    full, remainder = divmod(width_bits, maximum_bank_width)
    widths = [maximum_bank_width] * full
    if remainder:
        widths.append(remainder)
    return widths


def evaluate_memory(
    spec: MemorySpec,
    *,
    technology_nm: float,
    maximum_bank_width: int,
    clock_period_ns: float,
    work_root: Path,
    cacti_binary: Path | None,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    bank_results: List[Dict[str, object]] = []
    for index, bank_width in enumerate(split_width(spec.width_bits, maximum_bank_width)):
        logical_bank_bits = spec.depth * bank_width
        # The physical macro must be byte addressable.  Any byte/line rounding
        # is reported as modeled capacity rather than hidden.
        byte_aligned_bits = math.ceil(logical_bank_bits / 8) * 8
        result = CactiSimulation(
            {
                "technology": technology_nm / 1000.0,
                "mem_type": "ram",
                "size": byte_aligned_bits,
                "bank_count": 1,
                "rw_bw": bank_width,
                "r_port": spec.read_ports,
                "w_port": spec.write_ports,
                "rw_port": spec.readwrite_ports,
            },
            cacti_binary=cacti_binary,
            work_dir=work_root / spec.module / f"slice_{index:02d}_w{bank_width}",
            keep_files=True,
        ).run_cacti()
        bank_results.append(
            {
                "module": spec.module,
                "purpose": spec.purpose,
                "slice_index": index,
                "slice_width_bits": bank_width,
                "logical_bits": logical_bank_bits,
                "modeled_bits": int(result["size"]),
                "area_mm2": float(result["area_mm2"]),
                "leakage_mw": float(result["leakage_mw"]),
                "read_energy_pj": float(result["read_energy_pj"]),
                "write_energy_pj": float(result["write_energy_pj"]),
                "access_time_ns": float(result["access_time_ns"]),
                "cycle_time_ns": float(result["cycle_time_ns"]),
                "output_width_bits": int(result["rw_bw"]),
            }
        )

    modeled_bits = sum(int(row["modeled_bits"]) for row in bank_results)
    area = sum(float(row["area_mm2"]) for row in bank_results)
    leakage = sum(float(row["leakage_mw"]) for row in bank_results)
    read_energy = sum(float(row["read_energy_pj"]) for row in bank_results)
    write_energy = sum(float(row["write_energy_pj"]) for row in bank_results)
    access_time = max(float(row["access_time_ns"]) for row in bank_results)
    cycle_time = max(float(row["cycle_time_ns"]) for row in bank_results)

    summary = {
        "module": spec.module,
        "purpose": spec.purpose,
        "depth": spec.depth,
        "logical_width_bits": spec.width_bits,
        "instances": spec.instances,
        "width_slices": len(bank_results),
        "slice_widths": "+".join(str(row["slice_width_bits"]) for row in bank_results),
        "read_ports": spec.read_ports,
        "write_ports": spec.write_ports,
        "readwrite_ports": spec.readwrite_ports,
        "logical_bits_total": spec.total_bits,
        "modeled_bits_total": modeled_bits * spec.instances,
        "capacity_overhead_pct":
            (modeled_bits * spec.instances / spec.total_bits - 1.0) * 100.0,
        "area_mm2_total": area * spec.instances,
        "leakage_mw_total": leakage * spec.instances,
        "read_energy_pj_per_instance_access": read_energy,
        "write_energy_pj_per_instance_access": write_energy,
        "read_energy_pj_all_instances": read_energy * spec.instances,
        "write_energy_pj_all_instances": write_energy * spec.instances,
        "access_time_ns": access_time,
        "cycle_time_ns": cycle_time,
        "latency_cycles": max(1, math.ceil(access_time / clock_period_ns)),
        "initiation_interval_cycles": max(1, math.ceil(cycle_time / clock_period_ns)),
    }
    return summary, bank_results


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_inventory(args: argparse.Namespace) -> Dict[str, object]:
    inventory_path = args.memories_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = load_inventory(inventory_path)
    cacti_binary = (args.cacti_binary or CactiSimulation.CACTI_PROGRAM_PATH).resolve()
    cacti_sha256 = hashlib.sha256(cacti_binary.read_bytes()).hexdigest()

    summaries: List[Dict[str, object]] = []
    bank_rows: List[Dict[str, object]] = []
    for spec in specs:
        summary, banks = evaluate_memory(
            spec,
            technology_nm=args.technology_nm,
            maximum_bank_width=args.maximum_bank_width,
            clock_period_ns=args.clock_period_ns,
            work_root=output_dir / "cacti_runs",
            cacti_binary=cacti_binary,
        )
        summaries.append(summary)
        bank_rows.extend(banks)

    totals = {
        "logical_bits": sum(int(row["logical_bits_total"]) for row in summaries),
        "modeled_bits": sum(int(row["modeled_bits_total"]) for row in summaries),
        "area_mm2": sum(float(row["area_mm2_total"]) for row in summaries),
        "leakage_mw": sum(float(row["leakage_mw_total"]) for row in summaries),
        # These are useful upper-bound normalization numbers, not workload
        # energy.  Workload energy requires measured read/write access counts.
        "one_access_each_read_energy_pj": sum(
            float(row["read_energy_pj_all_instances"]) for row in summaries
        ),
        "one_access_each_write_energy_pj": sum(
            float(row["write_energy_pj_all_instances"]) for row in summaries
        ),
    }
    totals["capacity_overhead_pct"] = (
        totals["modeled_bits"] / totals["logical_bits"] - 1.0
    ) * 100.0

    report = {
        "schema": "briskkv-cacti-v0",
        "input": str(inventory_path),
        "assumptions": {
            "cacti_binary": str(cacti_binary),
            "cacti_binary_sha256": cacti_sha256,
            "technology_nm": args.technology_nm,
            "clock_period_ns": args.clock_period_ns,
            "maximum_bank_width_bits": args.maximum_bank_width,
            "memory_type": "ram",
            "cell_and_peripheral_type": "itrs-hp",
            "ecc": False,
            "temperature_k": 300,
            "banking": "parallel width slices; all slices active per logical access",
            "ports": "one exclusive read plus one exclusive write unless overridden by CSV",
            "dynamic_energy": "CACTI per-access energy; multiply by measured accesses separately",
        },
        "memories": summaries,
        "totals": totals,
    }
    _write_csv(output_dir / "memory_summary.csv", summaries)
    _write_csv(output_dir / "bank_detail.csv", bank_rows)
    (output_dir / "memory_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CACTI for every SRAM in a generated BRISK-KV memories.csv inventory."
    )
    parser.add_argument("--memories-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--technology-nm", type=float, default=22.0)
    parser.add_argument("--maximum-bank-width", type=int, default=128)
    parser.add_argument("--clock-period-ns", type=float, default=2.0)
    parser.add_argument("--cacti-binary", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 22.0 <= args.technology_nm <= 180.0:
        parser.error("--technology-nm must be within the bundled CACTI range [22, 180]")
    if args.maximum_bank_width <= 0:
        parser.error("--maximum-bank-width must be positive")
    if args.clock_period_ns <= 0:
        parser.error("--clock-period-ns must be positive")
    report = evaluate_inventory(args)
    totals = report["totals"]
    print(f"Evaluated {len(report['memories'])} BRISK-KV memories")
    print(f"Area: {totals['area_mm2']:.6f} mm^2")
    print(f"Leakage: {totals['leakage_mw']:.6f} mW")
    print(f"Logical capacity: {totals['logical_bits']} bits")
    print(f"CACTI modeled capacity: {totals['modeled_bits']} bits")
    print(f"Reports: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
