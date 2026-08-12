"""Safe, reproducible wrapper around the bundled CACTI executable."""

from __future__ import annotations

import csv
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional

try:  # Package import (recommended).
    from .cacti_config import CactiConfig
except ImportError:  # Compatibility with older PYTHONPATH=hardware/evaluation use.
    try:
        from mem.cacti_config import CactiConfig
    except ImportError:  # Direct execution from this directory.
        from cacti_config import CactiConfig


class CactiSimulation:
    """Run one physical SRAM macro point through CACTI.

    ``size`` and ``rw_bw`` are expressed in bits.  One instance of this class
    models one physical width slice with one CACTI bank; logical BRISK-KV
    memories are split and aggregated by :mod:`briskkv_memory_eval`.
    """

    CACTI_TOP_PATH = Path(__file__).resolve().parent
    CACTI_PROGRAM_PATH = CACTI_TOP_PATH / "cacti" / "cacti"

    REQUIRED_KEYS = {
        "technology",
        "mem_type",
        "size",
        "bank_count",
        "rw_bw",
        "r_port",
        "w_port",
        "rw_port",
    }

    def __init__(
        self,
        mem_config: Mapping[str, object],
        *,
        cacti_binary: Optional[os.PathLike[str] | str] = None,
        work_dir: Optional[os.PathLike[str] | str] = None,
        keep_files: bool = False,
    ) -> None:
        self._check_valid_config(mem_config)
        self.mem_config = dict(mem_config)
        self.cacti_config = CactiConfig()
        self.cacti_binary = Path(cacti_binary or self.CACTI_PROGRAM_PATH).resolve()
        self.work_dir = Path(work_dir).resolve() if work_dir else None
        self.keep_files = keep_files

    def run_cacti(self) -> Dict[str, object]:
        if not self.cacti_binary.is_file() or not os.access(self.cacti_binary, os.X_OK):
            raise FileNotFoundError(f"CACTI executable is missing or not executable: {self.cacti_binary}")

        if self.work_dir:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            return self._run_in_directory(self.work_dir)

        with tempfile.TemporaryDirectory(prefix="briskkv-cacti-") as temp_dir:
            return self._run_in_directory(Path(temp_dir))

    def _run_in_directory(self, directory: Path) -> Dict[str, object]:
        config_path = directory / "cache.cfg"
        output_path = Path(f"{config_path}.out")
        if output_path.exists():
            output_path.unlink()
        self._prepare_config_file(config_path)

        completed = subprocess.run(
            [str(self.cacti_binary), "-infile", str(config_path)],
            cwd=self.cacti_binary.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        (directory / "cacti.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (directory / "cacti.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0 or not output_path.is_file():
            details = (completed.stderr or completed.stdout or "no CACTI diagnostics").strip()
            raise RuntimeError(
                f"CACTI failed for {self.mem_config} (exit={completed.returncode}): {details}"
            )

        with output_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise RuntimeError(f"CACTI produced an empty result file: {output_path}")
        raw = {key.strip(): value.strip() for key, value in rows[0].items() if key is not None}

        def number(field: str) -> float:
            value = raw.get(field)
            if value in (None, "", "N/A"):
                raise RuntimeError(f"CACTI result is missing numeric field {field!r}: {raw}")
            return float(value)

        bank_count = int(self.mem_config["bank_count"])
        if bank_count != 1:
            raise ValueError(
                "CactiSimulation models one physical macro at a time; split logical banks before calling it"
            )

        return {
            "technology_nm": number("Tech node (nm)"),
            "size": int(number("Capacity (bytes)")) * 8,
            "bank_count": bank_count,
            "mem_type": self.mem_config["mem_type"],
            "r_port": int(self.mem_config["r_port"]),
            "w_port": int(self.mem_config["w_port"]),
            "rw_port": int(self.mem_config["rw_port"]),
            "rw_bw": int(number("Output width (bits)")),
            "access_time_ns": number("Access time (ns)"),
            "cycle_time_ns": number("Random cycle time (ns)"),
            "area_mm2": number("Area (mm2)"),
            "read_energy_pj": number("Dynamic read energy (nJ)") * 1000.0,
            "write_energy_pj": number("Dynamic write energy (nJ)") * 1000.0,
            "leakage_mw": number("Standby leakage per bank(mW)"),
            "raw": raw,
        }

    def _check_valid_config(self, mem_config: Mapping[str, object]) -> None:
        missing = sorted(self.REQUIRED_KEYS - set(mem_config))
        if missing:
            raise ValueError(f"missing keys {missing} from mem_config")
        for key in ("size", "rw_bw", "r_port", "w_port", "rw_port", "bank_count"):
            if int(mem_config[key]) < 0:
                raise ValueError(f"{key} must be non-negative")
        if int(mem_config["size"]) <= 0 or int(mem_config["rw_bw"]) <= 0:
            raise ValueError("size and rw_bw must be positive")
        if int(mem_config["size"]) % 8:
            raise ValueError("physical CACTI size must be byte aligned")
        if int(mem_config["bank_count"]) != 1:
            raise ValueError("one CactiSimulation call represents exactly one physical macro")
        technology = float(mem_config["technology"])
        if not 0.022 <= technology <= 0.180:
            raise ValueError(
                "bundled CACTI data supports 22-180 nm; use a foundry memory compiler below 22 nm"
            )

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        return 1 if value <= 1 else 1 << (value - 1).bit_length()

    def _prepare_config_file(self, config_path: Path) -> None:
        width_bits = int(self.mem_config["rw_bw"])
        # CACTI requires a line at least as wide as the output.  Eight bytes is
        # retained as a conservative minimum for very narrow control SRAMs.
        line_size_bytes = max(8, self._next_power_of_two(math.ceil(width_bits / 8)))
        requested_bytes = int(self.mem_config["size"]) // 8
        modeled_bytes = max(
            64,
            math.ceil(requested_bytes / line_size_bytes) * line_size_bytes,
        )

        values = {
            "technology": float(self.mem_config["technology"]),
            "mem_type": f'"{self.mem_config["mem_type"]}"',
            "bank_count": 1,
            "cache_size": modeled_bytes,
            "line_size": line_size_bytes,
            "IO_bus_width": width_bits,
            "ex_rd_port": int(self.mem_config["r_port"]),
            "ex_wr_port": int(self.mem_config["w_port"]),
            "rd_wr_port": int(self.mem_config["rw_port"]),
        }

        user_config = []
        for name, option in self.cacti_config.config_option.items():
            value = values.get(name, option["default"])
            user_config.append(option["string"] + str(value) + "\n")

        config_path.write_text(
            "".join(self.cacti_config.baseline_config) + "".join(user_config),
            encoding="utf-8",
        )
