"""Compatibility object for consumers that need one CACTI memory instance."""

from __future__ import annotations

from typing import Dict, Optional

try:
    from .cacti_simulation import CactiSimulation
except ImportError:
    try:
        from mem.cacti_simulation import CactiSimulation
    except ImportError:
        from cacti_simulation import CactiSimulation


class MemoryInstance:
    def __init__(
        self,
        mem_config: Dict,
        r_cost: float = 0,
        w_cost: float = 0,
        latency: float = 1,
        min_r_granularity: Optional[int] = None,
        min_w_granularity: Optional[int] = None,
        get_cost_from_cacti: bool = True,
    ) -> None:
        self.name = str(mem_config.get("name", "unnamed_memory"))
        if get_cost_from_cacti:
            result = CactiSimulation(mem_config).run_cacti()
            self.r_cost = float(result["read_energy_pj"])
            self.w_cost = float(result["write_energy_pj"])
            self.area = float(result["area_mm2"])
            self.latency = round(float(result["access_time_ns"]), 6)
            effective = dict(mem_config)
            effective["size"] = result["size"]
        else:
            self.r_cost = r_cost
            self.w_cost = w_cost
            self.area = 0.0
            self.latency = latency
            effective = mem_config

        self.size = int(effective["size"])
        self.bank = int(effective["bank_count"])
        self.rw_bw = int(effective["rw_bw"])
        self.r_port = int(effective["r_port"])
        self.w_port = int(effective["w_port"])
        self.rw_port = int(effective["rw_port"])

        self.r_bw_min = min_r_granularity or self.rw_bw
        self.w_bw_min = min_w_granularity or self.rw_bw
        self.r_cost_min = self.r_cost * self.r_bw_min / self.rw_bw
        self.w_cost_min = self.w_cost * self.w_bw_min / self.rw_bw

    def get_cacti_cost(self) -> Dict[str, float]:
        return {
            "r_cost": self.r_cost,
            "w_cost": self.w_cost,
            "area": self.area,
            "latency": self.latency,
        }

    def __jsonrepr__(self) -> Dict:
        return self.__dict__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MemoryInstance) and self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return id(self)

    def __str__(self) -> str:
        return f"MemoryInstance({self.name})"

    def __repr__(self) -> str:
        return str(self)
