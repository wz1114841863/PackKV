import tempfile
import unittest
from pathlib import Path

from hardware.evaluation.synthesis.dc_baseline_report import (
    REQUIRED_REPORTS,
    extract_metrics,
    validate_run,
)


class DcBaselineReportTest(unittest.TestCase):
    def _valid_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        reports = root / "outputs"
        reports.mkdir()
        modules = root / "memory_modules.tcl"
        modules.write_text(
            "set BRISKKV_MEMORY_MODULES { queryMemory memory_8192x424 }\n",
            encoding="utf-8",
        )
        for name in REQUIRED_REPORTS:
            (reports / name).write_text("report\n", encoding="utf-8")
        audit = (
            "BRISK-KV architectural SRAM black-box audit\n"
            "queryMemory instances=1 source=syn_black_box\n"
            "memory_8192x424 instances=1 source=syn_black_box\n"
            "total_instances=2\n"
        )
        (reports / "memory_blackboxes_precompile.rpt").write_text(audit, encoding="utf-8")
        (reports / "memory_blackboxes_postcompile.rpt").write_text(audit, encoding="utf-8")
        log = root / "dc.log"
        log.write_text("BRISK-KV DC completed successfully\n", encoding="utf-8")
        return reports, log, modules

    def test_validates_blackboxes_and_extracts_common_dc_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            reports, log, modules = self._valid_fixture(Path(directory))
            (reports / "area_hier.rpt").write_text(
                "Total cell area: 1234.50\n", encoding="utf-8"
            )
            (reports / "qor.rpt").write_text(
                "Critical Path Slack: -0.125\nTotal Negative Slack: -1.50\n",
                encoding="utf-8",
            )
            (reports / "timing_setup.rpt").write_text(
                "  slack (VIOLATED) -0.125\n", encoding="utf-8"
            )
            (reports / "timing_hold.rpt").write_text(
                "  slack (MET) 0.031\n", encoding="utf-8"
            )
            (reports / "power.rpt").write_text(
                "Total Dynamic Power = 4.20 mW\nCell Leakage Power = 0.30 mW\n",
                encoding="utf-8",
            )

            result = validate_run(reports, log, modules)
            self.assertTrue(result["valid"])
            self.assertEqual(result["postcompile_memory_instances"]["queryMemory"], 1)
            self.assertEqual(result["metrics"]["cell_area"], 1234.5)
            self.assertEqual(result["metrics"]["setup_slack"], -0.125)
            self.assertEqual(result["metrics"]["hold_slack"], 0.031)

    def test_rejects_memory_disappearing_during_compile(self):
        with tempfile.TemporaryDirectory() as directory:
            reports, log, modules = self._valid_fixture(Path(directory))
            (reports / "memory_blackboxes_postcompile.rpt").write_text(
                "queryMemory instances=1\n", encoding="utf-8"
            )
            result = validate_run(reports, log, modules)
            self.assertFalse(result["valid"])
            self.assertTrue(
                any("postcompile audit missing" in error for error in result["errors"])
            )
            self.assertTrue(
                any("changed between" in error for error in result["errors"])
            )

    def test_rejects_log_without_explicit_success(self):
        with tempfile.TemporaryDirectory() as directory:
            reports, log, modules = self._valid_fixture(Path(directory))
            log.write_text("DC stopped before write\n", encoding="utf-8")
            result = validate_run(reports, log, modules)
            self.assertFalse(result["valid"])
            self.assertIn("DC success marker is absent from the log", result["errors"])


if __name__ == "__main__":
    unittest.main()
