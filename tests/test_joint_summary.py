import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "summarize_accuracy_cr.py"
SPEC = importlib.util.spec_from_file_location("summarize_accuracy_cr", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class JointSummaryTest(unittest.TestCase):
    def test_fixed_configuration_filter_and_join(self):
        model = "synthetic/model"
        common_accuracy = {
            "Model": model,
            "Task": "gsm8k",
            "Limit": "",
            "Block_Size": "64",
            "Buffer_Size": "192",
            "Pack_Size": "16",
            "High_Precision_Zero_Point": "False",
            "Created_At": "2026-08-08T00:00:00",
            "__source": "accuracy.csv",
        }
        accuracy_rows = [
            {
                **common_accuracy,
                "Enable_Quant": "False",
                "Strict_Match": "0.50",
                "Flexible_Extract": "0.51",
            }
        ]
        for repack, strict in (("NONE", "0.49"), ("BUCKET", "0.495")):
            accuracy_rows.append(
                {
                    **common_accuracy,
                    "Enable_Quant": "True",
                    "Quant_Method": "PackKV",
                    "Scale_Method": "po2_nearest",
                    "K_Scale": "0.03",
                    "V_Scale": "0.10",
                    "Repack_Method": repack,
                    "Bucket_Count": "4",
                    "Bucket_Score_Method": "k_sum",
                    "Strict_Match": strict,
                    "Flexible_Extract": "0.50",
                }
            )

        cr_rows = []
        for repack, overall, total in (
            ("NONE", "4.0", 1000),
            ("BUCKET", "4.2", 950),
        ):
            cr_rows.append(
                {
                    "Model": model,
                    "Ctx_Len": "4096",
                    "Quant_Method": "PackKV",
                    "Scale_Method": "po2_nearest",
                    "Repack_Method": repack,
                    "K_Scale": "0.03",
                    "V_Scale": "0.10",
                    "Block_Size": "64",
                    "Buffer_Size": "192",
                    "Pack_Size": "16",
                    "Bucket_Count": "4",
                    "Bucket_Score_Method": "k_sum",
                    "High_Precision_Zero_Point": "False",
                    "Generated_At": "2026-08-08 00:00:00",
                    "K_Global_CR": "4.1",
                    "V_Global_CR": "3.9",
                    "Overall_Global_CR": overall,
                    "K_Compressed_Bytes": str(total // 2),
                    "V_Compressed_Bytes": str(total // 2),
                    "__source": "cr.csv",
                }
            )

        rows = MODULE.build_rows(accuracy_rows, cr_rows, [model], 4096)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["Repack_Method"], "NONE")
        self.assertEqual(rows[0]["Strict_Delta_vs_FP_pp"], "-1.000000")
        self.assertEqual(rows[0]["FP_Baseline_Status"], "recorded")
        self.assertEqual(rows[1]["Repack_Method"], "BUCKET")
        self.assertEqual(rows[1]["Bucket_CR_Gain_vs_NONE_Pct"], "5.000000")
        self.assertEqual(rows[1]["Bucket_Bytes_Change_vs_NONE_Pct"], "-5.000000")

    def test_missing_fp_does_not_block_joint_summary(self):
        model = "synthetic/model"
        accuracy_rows = []
        for repack in ("NONE", "BUCKET"):
            accuracy_rows.append(
                {
                    "Model": model,
                    "Task": "gsm8k",
                    "Limit": "",
                    "Enable_Quant": "True",
                    "Quant_Method": "PackKV",
                    "Scale_Method": "po2_nearest",
                    "K_Scale": "0.03",
                    "V_Scale": "0.10",
                    "Repack_Method": repack,
                    "Block_Size": "64",
                    "Buffer_Size": "192",
                    "Pack_Size": "16",
                    "Bucket_Count": "4",
                    "Bucket_Score_Method": "k_sum",
                    "High_Precision_Zero_Point": "False",
                    "Strict_Match": "0.49",
                    "Flexible_Extract": "0.50",
                    "Created_At": "2026-08-08T00:00:00",
                    "__source": "accuracy.csv",
                }
            )
        cr_rows = []
        for repack in ("NONE", "BUCKET"):
            cr_rows.append(
                {
                    "Model": model,
                    "Ctx_Len": "4096",
                    "Quant_Method": "PackKV",
                    "Scale_Method": "po2_nearest",
                    "K_Scale": "0.03",
                    "V_Scale": "0.10",
                    "Repack_Method": repack,
                    "Block_Size": "64",
                    "Buffer_Size": "192",
                    "Pack_Size": "16",
                    "Bucket_Count": "4",
                    "Bucket_Score_Method": "k_sum",
                    "High_Precision_Zero_Point": "False",
                    "Generated_At": "2026-08-08 00:00:00",
                    "K_Global_CR": "4.1",
                    "V_Global_CR": "3.9",
                    "Overall_Global_CR": "4.0",
                    "K_Compressed_Bytes": "500",
                    "V_Compressed_Bytes": "500",
                    "__source": "cr.csv",
                }
            )

        rows = MODULE.build_rows(accuracy_rows, cr_rows, [model], 4096)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["FP_Baseline_Status"], "missing")
        self.assertEqual(rows[0]["FP_Strict_Match_Pct"], "")
        self.assertEqual(rows[0]["Strict_Delta_vs_FP_pp"], "")


if __name__ == "__main__":
    unittest.main()
