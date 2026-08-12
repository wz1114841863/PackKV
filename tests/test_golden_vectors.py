import json
import tempfile
import unittest
from pathlib import Path

from utils.golden_vectors import (
    build_case_artifacts,
    directed_cases,
    export_cases,
)


class GoldenVectorExportTest(unittest.TestCase):
    def test_cases_self_validate_and_cover_width_zero(self):
        cases = directed_cases()
        self.assertEqual(len(cases), 3)
        artifacts = {
            case.name: build_case_artifacts(case) for case in cases
        }

        width0_descriptor, width0_files = artifacts["directed_width0"]
        self.assertEqual(width0_files["k_payload.bin"], b"")
        self.assertEqual(width0_files["v_payload.bin"], b"")
        self.assertEqual(
            width0_descriptor["bitpack"]["k"]["code_value_bits"], 6
        )
        self.assertEqual(
            width0_descriptor["bitpack"]["v"]["code_value_bits"], 4
        )
        self.assertEqual(
            width0_descriptor["bitpack"]["k"]["encode_length_field_bits"], 3
        )
        self.assertEqual(
            width0_descriptor["bitpack"]["v"]["encode_length_field_bits"], 3
        )
        self.assertEqual(len(width0_files["k_pack_mins.bin"]), 12)
        self.assertEqual(len(width0_files["v_pack_mins.bin"]), 8)
        self.assertEqual(len(width0_files["k_encode_lengths.bin"]), 6)
        self.assertEqual(len(width0_files["v_encode_lengths.bin"]), 6)
        self.assertTrue(
            width0_descriptor["validation"][
                "joint_q_metadata_dequant_roundtrip"
            ]
        )
        self.assertTrue(
            width0_descriptor["validation"]["qk_fixed_point_reference"]
        )
        self.assertEqual(len(width0_files["qk_query_q6_i32.bin"]), 4 * 4)
        self.assertEqual(
            len(width0_files["expected_qk_logits_q12_i64.bin"]),
            64 * 8,
        )

    def test_export_writes_manifest_descriptors_and_refuses_accidental_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "vectors"
            manifest = export_cases(output, directed_cases())
            self.assertEqual(manifest["case_count"], 3)
            loaded_manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(loaded_manifest, manifest)
            for entry in manifest["cases"]:
                descriptor_path = (
                    output / entry["directory"] / "descriptor.json"
                )
                descriptor = json.loads(
                    descriptor_path.read_text(encoding="utf-8")
                )
                self.assertEqual(descriptor["format"], "briskkv-format-v0")
                self.assertTrue(descriptor["files"])

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                export_cases(output, directed_cases())
            export_cases(output, directed_cases(), overwrite=True)


if __name__ == "__main__":
    unittest.main()
