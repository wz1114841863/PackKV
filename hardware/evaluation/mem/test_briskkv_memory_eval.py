import tempfile
import unittest
from pathlib import Path

from hardware.evaluation.mem.briskkv_memory_eval import load_inventory, split_width


class BriskKvMemoryEvalTest(unittest.TestCase):
    def test_parallel_width_slices_preserve_width(self):
        self.assertEqual(split_width(709, 128), [128, 128, 128, 128, 128, 69])
        self.assertEqual(split_width(288, 128), [128, 128, 32])
        self.assertEqual(split_width(256, 128), [128, 128])
        self.assertEqual(sum(split_width(18, 128)), 18)

    def test_inventory_defaults_and_total_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.csv"
            path.write_text(
                "module,purpose,depth,width_bits,instances,total_bits\n"
                "memory_8x18,test,8,18,1,144\n",
                encoding="utf-8",
            )
            spec = load_inventory(path)[0]
            self.assertEqual(spec.read_ports, 1)
            self.assertEqual(spec.write_ports, 1)
            self.assertEqual(spec.readwrite_ports, 0)
            self.assertEqual(spec.total_bits, 144)

    def test_inventory_rejects_incorrect_total(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.csv"
            path.write_text(
                "module,purpose,depth,width_bits,instances,total_bits\n"
                "memory_8x18,test,8,18,1,143\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "total_bits mismatch"):
                load_inventory(path)


if __name__ == "__main__":
    unittest.main()
