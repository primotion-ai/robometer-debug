import json
import tempfile
import unittest
from pathlib import Path

import torch

from robometer.utils.numerics import check_finite_outputs, install_finite_output_hooks


class NumericsTests(unittest.TestCase):
    def test_finite_nested_outputs_keep_autograd(self):
        x = torch.tensor([2.0], requires_grad=True)
        with tempfile.TemporaryDirectory() as directory:
            check_finite_outputs({"success": {"A": x, "B": None}, "other": [torch.tensor([1])]},
                                 location="heads", output_dir=directory, rank=0)
            self.assertEqual(list(Path(directory).iterdir()), [])
            x.square().sum().backward()
            self.assertEqual(x.grad.item(), 4.0)

    def test_invalid_outputs_are_reported_not_sanitized(self):
        x = torch.tensor([float("nan"), float("inf"), 0.0], dtype=torch.float16)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FloatingPointError, "heads.success.A"):
                check_finite_outputs({"success": {"A": x}}, location="heads",
                                     output_dir=directory, rank=3, step=0)
            report = json.loads((Path(directory) / "first_nonfinite_rank_3.json").read_text())
            self.assertEqual((report["nan_count"], report["inf_count"], report["step"]), (1, 1, 0))
            self.assertTrue(torch.isnan(x[0]))

    def test_hook_locates_overflow_before_later_nan(self):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False), torch.nn.LayerNorm(1)).half()
        with torch.no_grad():
            model[0].weight.fill_(2)
        with tempfile.TemporaryDirectory() as directory:
            handles = install_finite_output_hooks(model, output_dir=directory, rank=0)
            with self.assertRaisesRegex(FloatingPointError, "at 0:"):
                model(torch.tensor([[40000.0]], dtype=torch.float16))
            for handle in handles:
                handle.remove()
            self.assertTrue(torch.isnan(model(torch.tensor([[40000.0]], dtype=torch.float16))).all())


if __name__ == "__main__":
    unittest.main()
