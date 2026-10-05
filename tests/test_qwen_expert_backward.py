import copy
import unittest

import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

from robometer.utils.qwen_kernels import optimize_qwen_moe_expert_backward


class ExpertBackwardTests(unittest.TestCase):
    def test_matches_original_outputs_and_all_gradients_with_unused_expert(self):
        for dtype in (torch.float64, torch.float16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(42)
                config = Qwen3_5MoeTextConfig(hidden_size=8, num_experts=4, num_experts_per_tok=2,
                                             moe_intermediate_size=4)
                config._experts_implementation = "eager"
                original = Qwen3_5MoeExperts(config).to(dtype)
                with torch.no_grad():
                    for p in original.parameters():
                        p.normal_(std=0.1)
                optimized = copy.deepcopy(original)
                optimize_qwen_moe_expert_backward(optimized)
                indices = torch.tensor([[0, 2], [1, 2], [2, 0]])  # Expert 3 unused.
                inputs = torch.randn(3, 8, dtype=dtype, requires_grad=True)
                weights = torch.tensor([[0.3, 0.7], [0.6, 0.4], [0.8, 0.2]], dtype=dtype, requires_grad=True)
                x = inputs.detach().clone().requires_grad_()
                w = weights.detach().clone().requires_grad_()
                expected = original(inputs, indices, weights)
                actual = optimized(x, indices, w)
                tolerance = dict(rtol=0.003, atol=0.0001) if dtype == torch.float16 else dict(rtol=1e-9, atol=1e-12)
                torch.testing.assert_close(actual, expected, **tolerance)
                expected.square().mean().backward()
                actual.square().mean().backward()
                torch.testing.assert_close(x.grad, inputs.grad, **tolerance)
                torch.testing.assert_close(w.grad, weights.grad, **tolerance)
                for old, new in zip(original.parameters(), optimized.parameters()):
                    torch.testing.assert_close(new.grad, old.grad, **tolerance)
                    self.assertEqual(new.grad.dtype, dtype)
                    self.assertEqual(torch.count_nonzero(new.grad[3]), 0)
                self.assertEqual(list(original.state_dict()), list(optimized.state_dict()))


if __name__ == "__main__":
    unittest.main()
