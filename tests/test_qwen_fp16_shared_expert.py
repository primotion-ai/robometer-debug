import copy
import unittest

import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

from robometer.utils.qwen_kernels import stabilize_qwen_fp16_shared_experts


class SharedExpertFP16Tests(unittest.TestCase):
    def make_block(self):
        cfg = Qwen3_5MoeTextConfig(hidden_size=2, num_experts=2, num_experts_per_tok=1,
                                  moe_intermediate_size=2, shared_expert_intermediate_size=2)
        cfg._experts_implementation = "eager"
        block = Qwen3_5MoeSparseMoeBlock(cfg).half()
        with torch.no_grad():
            for p in block.parameters():
                p.zero_()
            block.shared_expert.gate_proj.weight.fill_(1)
            block.shared_expert.up_proj.weight.fill_(1)
            block.shared_expert.down_proj.weight.fill_(50000)
            block.shared_expert_gate.weight.fill_(-2)
        return block

    def test_gate_before_half_cast_prevents_overflow_and_preserves_gradients(self):
        block = self.make_block()
        x = torch.ones(1, 2, 2, dtype=torch.float16, requires_grad=True)
        self.assertFalse(torch.isfinite(block(x)).all())
        reference = copy.deepcopy(block).float()
        expected = reference(x.detach().float())
        original_keys = list(block.state_dict())
        stabilize_qwen_fp16_shared_experts(block)
        actual = block(x)
        self.assertEqual(actual.dtype, torch.float16)
        torch.testing.assert_close(actual.float(), expected, rtol=0.002, atol=1)
        actual.float().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for parameter in block.parameters():
            self.assertEqual(parameter.dtype, torch.float16)
            if parameter.grad is not None:
                self.assertEqual(parameter.grad.dtype, torch.float16)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(list(block.state_dict()), original_keys)

    def test_bfloat16_path_is_unchanged(self):
        block = self.make_block().bfloat16()
        original = block.forward.__func__
        stabilize_qwen_fp16_shared_experts(block)
        self.assertIs(block.forward.__func__, original)


if __name__ == "__main__":
    unittest.main()
