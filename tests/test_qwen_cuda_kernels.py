"""Opt-in CUDA correctness checks: RBM_TEST_CUDA_KERNELS=1 python -m unittest ..."""

import copy
import os
import unittest

import torch
from transformers import Qwen3_5Model, Qwen3_5MoeModel

from test_qwen_backbones import tiny_config
from robometer.utils.qwen_kernels import configure_qwen_linear_attention


@unittest.skipUnless(os.environ.get("RBM_TEST_CUDA_KERNELS") == "1", "opt-in CUDA kernel test")
class QwenCudaKernelTests(unittest.TestCase):
    def test_dense_and_moe_forward_backward(self):
        torch.set_num_threads(8)
        torch.manual_seed(42)
        for kind, cls in [("qwen3_5", Qwen3_5Model), ("qwen3_5_moe", Qwen3_5MoeModel)]:
            with self.subTest(kind=kind):
                config = tiny_config(kind)
                config.text_config.linear_key_head_dim = 128
                config.text_config.linear_value_head_dim = 128
                config.vision_config.hidden_size = 288  # actual vision head dimension: 72
                config._attn_implementation = "sdpa"
                # PyTorch 2.8's grouped_mm only supports SM90, not B200/SM100.
                config._experts_implementation = "eager"
                reference = configure_qwen_linear_attention(cls(config), "torch").to("cuda", torch.bfloat16)
                reference.visual.requires_grad_(False)
                fused = configure_qwen_linear_attention(copy.deepcopy(reference), "fla")
                ids = torch.tensor([[1, 8, 7, 9, 5, 8, 7, 9, 5, 6]], device="cuda")
                inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                              mm_token_type_ids=(ids == 7).long(),
                              pixel_values=torch.randn(8, 3 * 2 * 16 * 16, device="cuda", dtype=torch.bfloat16),
                              image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]], device="cuda"), use_cache=False)
                outputs = []
                loss_weight = torch.randn(1, ids.shape[1], config.text_config.hidden_size, device="cuda")
                for model in (reference, fused):
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    output = model(**inputs).last_hidden_state
                    (output.float() * loss_weight).mean().backward()
                    outputs.append(output.detach())
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0.05, atol=0.03)
                torch.testing.assert_close(reference.language_model.embed_tokens.weight.grad,
                                           fused.language_model.embed_tokens.weight.grad, rtol=0.1, atol=0.005)

    def test_actual_linear_head_shapes(self):
        torch.set_num_threads(8)
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

        for heads in (48, 64):
            with self.subTest(value_heads=heads):
                torch.manual_seed(42)
                inputs = [torch.randn(1, 128, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
                inputs += [-torch.rand(1, 128, heads, device="cuda"),
                           torch.rand(1, 128, heads, device="cuda", dtype=torch.bfloat16)]
                outputs, gradients = [], []
                for kernel in (torch_chunk_gated_delta_rule, chunk_gated_delta_rule):
                    q, k, v, g, beta = [x.clone().requires_grad_() for x in inputs]
                    output, _ = kernel(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
                    output.float().square().sum().backward()
                    outputs.append(output.detach())
                    gradients.append([x.grad for x in (q, k, v, g, beta)])
                torch.testing.assert_close(*outputs, rtol=0.05, atol=0.005)
                for ref, actual in zip(*gradients):
                    torch.testing.assert_close(ref, actual, rtol=0.1, atol=0.04)


if __name__ == "__main__":
    unittest.main()
