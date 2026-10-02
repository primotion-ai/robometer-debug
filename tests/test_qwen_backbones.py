"""Small real Transformers models exercise loading and all three RBM heads."""

import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, Qwen3VLForConditionalGeneration, Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration

from robometer.configs.experiment_configs import ModelConfig
from robometer.models.rbm import RBM
from robometer.utils.setup_utils import _load_base_model_standard
from scripts.launch_b200 import check_weights


def tiny_config(model_type):
    return AutoConfig.for_model(
        model_type,
        image_token_id=7, video_token_id=10, vision_start_token_id=8, vision_end_token_id=9,
        text_config=dict(
            vocab_size=32, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            use_cache=False,
            **(dict(layer_types=["linear_attention", "full_attention"],
                    linear_num_key_heads=2, linear_num_value_heads=4,
                    linear_key_head_dim=16, linear_value_head_dim=16,
                    num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
                    shared_expert_intermediate_size=32) if model_type != "qwen3_vl" else {}),
            rope_parameters=dict(rope_type="default", rope_theta=10000.0,
                                 partial_rotary_factor=1.0, mrope_section=[2, 3, 3]),
        ),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                           out_hidden_size=64, num_position_embeddings=16,
                           patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
                           **({"deepstack_visual_indexes": [0]} if model_type == "qwen3_vl" else {})),
    )


class QwenBackboneTests(unittest.TestCase):
    def test_pretrained_dense_and_moe_multimodal_backward(self):
        torch.set_num_threads(2)
        for kind, cls in [("qwen3_vl", Qwen3VLForConditionalGeneration),
                          ("qwen3_5", Qwen3_5ForConditionalGeneration),
                          ("qwen3_5_moe", Qwen3_5MoeForConditionalGeneration)]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix="Qwen3-test-") as folder:
                original = cls(tiny_config(kind))
                # Write the upstream conditional-generation keys exactly; avoid
                # the library's reverse text-only conversion during saving.
                from pathlib import Path
                original.config.save_pretrained(folder)
                save_file(original.state_dict(), str(Path(folder) / "model.safetensors"))
                check_weights(folder)
                cfg = ModelConfig(base_model_id=folder, use_multi_image=True,
                                  use_per_frame_progress_token=True, attn_implementation="sdpa",
                                  linear_attention_backend="torch", experts_implementation="eager")
                backbone = _load_base_model_standard(cfg, torch.float32, {"attn_implementation": "sdpa"}, None)
                if kind != "qwen3_vl":
                    self.assertEqual(backbone.config.text_config._experts_implementation, "eager")
                # A successful load must retain every language and vision weight,
                # not silently initialize the backbone after a prefix mismatch.
                for name, weight in original.model.state_dict().items():
                    torch.testing.assert_close(backbone.state_dict()[name], weight, rtol=0, atol=0)
                tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: {
                    "<|prog_token|>": 5, "<|pref_token|>": 6,
                }[token])
                model = RBM(backbone.config, SimpleNamespace(tokenizer=tokenizer), tokenizer,
                            base_model=backbone, base_model_id=folder, model_config=cfg)
                model.train()
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                ids = torch.tensor([[1, 8, 7, 9, 5, 8, 7, 9, 5, 6]])
                output, _ = model(
                    input_ids=ids, attention_mask=torch.ones_like(ids), sample_type="preference",
                    pixel_values=torch.randn(8, 3 * 2 * 16 * 16),
                    image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]),
                )
                values = [output.pref_logits, *output.progress_logits.values(), *output.success_logits.values()]
                loss = sum(value.float().square().mean() for value in values if value is not None)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for head in (model.progress_head, model.preference_head, model.success_head):
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))
                self.assertIsNotNone(backbone.language_model.embed_tokens.weight.grad)
                # The preflight must reject incomplete downloads.
                from pathlib import Path
                weight_path = Path(folder) / "model.safetensors"
                with weight_path.open("r+b") as stream:
                    stream.truncate(weight_path.stat().st_size - 1)
                with self.assertRaisesRegex(ValueError, "Truncated"):
                    check_weights(folder)


if __name__ == "__main__":
    unittest.main()
