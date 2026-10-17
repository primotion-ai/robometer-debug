"""Opt-in GPU integration test: RUN_MULTIGPU_TEST=1 with two visible GPUs."""
import os
import unittest

import numpy as np
import torch
from transformers import Qwen3_5MoeConfig, Qwen3_5MoeModel

from robometer.configs.experiment_configs import ExperimentConfig
from robometer.data.dataset_types import Trajectory, ProgressSample, PreferenceSample
from robometer.evals.checkpoint_loader import dispatch_offline_model
from robometer.models.rbm import RBM
from robometer.utils.qwen_kernels import configure_qwen_linear_attention
from robometer.utils.setup_utils import _setup_processor_and_tokenizer, _add_special_tokens_and_resize, setup_batch_collator


@unittest.skipUnless(os.getenv('RUN_MULTIGPU_TEST') == '1', 'Explicit GPU integration test opt-in required')
class MultiGPUOfflineTest(unittest.TestCase):
    def test_moe_multimodal_scores_match_single_gpu(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        torch.manual_seed(42)
        exp = ExperimentConfig()
        exp.model.base_model_id = '/mnt/base_model/Qwen3.5-397B-A17B'
        exp.model.use_unsloth = False
        exp.model.use_multi_image = exp.data.use_multi_image = True
        exp.model.use_per_frame_progress_token = exp.data.use_per_frame_progress_token = True
        exp.model.progress_loss_type = 'discrete'
        exp.model.progress_discrete_bins = 10
        exp.model.average_temporal_patches = False
        processor = _setup_processor_and_tokenizer(exp.model)
        config = Qwen3_5MoeConfig.from_pretrained(exp.model.base_model_id, local_files_only=True)
        text = config.text_config
        for key, value in dict(hidden_size=256, num_hidden_layers=4, num_attention_heads=4,
                               num_key_value_heads=1, num_experts=4, num_experts_per_tok=2,
                               moe_intermediate_size=64, shared_expert_intermediate_size=128,
                               linear_num_key_heads=2, linear_num_value_heads=4,
                               linear_key_head_dim=32, linear_value_head_dim=32, use_cache=False).items():
            setattr(text, key, value)
        text.layer_types = ['linear_attention'] * 3 + ['full_attention']
        for key, value in dict(hidden_size=128, depth=2, num_heads=4, intermediate_size=256,
                               out_hidden_size=256, deepstack_visual_indexes=[]).items():
            setattr(config.vision_config, key, value)
        config._attn_implementation = 'sdpa'
        config._experts_implementation = 'eager'
        backbone = Qwen3_5MoeModel(config)
        _add_special_tokens_and_resize(exp.model, processor, backbone)
        model = RBM(config, processor, processor.tokenizer, base_model=backbone,
                    base_model_id=exp.model.base_model_id, model_config=exp.model).to(dtype=torch.bfloat16)
        configure_qwen_linear_attention(model.model, 'fla')
        model.requires_grad_(False).eval().to('cuda:0')
        frames = np.random.default_rng(42).integers(0, 256, (2, 64, 64, 3), dtype=np.uint8)
        a = Trajectory(frames=frames, frames_shape=frames.shape, task='Pick up the block')
        b = Trajectory(frames=frames[::-1].copy(), frames_shape=frames.shape, task=a.task)
        collator = setup_batch_collator(processor, processor.tokenizer, exp, is_eval=True)
        inputs = {}
        for kind, sample in [('progress', ProgressSample(trajectory=a)),
                             ('preference', PreferenceSample(chosen_trajectory=a, rejected_trajectory=b))]:
            inputs[kind] = {k: v.to('cuda:0') if torch.is_tensor(v) else v
                            for k, v in collator([sample])[kind + '_inputs'].items()}

        def scores():
            result = {}
            with torch.inference_mode():
                for kind, batch in inputs.items():
                    output, _ = model(**dict(batch, sample_type=kind))
                    for name in ['progress_logits', 'success_logits', 'pref_logits']:
                        value = getattr(output, name)
                        for side, tensor in (value.items() if isinstance(value, dict) else [('', value)]):
                            if tensor is not None:
                                result[(kind, name, side)] = tensor.float().cpu()
            return result

        reference = scores()
        model.cpu()
        placement = {name: 0 for name, _ in model.named_children() if name != 'model'}
        placement.update({'model.visual': 0, 'model.language_model.embed_tokens': 0,
                          'model.language_model.rotary_emb': 0, 'model.language_model.norm': 1})
        placement.update({f'model.language_model.layers.{i}': int(i >= 2) for i in range(4)})
        model = dispatch_offline_model(model, torch.device('cuda:0'), placement)
        actual = scores()
        self.assertEqual(reference.keys(), actual.keys())
        for key in reference:
            self.assertTrue(torch.isfinite(actual[key]).all(), key)
            torch.testing.assert_close(actual[key], reference[key], rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
