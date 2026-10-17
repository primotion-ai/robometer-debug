"""Small real-FSDP integration harness for the production ICL loop.

Launch with accelerate --fsdp_transformer_layer_cls_to_wrap ICLTestBlock.
Replaces only pretrained weight loading; uses real images/processor, optimizer,
checkpoint writer/reader, evaluation and deterministic candidate sampling.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch import nn
from transformers import AutoProcessor
from robometer.icl_progress import train
from robometer.icl_progress.model import ICLProgressModel


class ICLTestBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32,32)

    def forward(self, x):
        return x + self.linear(x).tanh()


class TinyBackbone(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=32))
        self.embedding = nn.Embedding(vocab_size,32)
        self.layers = nn.ModuleList([ICLTestBlock(), ICLTestBlock()])

    def forward(self, input_ids, **kwargs):
        h = self.embedding(input_ids)
        for layer in self.layers:
            h = layer(h)
        return SimpleNamespace(last_hidden_state=h)


def tiny_load(cfg):
    processor = AutoProcessor.from_pretrained('/mnt/base_model/Qwen3-VL-4B-Instruct',local_files_only=True)
    # Reproduce the real 27B/397B nonzero-rank meta initialization path.
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        from accelerate import init_empty_weights
        with init_empty_weights():
            backbone = TinyBackbone(len(processor.tokenizer))
    else:
        backbone = TinyBackbone(len(processor.tokenizer))
    return ICLProgressModel(backbone,16), processor


train.load_model = tiny_load
train.main()
