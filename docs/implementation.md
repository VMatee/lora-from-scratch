# LoRA adapter injection without PEFT

The original LoRALinear in src/lora_lab/layers.py calculates two low-rank matrix products. This upgrade adds explicit traversal and replacement of nested linear layers while retaining the same basic forward path and merging mathematics.

## Mathematics

For a frozen weight matrix W of shape (out, in), LoRA learns A of shape (r, in) and B of shape (out, r).

Updated output = x @ W.T + bias + (alpha/r) * ((x @ A.T) @ B.T).

The original base parameters do not receive gradients. An adapter has r*(in + out) trainable parameters versus in*out dense weights. B starts at zeros, so inserting LoRA into a model does not change its initial predictions.

## Modules and behavior

- src/lora_lab/layers.py: forward low-rank update and merge/unmerge.
- src/lora_lab/inject.py: recursively locate Linear modules, select exact dotted names, install LoRA, freeze non-adapter parameters and snapshot A/B tensors.
- src/lora_lab/inject_demo.py: construct a small multilayer network, inject adapters, fit a synthetic target, and compare merged output.
- tests/test_inject.py: numerical equivalence with an explicit matrix formula; nested module replacement; no initial behavior change; checkpoint compatibility.

Layer replacement does not require Hugging Face, PEFT, pretrained weights, CUDA, or cloud inference. PyTorch remains the tensor/autograd backend.

## Run

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m lora_lab.demo --steps 160
python -m lora_lab.inject_demo
~~~

All comparisons are local synthetic regression tests, not a large-language-model fine-tuning benchmark. Optimized tensor kernels are retained to avoid sacrificing training speed merely to increase source lines.
