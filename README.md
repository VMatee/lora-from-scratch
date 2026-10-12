# LoRA from Scratch — Low-Rank Adaptation in PyTorch

[![Tests](https://github.com/VMatee/lora-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/VMatee/lora-from-scratch/actions/workflows/tests.yml)

Understand how **LoRA** reduces the number of trainable parameters using the update
`W' = W + (alpha/r) * (B @ A)` with a frozen base matrix `W`.

This is a compact, independent PyTorch implementation of a **LoRA-adapted linear
layer**. It deliberately avoids `peft` and Hugging Face adapters so every learned
matrix and operation is visible. It is NOT an end-to-end LLM fine-tuning system.

## Try it

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m lora_lab.demo --steps 160
python -m lora_lab.inject_demo
```

The toy experiment fits a synthetic low-rank correction to a frozen layer;
compare initial and final MSE and the fraction of parameters trained.
Results are not LLM benchmark results.

## From-scratch adapter injection

The code also contains a hand-written module traversal and injection layer:
insert LoRA into selected nested linear layers, freeze the base model,
save **only** the low-rank A/B matrices, and verify merge/unmerge equivalence.
This supports small neural networks without using PEFT or Transformers.

See [the implementation and mathematics](docs/implementation.md).

## What happens

```text
input x  ----> [ Frozen linear layer W, b ] -----+----> output
          \                                    /
           +--> [ trainable A: r x in ] ---------|
                  [ trainable B: out x r ] * alpha/r
```

- `src/lora_lab/layers.py` implements a reusable `LoRALinear`.
- `src/lora_lab/demo.py` trains the adapter against synthetic targets.
- `tests/test_lora.py` covers initial equality, gradients, trained parameter
  count, merge/unmerge equivalence, and invalid ranks.

## Design details

- The initial `B` is zero, so before training the adapter exactly matches its
  frozen base layer.
- The only trainable parameters are A and B; the original bias is frozen.
- Merge/unmerge are inference conveniences, not operations to interleave with
  optimizer steps.
- Rank must be between 1 and `min(in_features,out_features)` here; full-rank
  adapters are outside the pedagogical scope.

## Where this stops

No QLoRA quantization, paged optimizer, pretrained checkpoint integration,
checkpoint compatibility, distributed training, or task benchmarks. These
would be natural extensions after understanding the matrix update.

**Reading:** Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*.

## Interactive notebook

Open [the step-by-step notebook](notebooks/01_walkthrough.ipynb) in Jupyter after installing this repository with `python -m pip install -e ".[dev]"`.

## Architecture walk-through

See [the implementation map](docs/architecture.md) for a component-by-component overview.

## Related educational reading

Sebastian Raschka maintains a useful [independent educational reference](https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-E). This repository uses its own small code examples and does not reproduce his book content.
