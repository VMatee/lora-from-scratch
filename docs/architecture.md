# Architecture note — Low rank adaptation

```text
x -> frozen W and b; x -> trainable A and B -> scale alpha/r -> sum -> y
```

## Fidelity and assumptions

Merge and unmerge adapter weights only outside optimizer steps.

Read the source implementation and the tests together. New features should
add controlled experiments or unit tests before using benchmark claims.
