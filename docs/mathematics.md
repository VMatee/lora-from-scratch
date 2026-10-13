# Low-rank adaptation: derivation, implementation and verification

*How a neural network can adapt a frozen linear layer by training only two smaller matrices.*

When a model has millions of weights, updating the entire model may be unnecessary for a small change in behavior. Low-Rank Adaptation (LoRA) proposes a practical alternative: keep the original weight matrix fixed and represent the learned correction as the product of two thinner matrices. This repository implements that idea directly in PyTorch, including insertion into nested modules and explicit checks that the transformed layer behaves as expected.

## 1. Start with the ordinary linear layer

For an input feature vector $x\in\mathbb{R}^{d_{\mathrm{in}}}$, a conventional dense layer computes

$$
y=Wx+b,
\qquad
W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}.
$$

An ordinary full-parameter adaptation would update all $d_{\mathrm{out}}d_{\mathrm{in}}$ elements of $W$. Instead, LoRA holds $W$ and $b$ fixed and learns a correction $\Delta W$:

$$
y_{\mathrm{LoRA}}=(W+\Delta W)x+b.
$$

The central assumption is not that $W$ has low rank; it is that a useful *change* to $W$ may be approximated by a low-rank factorization.

## 2. Derive the low-rank update

Choose an adapter rank $r$ satisfying $1\le r\le\min(d_{\mathrm{in}},d_{\mathrm{out}})$. Construct two trainable matrices:

$$
A\in\mathbb{R}^{r\times d_{\mathrm{in}}},
\qquad
B\in\mathbb{R}^{d_{\mathrm{out}}\times r}.
$$

Their product has exactly the shape of the original weight matrix. The update used in [layers.py](../src/lora_lab/layers.py) is

$$
\boxed{
\Delta W=\frac{\alpha}{r}BA
}
$$

where $\alpha>0$ is a user-configured scale. Therefore,

$$
\boxed{
y_{\mathrm{LoRA}}
=Wx+b+\frac{\alpha}{r}B(Ax).
}
$$

The parentheses matter computationally. First multiply $A$ by $x$, producing an $r$-dimensional intermediate vector; then multiply by $B$. The code performs the two small linear operations directly rather than constructing a dense correction matrix at every forward pass.

### Worked example using actual matrices

Let the input and output dimensions both be 2, let $r=1$, and take $\alpha=1$. Start with an identity base layer, no bias, and the following adapter factors:

$$
W=\begin{bmatrix}1&0\\0&1\end{bmatrix},
\qquad
A=\begin{bmatrix}1&-1\end{bmatrix},
\qquad
B=\begin{bmatrix}2\\3\end{bmatrix}.
$$

The correction becomes

$$
BA=
\begin{bmatrix}2&-2\\3&-3\end{bmatrix},
\qquad
W+\Delta W=
\begin{bmatrix}3&-2\\3&-2\end{bmatrix}.
$$

For $x=[1,2]^\top$, the frozen base produces $Wx=[1,2]^\top$. The adapter first gives $Ax=-1$ and then $B(Ax)=[-2,-3]^\top$. The combined output is

$$
y_{\mathrm{LoRA}}=
\begin{bmatrix}1\\2\end{bmatrix}
+
\begin{bmatrix}-2\\-3\end{bmatrix}
=
\begin{bmatrix}-1\\-1\end{bmatrix}.
$$

Computing the merged matrix $(W+BA)x$ gives the same result, illustrating the merge/unmerge equivalence checked by the test suite.

## 3. Where the parameter savings come from

The dense layer has $d_{\mathrm{out}}d_{\mathrm{in}}$ weight elements. The two LoRA matrices together have

$$
N_{\mathrm{adapter}}
=rd_{\mathrm{in}}+d_{\mathrm{out}}r
=r(d_{\mathrm{in}}+d_{\mathrm{out}})
$$

trainable elements. Ignoring the frozen bias, their ratio is

$$
\rho=\frac{r(d_{\mathrm{in}}+d_{\mathrm{out}})}
{d_{\mathrm{in}}d_{\mathrm{out}}}.
$$

As a numerical illustration, suppose $d_{\mathrm{in}}=d_{\mathrm{out}}=4096$ and $r=8$:

$$
N_{\mathrm{dense}}=4096^2=16\,777\,216,
\qquad
N_{\mathrm{adapter}}=8(4096+4096)=65\,536,
$$

$$
\rho=0.00390625=0.390625\%.
$$

Thus, **only 0.39% as many weight parameters would be trainable** for that specific linear layer. This is a parameter-count example, not a measured training-memory or throughput reduction. Actual memory use also depends on activations, gradients, optimizer state and the rest of the network.

## 4. How the gradients reach the adapter

The base layer is frozen by setting its parameters to not require gradients. Gradients still flow through its output and through the low-rank branch.

For one training example, define $s=\alpha/r$, $u=Ax$ and $g=\partial\mathcal{L}/\partial y$. From

$$
y=Wx+b+sBu
$$

the chain rule gives the matrix gradients

$$
\frac{\partial\mathcal{L}}{\partial B}
=s\,g\,u^\top,
$$

$$
\frac{\partial\mathcal{L}}{\partial A}
=s\,(B^\top g)\,x^\top.
$$

The implementation initializes $A$ with small random values and initializes $B=0$. Initially, $\Delta W=0$, so the adapted layer exactly matches its original base layer. There is a useful subtlety: at the first backward pass, $\partial\mathcal{L}/\partial A=0$ because $B$ is still zero, whereas the gradient for $B$ can be nonzero. Once $B$ starts changing, both matrices can learn.

These properties are checked using actual PyTorch gradients in [test_lora.py](../tests/test_lora.py) and [test_inject.py](../tests/test_inject.py).

## 5. Merging the weights for inference

After training, the learned correction can be incorporated into the frozen weight matrix once:

$$
W_{\mathrm{merged}}
=W+sBA.
$$

Inference then uses the ordinary dense computation $W_{\mathrm{merged}}x+b$. The corresponding unmerge operation subtracts $sBA$ to recover the original $W$.

There are three operational restrictions worth noting. First, merging twice must not add the correction twice; [LoRALinear](../src/lora_lab/layers.py) maintains a flag to prevent that. Second, changing $A$ or $B$ while the weights remain merged would break this simple equivalence. Third, floating-point roundoff means outputs are expected to be *numerically close*, not necessarily bit-for-bit identical.

## 6. Adapting several layers without a high-level adapter library

A real network may contain nested linear modules such as an encoder's projections and an output layer. [inject.py](../src/lora_lab/inject.py) recursively walks the module tree, selects layers by their dotted module names, verifies the selected ranks, replaces the linear modules with LoRA wrappers and freezes everything except the adapter factors.

A lightweight checkpoint stores, for each adapted layer, the name, rank, scale and the tensors $A$ and $B$. This matters because the adapter can be stored without duplicating the original base weights. Restoring such a checkpoint requires a compatible base model with the same module structure and initial weights; the adapter checkpoint by itself is not a complete standalone model.

## 7. How the experiment is evaluated

The bundled demonstration constructs a synthetic target that differs from a frozen layer by a low-rank correction. Training minimizes mean squared error:

$$
\operatorname{MSE}
=\frac{1}{n}\sum_{i=1}^{n}
\left\|f_{\mathrm{LoRA}}(x_i)-y_i^\star\right\|_2^2
\Big/ d_{\mathrm{out}}.
$$

This expression is the mean over samples and output features, matching the tensor mean of squared elementwise differences in [demo.py](../src/lora_lab/demo.py). The script prints the MSE before and after adaptation and the ratio of trained parameters. Numerical tests also check merge/unmerge equivalence and adapter-only checkpoint loading.

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m lora_lab.demo --steps 160
python -m lora_lab.inject_demo
~~~

### What the results mean

A lower synthetic MSE demonstrates that the implementation can optimize a low-rank change in the controlled example. It does not establish the quality of fine-tuning a pretrained language model, and the repository does not implement QLoRA, quantized optimizer states or multi-GPU training. Those require separate engineering and evaluation.

**Source crosswalk:** [low-rank layer](../src/lora_lab/layers.py) · [adapter injection and checkpoints](../src/lora_lab/inject.py) · [demonstration](../src/lora_lab/demo.py) · [implementation guide](implementation.md).
