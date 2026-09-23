# RQNKUP implementations

**paper title:** HAWQ-V3: Dyadic Neural Network Quantization

## ⚲ project summary
**HAWQ-V3** is a method that shrinks deep learning models so they can run on cheap hardware chips (like microcontrollers or phone processors) using **100% integer math (INT8/INT4)**—completely eliminating the need for decimal/floating-point math during execution. To figure out which layers get 8-bit, 4-bit, or 2-bit precision, HAWQ-V3 measures how "fragile" each layer is by calculating its **average loss steepness (Hessian Trace)**. Then, it uses a super-fast math solver (Integer Linear Programming) to automatically assign the best bit sizes across the whole network in milliseconds.

## ⌖ motivation
Traditional quantization methods hit two big roadblocks:

1. **The "Decimal Trap" in Hardware**: Most "compressed" models still cheat during execution—they switch back to decimals (floats) for tricky steps like scaling or adding skip connections. Cheap hardware chips don't have decimal math units, so they freeze up or run super slowly when forced to emulate decimals.
2. **Too Many Combinations**: Trying to hand-pick the right bit size (8-bit, 4-bit, or 2-bit) for every single layer in a deep model is impossible because there are billions of possible combinations.

HAWQ-V3 solves both: it makes models run 100% on pure-integer hardware while automatically selecting the best bit setup in seconds.

## ✎ᝰ novelty
* 100% Integer Math (Dyadic Scaling): Forces all scale math to use "dyadic numbers" (integers divided by powers of 2, like $m \cdot 2^{-e}$). This means hardware can perform all scaling using fast bit shifts instead of slow decimal division.
* Hessian Trace (Average Layer Fragility): Instead of measuring just the single steepest direction of a layer, it measures the average steepness across all directions (Hessian Trace) to get a truer picture of overall layer fragility.
* Instant Bit Solver (ILP): Instead of spending days training reinforcement learning agents to guess bit allocations, it uses an off-the-shelf math solver (ILP) that calculates the perfect bit budget in milliseconds.

## ✰ methodology
1. **dataset(s) used:**
* [ImageNet (ILSVRC2012)](https://www.google.com/url?sa=E&amp;q=https%3A%2F%2Fwww.image-net.org%2F)
* [CIFAR-10 / CIFAR-100](https://www.google.com/url?sa=E&amp;q=https%3A%2F%2Fwww.cs.toronto.edu%2F~kriz%2Fcifar.html)

2. **model(s) used:**
* ResNet-18, ResNet-50
* Inception-V3
* MobileNetV2

3. **sensitivity metric**: Hessian Trace ($\text{Tr}(H_i)$). Measures how "bumpy" or fragile a layer's error rate is on average using a quick numerical trick (Hutchinson's algorithm) rather than building massive derivative matrices.

4. **quantization pipeline**: Uniform asymmetric and symmetric dyadic integer linear quantization. Weights and activations are quantized into discrete integer bins, with layer scale factors constrained to dyadic form ($m \cdot 2^{-e}$). (Official HAWQ GitHub: [zhen-dong/hawq](https://github.com/zhen-dong/hawq))

5. **bit budgeting pipeline**: Feeds layer fragility scores ($\text{Tr}(H_i)$) and hardware limits into an Integer Linear Programming (ILP) solver (like PuLP or Gurobi) to select the optimal bit-width per layer in milliseconds.

6. **evaluation/metrics used:**
* Top-1 &amp; Top-5 Classification Accuracy (%)
* Model Memory Size (MB)
* BitOps (Bit Operations)
* Real Hardware Latency (ms) and Frames Per Second (FPS) on TensorRT/TVM engines.

7. **results:**
* **ResNet-50 (ImageNet):** Reached **75.39% Top-1 accuracy** under 4-bit/8-bit integer-only execution, running **1.5× to 2.3× faster** on real integer hardware.
* **MobileNetV2 (ImageNet):** Reached **71.56% Top-1 accuracy** running 100% on integer math without floating-point fallbacks.

## ⛰︎ impact
HAWQ-V3 proved that neural networks can run 100% on cheap integer hardware without relying on floating-point fallback units. By combining Hessian Trace sensitivity with Integer Linear Programming, it was shown that extreme mixed-precision compression can be computed in seconds and deployed on real edge hardware, achieving massive speedups.

#### future work
HAWQ-V3 sets the gold standard for integer-only execution and instant ILP bit-solving. However, its main flaw remains data dependency: computing Hessian traces ($\text{Tr}(H_i)$) requires passing real images through the model, incurring roughly 20 backpropagation passes per block. In RQNKUP, we can adopt HAWQ-V3's dyadic integer engine and ILP solver, but replace its backprop-heavy Hessian Trace calculation with our data-free weight metrics (like excess kurtosis for outliers and spectral rank for structure). This allows RQNKUP to output fully deployable, integer-only mixed-precision models with zero data, zero backprop passes, and zero privacy risks.

## **additional sources:**
* [HAWQ-V3 Paper (ICML 2021)](https://www.google.com/url?sa=E&amp;q=https%3A%2F%2Farxiv.org%2Fabs%2F2011.10680)
* [Official HAWQ GitHub Repository (zhen-dong/hawq)](https://www.google.com/url?sa=E&amp;q=https%3A%2F%2Fgithub.com%2Fzhen-dong%2Fhawq)
* [ZeroQ GitHub Repository (amirgholami/zeroq)](https://www.google.com/url?sa=E&amp;q=https%3A%2F%2Fgithub.com%2Famirgholami%2Fzeroq)
* [PuLP Linear Programming Python Library](https://www.google.com/url?sa=E&amp;q=https%3A%2F%2Fcoin-or.github.io%2Fpulp)
