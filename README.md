# RQNKUP implementations

**paper title:** HAWQ-V3: Dyadic Neural Network Quantization

## ⚲ project summary
**HAWQ-V3** is a mixed-precision quantization framework that lets deep learning models run on hardware without floating-point support using **100% integer math (INT8, INT4, or mixed INT4/8)**. Every operation during inference is an integer multiplication, integer addition, or bit shift. To figure out which layers get 8-bit or 4-bit precision, HAWQ-V3 scores how "fragile" each layer is with a **Hessian-based sensitivity score** (from HAWQ-V2), then feeds those scores, along with hardware measurements, into an Integer Linear Programming (ILP) solver.

## ⌖ motivation
Traditional quantization methods hit two big roadblocks:

1. **The "Decimal Trap" in Hardware**: Many "compressed" models use *simulated* (fake) quantization. Values are stored as low-bit integers but cast back to decimals (FP32) for inference, and steps like rescaling, batch normalization, and residual (skip) connections are computed in floating point. This adds overhead, prevents the model from using fast low-precision integer units, and means it cannot be deployed as-is on integer-only hardware. Because of mismatches such as FP32 batch-norm parameters and FP32 residual accumulation (quantization is not linear: $Q(a+b) \neq Q(a) + Q(b)$ ), a model fine-tuned with fake quantization behaves differently when run with true integer arithmetic. For ResNet-50 at uniform INT4, the normalized difference between the PyTorch (fake-quantized) and TVM (integer-only) feature maps grows layer by layer to **more than 95%** at the final layer.
2. **Too Many Combinations**: Choosing a bit-width (4-bit or 8-bit) for every layer creates a search space that grows exponentially with depth ( $B^L$ for $B$ bit choices and $L$ layers). Brute-force search is infeasible, and search methods such as reinforcement learning are computationally expensive and sensitive to hyperparameters and initialization.

HAWQ-V3 solves both: it runs models with integer-only arithmetic, and once layer sensitivities are computed, it selects a mixed-precision setting in under a second.

## ✎ᝰ novelty
* 100% Integer Math (Dyadic Scaling): Every requantization multiplier (e.g., $S_w S_h / S_a$ ) is converted to a "dyadic number," an integer divided by a power of 2 ( $b / 2^c$ ). Rescaling therefore becomes an INT32 integer multiplication followed by a bit shift, with no floating-point operations and no integer division anywhere in inference. Jacob et al. (2018) used this approach for INT8; HAWQ-V3 extends it to low-precision and mixed-precision quantization.
* Integer-Only BatchNorm and Skip Connections: BN is folded into the convolution. The BN standard deviation (σ) is merged into the weight scale, and the BN mean (µ) is quantized to INT32 and merged into the bias. Jacob et al. fold BN while its statistics are still updating, which requires computing each convolution twice. HAWQ-V3 instead keeps Conv and BN unfolded for several epochs, then freezes the BN statistics and folds them. The paper credits this, in part, for INT8 accuracy up to about 5% higher than Jacob et al. Residual connections are kept in INT32 and added with dyadic rescaling ( $q_a = DN(S_m/S_a) \cdot q_m + DN(S_r/S_a) \cdot q_r$ ). Inception's concatenation layers are handled the same way, with the pooling branch computed in INT32.
* Hessian Sensitivity (inherited from HAWQ-V2): The layer sensitivity is HAWQ-V2's Hessian-trace-based perturbation. Instead of using only the single steepest direction (the top Hessian eigenvalue, as in HAWQ-V1), the trace captures the average curvature across all directions, giving a fuller picture of a layer's fragility.
* Hardware-Aware Instant Bit Solver (ILP): Bit allocation is posed as an Integer Linear Programming problem and solved with an off-the-shelf solver in under a second on a laptop, compared to more than 10/50 hours of RL-based search (HAQ) for ResNet-50/Inception-V3 on 4 GPUs. Unlike HAWQ-V2's Pareto-frontier method, whose complexity grows exponentially with the number of constraints, the ILP easily handles multiple constraints. Unlike the contemporary ILP of Hubara et al. (2020), it is hardware-aware: it uses per-layer latency measured on the target hardware, because some layers get no speedup from INT4 while others benefit superlinearly.
* INT4 Support in TVM: To the authors' knowledge, the first framework to add INT4 support to Apache TVM. It supports uniform INT4 and mixed INT4/8 inference on NVIDIA T4 Tensor Cores by packing eight 4-bit values into one INT32 and adding a new direct-convolution schedule, so speedups are measured on real hardware rather than simulated.

## ✰ methodology
1. **dataset(s) used:**
    * [ImageNet (ILSVRC2012)](https://www.image-net.org/)

2. **model(s) used:**
    * ResNet-18, ResNet-50
    * Inception-V3
    * (pretrained models from PyTorchCV with no architectural changes; ResNet-101 used as the distillation teacher)

3. **sensitivity metric**: HAWQ-V2's Hessian-based perturbation, $\Omega_i = \overline{\mathrm{Tr}}(H_i) \cdot \lVert Q(W_i) - W_i \rVert_2^2$ , where $\overline{\mathrm{Tr}}(H_i)$ is the average Hessian trace (the trace of the layer's Hessian divided by its number of parameters). It measures how "bumpy" or fragile a layer's loss is on average, scaled by how much quantizing that layer changes its weights. The trace is computed with PyHessian, which uses Hutchinson's randomized trace estimator instead of building the full Hessian matrix. Computing it for all layers of ResNet-50/Inception-V3 takes under 30 minutes on 4 RTX 6000 GPUs.

4. **quantization pipeline**: Uniform, static quantization, $Q(r) = \mathrm{Int}(r/S) - Z$ , with all scaling factors precomputed. Weights use symmetric per-channel quantization ( $Z = 0$ ). Activations use asymmetric per-layer quantization, with their min/max ranges tracked during fine-tuning using momentum 0.99. BN is folded into the convolution; folded weights are quantized to 4 or 8 bits and biases to INT32, with the bias scale set to $S_b = S_h S_W$ so it can be added directly to the INT32 accumulator. Matrix multiplications and convolutions run in low precision (INT4/INT8) with INT32 accumulation, and every requantization multiplier is converted to dyadic form ( $b/2^c$ ). Weights and activations within a layer always use the same bit-width, so no casting between integer precisions is needed. The first and last layers are always kept at 8-bit. Accuracy is recovered with quantization-aware fine-tuning (PyTorch 1.6, learning rate 1e-4, weight decay 1e-4, batch size 128), optionally with distillation. (Official HAWQ GitHub: [zhen-dong/hawq](https://github.com/zhen-dong/hawq))

5. **bit budgeting pipeline**: Each layer's sensitivity is precomputed for each bit choice (only $B \cdot L$ computations), assuming layer perturbations are independent and additive ( $\Omega = \sum_i \Omega_i$ ). The ILP then minimizes $\sum_i \Omega_i^{(b_i)}$ subject to any combination of limits on total model size, total BOPS, and total measured latency. Constraints are optional, and the paper's experiments apply one at a time. The paper solves the ILP with PuLP; any other ILP solver (e.g., Gurobi) could be substituted. The result is optimal within the independence assumption.

6. **evaluation/metrics used:**
    * Top-1 Classification Accuracy (%)
    * Model Memory Size (MB)
    * BitOps (GBOPS = billions of bit operations; per layer, $b_w \cdot b_a \cdot \mathrm{MACs}$ )
    * Real Hardware Latency (ms/image) and Speedup vs. uniform INT8 on a T4 GPU via TVM (CUDA 10.2, Google Cloud Platform)
    * TVM and PyTorch outputs verified to match layer by layer to machine precision, including the final Top-1 accuracy

7. **[Results: ](https://drive.google.com/drive/folders/1FFBPrqUA9TktnjNikrIZcRLiyMFb2oCT?usp=sharing)**

   - **Uniform quantization (Table 1):**
     - **ResNet-18:** INT8 **71.56%** (FP32 71.47%, 11.1MB, 116 GBOPS); INT4 68.45% (5.8MB, 34 GBOPS).
     - **ResNet-50:** INT8 **77.58%** (FP32 77.72%, 24.5MB, 247 GBOPS), 2.68 percentage points above prior integer-only work (Jacob et al., 74.90%, from a weaker 76.40% FP32 baseline); INT4 74.24% (13.1MB, 67 GBOPS).
     - **Inception-V3:** INT8 **78.76%** (FP32 78.88%, 22.7MB, 366 GBOPS), 4.56 percentage points above Jacob et al. (74.20%, from a 78.30% FP32 baseline); INT4 70.39% (12.3MB, 92 GBOPS).
     - To the authors' knowledge, these are the first integer-only INT4 results reported. At W4A4 they also exceed CalibTIB, a post-training method that uses FP32 casting (ResNet-18: 68.45% vs 67.50%; ResNet-50: 74.24% vs 73.70%).

   - **Mixed-precision INT4/8 (Table 2):** the ILP picks 4 or 8 bits per layer under a model size, BOPS, or latency limit, each at High/Medium/Low levels. Weights and activations of a layer share one bit-width (W4/8A4/8). Speed is measured on a T4 GPU via TVM, relative to uniform INT8 (0.40 ms/image for ResNet-18, 1.06 ms/image for ResNet-50). Top-1 is shown without / with distillation. Table 2's size and BOPS for the uniform baselines differ slightly from Table 1; both are reproduced as reported. The mixed-precision entries in Table 1 correspond to the BOPS-Medium rows below.

     **ResNet-18**

     | Constraint | Level | Size (MB) | BOPS (G) | Speed | Top-1 (%) |
     |---|---|---|---|---|---|
     | Uniform INT8 | – | 11.2 | 114 | 1.00× | 71.56 |
     | Size | High | 9.9 | 103 | 1.03× | 71.20 / 71.59 |
     | Size | Medium | 7.9 | 98 | 1.06× | 70.50 / 71.09 |
     | Size | Low | 7.3 | 95 | 1.08× | 70.01 / 70.66 |
     | BOPS | High | 8.7 | 92 | 1.12× | 70.40 / 71.05 |
     | BOPS | Medium | 6.7 | 72 | 1.21× | 70.22 / 70.38 |
     | BOPS | Low | 6.1 | 54 | 1.35× | 68.72 / 69.72 |
     | Latency | High | 8.7 | 92 | 1.12× | 70.40 / 71.05 |
     | Latency | Medium | 7.2 | 76 | 1.19× | 70.34 / 70.55 |
     | Latency | Low | 6.1 | 54 | 1.35× | 68.56 / 69.72 |
     | Uniform INT4 | – | 5.6 | 28 | 1.48× | 68.45 |

     **ResNet-50**

     | Constraint | Level | Size (MB) | BOPS (G) | Speed | Top-1 (%) |
     |---|---|---|---|---|---|
     | Uniform INT8 | – | 24.5 | 247 | 1.00× | 77.58 |
     | Size | High | 21.3 | 226 | 1.09× | 77.38 / 77.58 |
     | Size | Medium | 19.0 | 197 | 1.13× | 75.95 / 76.96 |
     | Size | Low | 16.0 | 168 | 1.18× | 74.89 / 76.51 |
     | BOPS | High | 22.0 | 197 | 1.16× | 76.10 / 76.76 |
     | BOPS | Medium | 18.7 | 154 | 1.23× | 75.39 / 76.73 |
     | BOPS | Low | 16.7 | 110 | 1.30× | 74.45 / 76.03 |
     | Latency | High | 22.3 | 199 | 1.13× | 76.63 / 76.97 |
     | Latency | Medium | 18.5 | 155 | 1.21× | 74.95 / 76.39 |
     | Latency | Low | 16.5 | 114 | 1.28× | 74.26 / 76.19 |
     | Uniform INT4 | – | 13.1 | 67 | 1.45× | 74.24 |

     **Inception-V3** (Table 1; no latency reported): mixed INT4/8 reached **74.65%**, or 74.72% with distillation (19.6MB, 265 GBOPS), between uniform INT8 (78.76%, 22.7MB, 366 GBOPS) and uniform INT4 (70.39%, 12.3MB, 92 GBOPS).

   - **Takeaways:**
     - Mixed precision fills the gap between uniform INT8 and INT4. For example, ResNet-18 capped at 7.9MB (roughly midway between INT4 and INT8) keeps 71.09% with distillation, close to INT8, while running 1.06× faster.
     - Model size and BOPS are only weakly correlated, and model size doesn't directly track accuracy. On ResNet-50, Size-High (21.3MB) scores 77.58% with distillation, higher than BOPS-High (22.0MB, 76.76%).
     - Distillation helps mixed precision the most; the authors report little to no improvement for uniform INT8 or INT4. Their highlighted example is +1.34 percentage points for ResNet-50 BOPS-Medium (the Table 1 configuration), and the gain in Table 2 reaches +1.93 for ResNet-50 Latency-Low. Gains are largest at tight budgets, where more layers are at 4 bits.
     - **Note for comparison with our notebook:** all INT4 and mixed-precision numbers above come from quantization-aware fine-tuning with a fully integer-only pipeline (BN folded into the convolutions, integer-only residual and concatenation layers). Our notebook uses post-training fake quantization with no fine-tuning, so the two are not directly comparable. This is why our uniform INT4 collapses while the paper's reaches 68–74%.
  
## ⛰︎ impact
HAWQ-V3 showed that neural networks can run with integer-only arithmetic (integer multiplication, integer addition, and bit shifting) and no floating-point fallback, including for batch normalization, residual connections, and concatenation. By combining Hessian-based sensitivity with a hardware-aware Integer Linear Programming solver, it showed that mixed-precision bit allocation can be solved in under a second once sensitivities are computed (under 30 minutes of Hessian trace computation for ResNet-50/Inception-V3 on 4 GPUs). The quantized models were deployed on real hardware (T4 GPUs via TVM), achieving measured speedups over INT8 of up to 1.48× for uniform INT4 and up to 1.35× for mixed INT4/8.

#### future work
HAWQ-V3 sets the gold standard for integer-only execution and instant ILP bit-solving. However, its main flaw remains data dependency, in three places: (1) computing Hessian traces ( $\mathrm{Tr}(H_i)$ ) requires passing real images through the model with repeated backpropagation (under 30 minutes on 4 RTX 6000 GPUs for ResNet-50/Inception-V3); (2) accuracy is recovered through quantization-aware fine-tuning (plus optional distillation from a ResNet-101 teacher); and (3) static activation ranges are tracked on real data during that fine-tuning. In RQNKUP, we can adopt HAWQ-V3's dyadic integer engine and ILP solver, but replace its backprop-heavy Hessian Trace calculation with our data-free weight metrics (like excess kurtosis for outliers and spectral rank for structure). This allows RQNKUP to select integer-only mixed-precision bit-widths with zero data, zero backprop passes, and zero privacy risks. To be fully data-free end-to-end, RQNKUP also needs a data-free way to set activation scales and must work without quantization-aware fine-tuning. One option is ZeroQ-style synthetic calibration data, though generating it requires some backprop through the model. As a result, HAWQ-V3's fine-tuned accuracies are an upper-bound reference rather than a direct comparison.

## **additional sources:**
* [HAWQ-V3 Paper (ICML 2021)](https://arxiv.org/abs/2011.10680)
* [HAWQ-V2 Paper (NeurIPS 2020), source of the sensitivity metric](https://arxiv.org/abs/1911.03852)
* [PyHessian Paper, used for Hessian trace computation](https://arxiv.org/abs/1912.07145)
* [Official HAWQ GitHub Repository (zhen-dong/hawq)](https://github.com/zhen-dong/hawq)
* [ZeroQ GitHub Repository (amirgholami/zeroq)](https://github.com/amirgholami/zeroq)
* [PuLP Linear Programming Python Library](https://coin-or.github.io/pulp)
