# RQNKUP implementations | readme format template

**paper title:** AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

## ⚲ project summary
> AWQ is a post-training weight quantization method for large language models. It uses activation information to identify important weight channels and protects those channels during low-bit quantization.

## ⌖ motivation
> Large language models require a lot of memory, which makes them difficult to run on smaller devices. AWQ tries to reduce model size using low-bit weight quantization while keeping as much of the original model performance as possible.

## ✎ᝰ novelty
> AWQ identifies important weight channels using activation magnitude instead of only looking at the weights themselves. It then uses per-channel scaling to better preserve those important channels during quantization.

## ✰ methodology
1. **dataset(s) used**:
   - Pile validation data for calibration
   - WikiText-2 for perplexity evaluation

2. **model(s) used**:
   - OPT-125M was used for the initial sanity-check implementation and activation-stat collection
   - Llama-2-7B AWQ was used for the baseline evaluation
   - The Llama-2-7B checkpoint uses 4-bit AWQ weight quantization
   - The larger model was evaluated on a Google Colab T4 GPU

3. **sensitivity metric**:
   - AWQ uses activation magnitude to identify salient weight channels
   - I implemented a simplified activation-stat collection step and successfully collected statistics from 73 linear layers

4. **quantization pipeline**:
   - AWQ uses weight-only low-bit quantization, mainly INT3 and INT4 with group size 128
   - Important channels are identified using activation information and protected through per-channel scaling
   - The original MIT AWQ repository had dependency conflicts with the current Colab environment, so I adapted the implementation using a current Transformers + GPTQModel AWQ loading path
   - A Llama-2-7B AWQ INT4 checkpoint was successfully loaded and evaluated on GPU

5. **bit budgeting pipeline**:
   - AWQ itself does not use layer-wise mixed-precision bit allocation
   - It uses a uniform low-bit precision such as INT4 or INT3 across quantized weights
   - Activation-aware scaling protects important channels without keeping them at a different bit width
   - For RQNKUP, AWQ serves as the strong uniform LLM baseline that a future mixed-precision ranking method should match or outperform

7. **evaluation/metrics used**:
   - WikiText-2 perplexity
   - Lower perplexity indicates better language-model performance

8. **results**:
   - Pile calibration dataset loading completed
   - WikiText-2 evaluation dataset loading completed
   - OPT-125M forward pass completed successfully
   - Activation statistics collected from 73 linear modules
   - Llama-2-7B AWQ loaded successfully on CUDA
   - Measured WikiText-2 perplexity: **5.628**
   - Published AWQ INT4-g128 WikiText-2 perplexity: **5.60**
   - Published FP16 Llama-2-7B WikiText-2 perplexity: **5.47**
   - Estimated original parameter count: **6.738B parameters**
   - Approximate FP16 raw weight memory: **12.55 GiB**
   - Ideal INT4 raw weight memory: **3.14 GiB**
   - Theoretical raw weight-memory reduction: **75%**
   - Theoretical dense-compute estimate: **13.48 GFLOPs/token**
   - Theoretical FP16 BOP proxy: **1.73 trillion bit-ops/token**
   - Theoretical AWQ W4A16 BOP proxy: **0.43 trillion bit-ops/token**

## ⛰︎ impact
> AWQ makes low-bit LLM deployment more practical by reducing memory use while preserving much of the model’s performance.

#### future work
> Next, I want to compare AWQ's activation-based saliency with other sensitivity metrics and test whether those signals can be converted into a layer-level mixed-precision ranking for RQNKUP.

## **additional sources:**
> [AWQ paper](https://arxiv.org/abs/2306.00978)

> [Official AWQ implementation](https://github.com/mit-han-lab/llm-awq)

> [Pile calibration dataset](https://huggingface.co/datasets/mit-han-lab/pile-val-backup)

> [WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext)

> [Llama-2-7B AWQ checkpoint](https://huggingface.co/TheBloke/Llama-2-7B-AWQ)

> [GPTQModel](https://github.com/ModelCloud/GPTQModel)
