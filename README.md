# The Impact of Inference Acceleration on Bias of LLMs

This repository provides the code for the paper ["The Impact of Inference Acceleration on Bias of LLMs"](https://arxiv.org/pdf/2410.22118) [1].

We analyze how inference acceleration techniques, such as quantization and pruning, impact bias measurements across various benchmarks.


## Supported bias benchmarks:
The following benchmarks are supported for bias evaluation:

| Benchmark    | Paper | Dataset Source | File |
| --------     | ------- | ------- | ------- |
| CrowSPairs [2]  | [Link](https://arxiv.org/pdf/2010.00133) | [Dataset (csv)](https://github.com/nyu-mll/crows-pairs/blob/master/data/crows_pairs_anonymized.csv)    | `crowspairs_experiment.ipynb`| 
| DiscrimEval [3] | [Link](https://arxiv.org/pdf/2312.03689)     | [Dataset (hf)](https://huggingface.co/datasets/Anthropic/discrim-eval) | `discrimeval_experiment.ipynb`| 
| DiscrimEvalGen    | -    | Dataset (this repo) | `discrimevalgen_experiment.ipynb`| 
| DT-Stereotyping [4] | [Link](https://arxiv.org/pdf/2306.11698) | [Dataset (csv)](https://github.com/AI-secure/DecodingTrust/blob/main/data/stereotype/dataset/user_prompts.csv) | `dtstereotyping_experiment.ipynb`| 
| Global Opinion [5] | [Link](https://arxiv.org/pdf/2306.16388) | [Dataset (hf)](https://huggingface.co/datasets/Anthropic/llm_global_opinions) | `globalopinion_experiment.ipynb`| 
| WorldBench [6] | [Link](https://dl.acm.org/doi/pdf/10.1145/3630106.3658967) | [Dataset (csv)](https://github.com/mmoayeri/world-bench)  | `worldbench_experiment.ipynb`| 


## Supported inference acceleration strategies
We currently support the following acceleration techniques:
- **Quantization**
    - INT4, INT8 Quantization (BitsAndBytes)
    - KV-Cache Quantization (4-bit and 8-bit)
- **Pruning**
    - Wanda Pruning (Structured) [7]
    - Wanda Pruning (Unstructured) [7]

## Installation & Setup
### 1. Clone the Repository

    git clone https://github.com/aisoc-lab/inference-acceleration-bias.git
    cd inference-acceleration-bias


### 2. Install Dependencies
Using conda

    conda create --name biaseval python==3.8.20
    conda activate biaseval

Install requirements

     pip install -r requirements.txt
    
### 3. Download Datasets
* Refer to the dataset links in the table above and download them as needed.
* If the dataset source is **CSV**, manually download the file and specify the path in the corresponding notebook.
### 4. Run Experiments
* Open the Jupyter notebook for the benchmark you want to run (e.g., `crowspairs_experiment.ipynb`).
* Modify parameters for dataset loading, model configuration, and inference.
* Execute the notebook cells sequentially.




## Configuration

### Dataset Arguments
| Parameter    | Description |
| --------     | ------- |
| `shuffle`    | Whether to shuffle the dataset |
| `seed`    | Random seed for reproducibility |
| `num_samples`    | (Optional) Number of samples to load (for debugging); if unspecified, the full dataset is used |

### Model Arguments
| Parameter    | Description |
| --------     | ------- |
| `model_name`    | Model choice: "llama2chat", "llama3chat", "mistralchat" |
| `device`    | Device to run on: "cuda" |
| `sampling_method`    | Decoding strategy: "greedy" (no sampling) or "sampling" (temperature: 1.0, top_k: 0, top_p: 1) |
| `max_new_tokens`    | Maximum number of tokens to generate |
| `deployment`    | (Optional) dictionary specifying acceleration method; if unspecified, base model is used |

#### Deployment Parameters

| Key    | Values |
| --------     | ------- |
| `method`    | "quantization" or "pruning" |
| `type`    | "awq", "bitsandbytes", "kvcachequant" (for quantization) / "wanda_struct", "wanda_unstruct" (for pruning) |
| `nbits`    | "4" or "8" (INT4 vs INT8 (for quantization), 4-bit vs 8-bit KVCache quantization) |

###  Prompt Arguments
| Parameter    | Description |
| --------     | ------- |
| `use_chat_template`    | Whether to use the model provider's chat template |
| `system_message`    | System message to prepend (can be an empty string) |
| `answer_prefix`    | Prefix for the assistant's response (can be an empty string) |



### References

1. [The Impact of Inference Acceleration on Bias of LLMs](https://arxiv.org/pdf/2410.22118)

    Kirsten, E., Habernal, I., Nanda, V., & Zafar, M. B. (2024). The Impact of Inference Acceleration Strategies on Bias of LLMs. arXiv preprint arXiv:2410.22118.

2. Nangia, N., Vania, C., Bhalerao, R., & Bowman, S. R. (2020). CrowS-pairs: A challenge dataset for measuring social biases in masked language models. arXiv preprint arXiv:2010.00133.

3. Tamkin, A., Askell, A., Lovitt, L., Durmus, E., Joseph, N., Kravec, S., ... & Ganguli, D. (2023). Evaluating and mitigating discrimination in language model decisions. arXiv preprint arXiv:2312.03689.

4. Wang, B., Chen, W., Pei, H., Xie, C., Kang, M., Zhang, C., ... & Li, B. (2023, June). DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models. In NeurIPS.

5. Durmus, E., Nyugen, K., Liao, T. I., Schiefer, N., Askell, A., Bakhtin, A., ... & Ganguli, D. (2023). Towards measuring the representation of subjective global opinions in language models. arXiv preprint arXiv:2306.16388.

6. Moayeri, M., Tabassi, E., & Feizi, S. (2024, June). WorldBench: Quantifying Geographic Disparities in LLM Factual Recall. In The 2024 ACM Conference on Fairness, Accountability, and Transparency (pp. 1211-1228).

7. Sun, M., Liu, Z., Bair, A., & Kolter, J. Z. (2023). A simple and effective pruning approach for large language models. arXiv preprint arXiv:2306.11695.