<h1 align="center">⚡️FinBERT2: A Specialized Bidirectional Encoder for Bridging the Gap in Finance-Specific Deployment of LLMs</h1>
<p align="center">
    <a href="https://huggingface.co/valuesimplex-ai-lab/">
        <img alt="Build" src="https://img.shields.io/badge/FinBERT2--Suits-🤗-yellow">
    </a>
    <a href="https://github.com/valuesimplex/FinBERT2">
        <img alt="Build" src="https://img.shields.io/badge/Contribution-Welcome-blue">
    </a>
    <a href="https://github.com/valuesimplex/FinBERT2/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>

<h4 align="center">
    <p>
        <a href="#Background">Background</a> |
        <a href="#Installation">Installation</a> |
        <a href="#Model-List">Model List</a> |
        <a href="#Reference">Reference</a> |
        <a href="#Citation">Citation</a> |
        <a href="#License">License</a> 
    <p>
</h4>

[中文](README.md)  |  [English](https://github.com/valuesimplex/FinBERT2/blob/main/README_en.md) 


![projects](./imgs/projects.svg)

🌟Paper🌟: https://dl.acm.org/doi/10.1145/3711896.3737219

🌟Datasets and Checkpoints🌟: https://huggingface.co/valuesimplex-ai-lab/


## Background

The open-source FinBERT2 is the second-generation upgrade of the FinBERT model (open-sourced in 2020) by EntropyReduce Technology. FinBERT2 is deeply pre-trained on high-quality Chinese financial corpus of over 32 billion tokens, aiming to improve the performance of Large Language Models (LLMs) in financial domain application deployment.

This open-source work includes the pre-trained model FinBERT2, fine-tuned models for specific downstream tasks, and related datasets to support more innovative research and application practices in the fintech field, and to jointly promote the prosperity of the financial AI ecosystem with community partners.

## FinBERT2 Introduction

![projects](./imgs/overview.png)

FinBERT2 can bridge the gap in LLM deployment in finance-specific scenarios through the following aspects:

1. **Large-scale Chinese Financial Corpus Pre-training**: The financial corpus pre-training scale of FinBERT2 exceeds 32 billion tokens, and we will open-source this model subsequently. To our knowledge, among open-source Chinese financial domain BERT-like models, this will be the model with the largest pre-training corpus scale and best performance;

2. **Superior Financial Text Classification Performance**: FinBERT2 outperforms other (Fin)BERT variants by 0.4%-3.3% on average across various financial classification tasks, and leads mainstream large language models (such as GPT-4-turbo, Claude 3.5 Sonnet, Qwen2) by 9.7%-12.3%;

3. **Excellent Vectorized Information Retrieval (Fin-Retrievers)**: As a retrieval component of RAG systems, FinBERT2 surpasses open-source and commercial vectorization models in performance. On five typical financial retrieval tasks, FinBERT2 achieves an average performance improvement of +6.8% compared to BGE-base-zh and +4.2% compared to OpenAI's text-embedding-3-large.

For more detailed introductions and evaluation results, please refer to our original paper.
 
# Installation
### 1. Clone the project source code
```
git clone https://github.com/valuesimplex/FinBERT2.git
```
### 2. Create a virtual environment
```bash
conda create --name FinBERT2 python=3.11
conda activate FinBERT2
```

### 3. Install dependencies

Run the following command to install all dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Run the project
Once installation is complete, you can enter the corresponding folder to run the project:
####  File Structure

```
FinBERT2
├── Fin-labeler                                    # * Fine-tuning models for classification tasks
│   ├── SC_2/                                      # * Sentiment classification dataset (2-class)
│   ├── runclassify.sh                             # * Shell script for running classification
│   └── sequence_inference.py                      # * Sequence inference script
├── Fin-retriever                                  # * Contrastive learning-based retrieval models
│   ├── contrastive_finetune.sh                    # * Shell script for contrastive fine-tuning
│   └── finetune_traindata_sample.json             # * Sample training data for retrieval fine-tuning
├── Fin-Topicmodel                                 # * Topic modeling for financial titles
│   ├── Fin-Topicmodel.ipynb                       # * Main notebook for topic modeling
├── FinBERT2/                                      # * Core FinBERT2 model implementation
│   ├── pretrain/                                  # * Pre-training scripts and modules
│   │   ├── run_mlm.sh                             # * MLM training script
│   │   └── run_retromae.sh                        # * RetroMAE training script
│   └── pretrain_wordpiece_tokenizer/              # * WordPiece tokenizer training
│       ├── spm.sh                                 # * SentencePiece training script
```


## Quick Start
### Incremental pre-training on FinBERT2 pre-trained model
```
cd FinBERT2\pretrain
sh run_mlm.sh
```
### Using FinBERT2 for financial text classification

```
# Fine-tuning
cd FinBERT2\Fin-labeler
sh runclassify.sh

# Inference
python sequence_inference.py
```

### Contrastive learning fine-tuning

Refer to the [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder) project

### Using Fin-Retriever-base vector retrieval model

```
from sentence_transformers import SentenceTransformer

# Load Fin-Retrievers model
model = SentenceTransformer('valuesimplex-ai-lab/fin-retriever-base')

# Prepare query and documents
query = "科技股近期波动原因"
documents = [
    "美联储加息预期引发纳斯达克指数下跌",  
    "半导体行业供应过剩导致股价回调",     
    "银行板块受益于利率上升" 
]

# Key Step: Add a retrieval prefix to the query
optimized_query = '为这个句子生成表示以用于检索相关文章：' + query 

# Combine all texts (query + documents)
all_texts = [optimized_query] + documents

# Generate embeddings for all texts
embeddings = model.encode(all_texts)

# Separate query vector and document vectors
query_vector = embeddings[0]
doc_vectors = embeddings[1:]

# Calculate similarity scores
scores = query_vector @ doc_vectors.T
print("Document matching scores:", scores)
```



## FIR-bench Evaluation Benchmark for Fin-Retrievers

| Dataset | Description |
|--|--|
| [FIR-Bench-Sin-Doc-FinQA](https://huggingface.co/datasets/valuesimplex-ai-lab/FIR-Bench-Sin-Doc-FinQA) | Single-document Q&A dataset |
| [FIR-Bench-Multi-Docs-FinQA](https://huggingface.co/datasets/valuesimplex-ai-lab/FIR-Bench-Multi-Docs-FinQA) | Multi-document Q&A dataset |
| [FIR-Bench-Research-Reports-FinQA](https://huggingface.co/datasets/valuesimplex-ai-lab/FIR-Bench-Research-Reports-FinQA) | Research report Q&A dataset |
| [FIR-Bench-Announcements-FinQA](https://huggingface.co/datasets/valuesimplex-ai-lab/FIR-Bench-Announcements-FinQA) | Announcement Q&A dataset |
| [FIR-Bench-Indicators-FinQA] | Indicator Q&A dataset (In Preparation) |

## Model List

| Model | Description |
|--|--|
| [**FinBERT2-base (Pre-trained Model)**](https://huggingface.co/valuesimplex-ai-lab/FinBERT2-base) | BERT-base language model pre-trained on 32 billion Chinese financial texts with additional financial domain tokenization |
| [**FinBERT2-large (Pre-trained Model)**](https://huggingface.co/valuesimplex-ai-lab/FinBERT2-large) | BERT-large language model pre-trained on 32 billion Chinese financial texts with additional financial domain tokenization |
| [**Fin-Retrievers-base (Retrieval Task Fine-tuned Model)**](https://huggingface.co/valuesimplex-ai-lab/Fin-Retriever-base) | Financial domain enhanced retrieval model |


## Updates
- February/2025: Created FinBERT2 GitHub project
- May/2025: FinBERT2 paper accepted by KDD 2025 ADS (Applied Data Science) track
- June/2025: Updated FinBERT2 GitHub project

## Reference:
Our suite is developed based on the following open-source projects. For more details, please refer to the original repositories:

- [**FinBERT1**](https://github.com/valuesimplex/FinBERT): The first-generation FinBERT by EntropyReduce Technology
- [**RoBERTa Chinese Pre-trained Model**](https://github.com/ymcui/Chinese-BERT-wwm): A Chinese language model pre-trained on large-scale Chinese corpus using RoBERTa method
- [**SentencePiece (Tokenization Tool)**](https://github.com/google/sentencepiece): An unsupervised text tokenizer developed by Google for neural network-based text generation tasks
- [**BGE Embedding (General Embedding Model)**](https://github.com/FlagOpen/FlagEmbedding): An open-source project aimed at developing retrieval and retrieval-augmented language models
- [**BERTopic**](https://github.com/MaartenGr/BERTopic): An advanced topic model that leverages BERT and class-based TF-IDF to create interpretable topics



## Citation

If you find our work helpful, please consider giving us a star :star: and citing the following paper:
```
@inproceedings{xu2025finbert2,
  author = {Xu Xuan and Wen Fufang and Chu Beilin and Fu Zhibing and Lin Qinhong and Liu Jiaqi and Fei Binjie and Li Yu and Zhou Linna and Yang Zhongliang},
  title = {FinBERT2: A Specialized Bidirectional Encoder for Bridging the Gap in Finance-Specific Deployment of Large Language Models},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '25)},
  year = {2025},
  doi = {10.1145/3711896.3737219},
  url = {https://doi.org/10.1145/3711896.3737219}
}
```
## License
Licensed under the [MIT License](LICENSE).
