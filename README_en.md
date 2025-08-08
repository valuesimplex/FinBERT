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

[中文](README.md)  |  [English](README_en.md) 


![projects](./pics/projects.svg)

🌟Paper🌟: https://dl.acm.org/doi/10.1145/3711896.3737219

🌟Datasets and Checkpoints🌟: https://huggingface.co/valuesimplex-ai-lab/


## Background

The open-source FinBERT2 is the second-generation upgrade of the FinBERT model (open-sourced in 2020) by EntropyReduce Technology. FinBERT2 is deeply pre-trained on high-quality Chinese financial corpus of over 32 billion tokens, aiming to improve the performance of Large Language Models (LLMs) in financial domain application deployment.

This open-source work includes the pre-trained model FinBERT2, fine-tuned models for specific downstream tasks, and related datasets to support more innovative research and application practices in the fintech field, and to jointly promote the prosperity of the financial AI ecosystem with community partners.

## FinBERT2 Introduction

![projects](./pics/overview.png)

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
import numpy as np

# Load the Fin-Retrievers model.
model = SentenceTransformer('valuesimplex-ai-lab/fin-retriever-base')

# Set the original query
query = "美联储加息对科技股的影响"

# Key step: Add a prompt to the query to ensure consistency with the model's training
optimized_query = "为这个句子生成表示以用于检索相关文章：" + query

# Build a document list (including structured information such as title, organization, author, etc.)
documents = [
    {
        "title": "美联储加息对科技股估值影响分析",
        "company": "",  
        "institution": "摩根士丹利",
        "industry": "科技综合",
        "author": "张伟",
        "type": "策略报告",
        "content": "2023年美联储连续加息导致科技股估值大幅回调，特别是高成长性科技公司。历史数据显示，利率每上升25个基点，科技板块平均下跌3.5%。FAANG股票受冲击最明显，其中Meta和Netflix估值压缩幅度最大。"
    },
    {
        "title": "美联储加息对全球资产配置的影响",
        "company": "",  
        "institution": "瑞银",
        "industry": "宏观经济",
        "author": "李强",
        "type": "宏观研究",
        "content": "本文系统分析美联储加息对全球股债汇市的影响，科技股作为权益资产的一部分，其波动在本文中被纳入整体市场分析框架。报告特别关注了新兴市场货币波动和商品价格变化，科技股仅占分析篇幅的15%。"
    },
    {
        "title": "中国科技股监管政策变化及投资机会",
        "company": "腾讯控股", 
        "institution": "中金公司",
        "industry": "互联网",
        "author": "陈明",
        "type": "行业研究",
        "content": "中国科技板块近期受监管政策放松驱动上涨，本文聚焦国内政策变化对互联网公司的影响，包括游戏版号发放、平台经济监管等。报告详细分析了腾讯、阿里巴巴和美团的监管风险变化，未讨论美联储货币政策影响。"
    },
    {
        "title": "加息周期中银行股的投资价值分析",
        "company": "招商银行", 
        "institution": "中信证券",
        "industry": "银行",
        "author": "赵静",
        "type": "行业深度",
        "content": "美联储加息显著改善银行净息差，本文测算国内上市银行净利息收入将提升8-12%。重点分析了招商银行、工商银行和中国银行的受益程度。科技股作为对比板块简要提及，指出其融资成本上升的压力。"
    },
    {
        "title": "生猪养殖行业周期反转信号分析",
        "company": "牧原股份",  
        "institution": "长江证券",
        "industry": "农林牧渔",
        "author": "刘洋",
        "type": "行业报告",
        "content": "能繁母猪存栏量连续三个月下降，预示猪周期即将反转。我们预计2024年生猪价格将上涨40%，牧原股份、温氏股份和新希望等龙头养殖企业盈利改善空间显著。报告未涉及科技股或货币政策。"
    },
    {
        "title": "光伏产业链价格走势及技术迭代分析",
        "company": "隆基绿能",  
        "institution": "国金证券",
        "industry": "电力设备",
        "author": "杨光",
        "type": "产业链研究",
        "content": "硅料价格降至100元/kg以下，刺激下游装机需求。TOPCon电池量产效率突破25.5%，隆基绿能、通威股份和晶科能源技术领先。报告分析光伏行业供需格局，与科技股和货币政策无直接关联。"
    },
    {
        "title": "美债收益率上升对科技股ETF的影响",
        "company": "", 
        "institution": "华泰证券",
        "industry": "金融产品",
        "author": "周涛",
        "type": "基金研究",
        "content": "分析10年期美债收益率上升对科技行业ETF(如XLK、QQQ)的影响机制。报告指出利率上升1%将导致科技ETF估值下降10-15%，但未讨论具体科技公司基本面变化。重点比较了不同ETF产品的久期和利率敏感性。"
    }
]

# Define document type labels (used for displaying positive and negative example tags)
doc_types = {
    0: "正例",
    1: "难负例",
    2: "难负例",
    3: "难负例",
    4: "随机负例",
    5: "随机负例",
    6: "难负例"
}

# Format the documents into structured text to enhance the model's understanding
formatted_docs = []
for doc in documents:
    text = (
        f"文章标题：{doc['title']}\n"
        f"公司名称：{doc['company']}\n"
        f"发布机构：{doc['institution']}\n"
        f"行业：{doc['industry']}\n"
        f"作者：{doc['author']}\n"
        f"{doc['type']}：{doc['content']}"
    )
    formatted_docs.append(text)

# Concatenate the query and all documents as a single input
all_texts = [optimized_query] + formatted_docs

# Generate sentence embeddings (the first one is the query embedding, followed by document embeddings)
embeddings = model.encode(all_texts)

# Split the query embedding and the document embeddings
query_vector = embeddings[0]
doc_vectors = embeddings[1:]

# Compute dot product similarity (can be used as an alternative to cosine similarity)
scores = query_vector @ doc_vectors.T

# Output the query content
print("【查询】:", query)
print("【文档匹配分数】:")

# Sort all documents by score (from highest to lowest)
sorted_indices = np.argsort(scores)[::-1]

# Output each document's score, label, and key information in sequence
for idx in sorted_indices:
    doc = documents[idx]
    print(f"分数: {scores[idx]:.4f} | 类型: {doc_types[idx]} | 标题: {doc['title']} | 机构: {doc['institution']},内容: {doc['content'][:60]}...")

#【查询】: 美联储加息对科技股的影响
# 【文档匹配分数】:
# 分数: 0.9015 | 类型: 正例 | 标题: 美联储加息对科技股估值影响分析 | 机构: 摩根士丹利 | 内容: 2023年美联储连续加息导致科技股估值大幅回调，特别是高成长...
# 分数: 0.8482 | 类型: 难负例 | 标题: 美联储加息对全球资产配置的影响 | 机构: 瑞银 | 内容: 本文系统分析美联储加息对全球股债汇市的影响，科技股作为权益资...
# 分数: 0.8167 | 类型: 难负例 | 标题: 美债收益率上升对科技股ETF的影响 | 机构: 华泰证券 | 内容: 分析10年期美债收益率上升对科技行业ETF(如XLK、QQQ...
# 分数: 0.7839 | 类型: 难负例 | 标题: 加息周期中银行股的投资价值分析 | 机构: 中信证券 | 内容: 美联储加息显著改善银行净息差，本文测算国内上市银行净利息收入...
# 分数: 0.7562 | 类型: 难负例 | 标题: 中国科技股监管政策变化及投资机会 | 机构: 中金公司 | 内容: 中国科技板块近期受监管政策放松驱动上涨，本文聚焦国内政策变化...
# 分数: 0.6164 | 类型: 随机负例 | 标题: 光伏产业链价格走势及技术迭代分析 | 机构: 国金证券 | 内容: 硅料价格降至100元/kg以下，刺激下游装机需求。TOPCo...
# 分数: 0.6151 | 类型: 随机负例 | 标题: 生猪养殖行业周期反转信号分析 | 机构: 长江证券 | 内容: 能繁母猪存栏量连续三个月下降，预示猪周期即将反转。我们预计2...
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
