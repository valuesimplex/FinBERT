<h1 align="center">⚡️FinBERT2: 弥合金融领域LLMs部署差距的专业双向编码器 </h1>
<p align="center">
    <a href="https://huggingface.co/valuesimplex-ai-lab/">
        <img alt="Build" src="https://img.shields.io/badge/FinBERT2--Suits-🤗-yellow">
    </a>
    <a href="https://github.com/
valuesimplex/FinBERT2">
            <img alt="Build" src="https://img.shields.io/badge/Contribution-Welcome-blue">
    </a>
    <a href="https://github.com/
valuesimplex/FinBERT2/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>

<h4 align="center">
    <p>
        <a href="#背景">背景</a> |
        <a href="#安装">安装</a> |
        <a href="#模型列表">模型列表</a> |
        <a href="#Reference">Reference</a> |
        <a href="#citation">Citation</a> |
        <a href="#license">License</a> 
    <p>
</h4>

[FinBERT2-中文](README.md)  |  [FinBERT2-English](https://github.com/valuesimplex/FinBERT2/blob/main/README_en.md) |  [FinBERT1](https://github.com/valuesimplex/FinBERT/blob/main/FinBERT1_README.md)


![projects](./pics/projects.svg)

🌟Paper🌟: https://dl.acm.org/doi/10.1145/3711896.3737219

🌟Datasets and Checkpoints🌟: https://huggingface.co/valuesimplex-ai-lab/


## 背景

本次开源的 FinBERT2 是熵简科技开源模型 FinBERT （于 2020年开源）的第二代升级模型。FinBERT2 在 320亿+ Token 的高质量中文金融语料进行深度预训练，旨在提升大语言模型（LLMs）在金融领域应用部署中的表现。

本次开源工作包含了预训练模型 FinBERT2、应用于特定下游任务的微调模型及相关数据集，以支持更多金融科技领域的创新研究与应用实践，并与社区伙伴共同推动金融AI生态的繁荣发展。

## FinBERT2简介

![projects](./pics/overview.png)

FinBERT2 可以通过以下方面弥合LLM在金融特定场景部署方面的差距：

1. **大规模中文金融语料预训练**：本次 FinBERT2 的金融语料预训练规模超过 320亿 Token，后续我们将对于这一模型开源。据我们所知，在开源的中文金融领域 BERT 类模型中，这将是预训练语料规模最大、性能表现最好的模型；

2. **优越的金融文本分类性能**：FinBERT2 在各类金融分类任务上，平均表现优于其他（Fin）BERT变体0.4%-3.3%，并领先主流大语言模型（如GPT-4-turbo, Claude 3.5 Sonnet, Qwen2）9.7%-12.3%；

3. **卓越的向量化信息检索（Fin-Retrievers）**：作为RAG系统的检索组件，FinBERT2 在性能上超越了开源和商业的向量化模型。在五个典型金融检索任务上，FinBERT2 相较于 BGE-base-zh平均性能提升了+6.8%，相较于OpenAI的text-embedding-3-large平均性能提升了+4.2%。

其他更详细的介绍及评测结果请参考我们的原始论文。
 
# 安装
### 1. clone项目源码
```
git clone https://github.com/valuesimplex/FinBERT2.git
```
### 2. 创建虚拟环境
```bash
conda create --name FinBERT2 python=3.11
conda activate FinBERT2
```

### 3. 安装依赖

运行以下命令安装 `requirements.txt` 中列出的所有依赖：
```bash
pip install -r requirements.txt
```

### 4. 运行项目
一旦安装完成，可以进入对应文件夹运行项目：
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
│   └── pretrain_wordpiece_tokenizer/              # * WordPiece tokenizer training
│       ├── spm.sh                                 # * SentencePiece training script
```


## 快速开始
### 对 FinBERT2 预训练模型进行增量预训练
```
cd FinBERT2\pretrain
sh run_mlm.sh
```
### FinBERT2 用于金融文本分类

```
# 微调
cd FinBERT2\Fin-labeler
sh runclassify.sh

# 预测
python sequence_inference.py
```

### 对比学习微调

参考[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder
)项目

### 使用 Fin-Retriever-base 向量检索模型

```
from sentence_transformers import SentenceTransformer

# 加载Fin-Retrievers模型
model = SentenceTransformer('valuesimplex-ai-lab/fin-retriever-base')

# 准备查询和文档
query = "科技股近期波动原因"
documents = [
    "美联储加息预期引发纳斯达克指数下跌",
    "半导体行业供应过剩导致股价回调",
    "银行板块受益于利率上升"
]

# 关键步骤：为查询添加检索前缀
optimized_query = '为这个句子生成表示以用于检索相关文章：' + query

# 组合所有文本（查询+文档）
all_texts = [optimized_query] + documents

# 生成所有向量
embeddings = model.encode(all_texts)

# 分离查询向量和文档向量
query_vector = embeddings[0]
doc_vectors = embeddings[1:]

# 计算相似度
scores = query_vector @ doc_vectors.T
print("文档匹配分数:", scores)
```



## Fin-Retrievers的评测基准FIR-bench

| 数据集 | 描述 |
|--|--|
| [FIR-Bench-Sin-Doc-FinQA](https://huggingface.co/datasets/valuesimplex-ai-lab/FIR-Bench-Sin-Doc-FinQA) |单文档问答数据集|
| [FIR-Bench-Multi-Docs-FinQA](https://huggingface.co/datasets/valuesimplex-ai-lab/FIR-Bench-Multi-Docs-FinQA) | 多文档问答数据集 |
| [FIR-Bench-Research-Reports-FinQA](https://huggingface.co/datasets/valuesimplex-ai-lab/FIR-Bench-Research-Reports-FinQA) | 研报问答数据集 |
| [FIR-Bench-Announcements-FinQA](https://huggingface.co/datasets/valuesimplex-ai-lab/FIR-Bench-Announcements-FinQA) | 公告问答数据集 |
| [FIR-Bench-Indicators-FinQA] | 指标类数据问答数据集（整理中） |

## 模型列表

| 模型 | 描述 |
|--|--|
| [**FinBERT2-base（预训练模型）**](https://huggingface.co/valuesimplex-ai-lab/FinBERT2-base) | 基于 320 亿中文金融文本 预训练的 BERT-base 语言模型，增加额外金融领域分词 |
| [**FinBERT2-large（预训练模型）**](https://huggingface.co/valuesimplex-ai-lab/FinBERT2-large) | 基于 320 亿中文金融文本 预训练的 BERT-large 语言模型，增加额外金融领域分词 |
| [**Fin-Retrievers-base（检索任务微调模型）**](https://huggingface.co/valuesimplex-ai-lab/Fin-Retriever-base) | 金融领域增强检索模型|


## 更新
- 2月/2025：创建 FinBERT2 github 项目
- 5月/2025：FinBET2 论文被 KDD 2025 ADS(applied data science) track 录用
- 6月/2025：更新FinBERT2 github项目

## Reference:
我们的套件基于下列开源项目开发，关于更多细节，可以参考原仓库：

- [**FinBERT1**](https://github.com/valuesimplex/FinBERT)：熵简科技第一代FinBERT
- [**RoBERTa中文预训练模型**](https://github.com/ymcui/Chinese-BERT-wwm)：一个采用RoBERTa方法在大规模中文语料上进行预训练的中文语言模型
- [**SentencePiece（分词工具）**](https://github.com/google/sentencepiece)：Google开发的无监督文本分词器,用于基于神经网络的文本生成任务
- [**BGE Embedding（通用嵌入模型）**](https://github.com/FlagOpen/FlagEmbedding)：是一个旨在开发检索和检索增强的语言模型的开源项目
- [**BERTopic**](https://github.com/MaartenGr/BERTopic)：利用BERT和类TF-IDF来创建可解释主题的先进主题模型



## Citation
‘
如果您觉得我们的工作有所帮助，请考虑点个星 :star: 和引用以下论文:
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
基于[MIT License](LICENSE)开源协议。
