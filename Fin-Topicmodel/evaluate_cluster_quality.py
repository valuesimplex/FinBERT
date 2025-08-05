import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os
import pandas as pd
import ast

def calculate_topic_diversity(topic_keywords,top_n=10):
    """
    计算主题多样性指标（Topic Diversity，TD）。

    参数：
    - topic_keywords: dict，包含每个主题的关键词列表，格式为 {主题编号: [关键词列表]}。
    - top_n: int，取每个主题的前 N 个关键词，默认取前 20 个。

    返回：
    - td_value: float，主题多样性指标，取值范围 [0,1]。
    """
    top_n=10
    K = len(topic_keywords)
    top_words = []
    for topic_id in topic_keywords:
        words = topic_keywords[topic_id][:top_n]
        top_words.extend(words)

    # 计算独特词语的数量
    unique_words = set(top_words)
    unique_word_count = len(unique_words)

    # 总的词语数量
    total_word_count = K * top_n

    # 计算 TD 值
    td_value = unique_word_count / total_word_count

    print(f"主题数量（K）：{K}")
    print(f"每个主题取前 N 个关键词（N）：{top_n}")
    print(f"独特词语数量：{unique_word_count}")
    print(f"总的词语数量：{total_word_count}")
    print(f"主题多样性（Topic Diversity，TD）：{td_value:.4f}")

    return td_value


def evaluate_clustering(csv_filename, fused_embeddings, output_csv='clustering_evaluation_results.csv'):
    """
    评估聚类结果的函数，并将结果写入输出 CSV 文件。

    参数：
    - csv_filename: str，CSV 文件名，包含聚类标签和文档标题等信息。
    - fused_embeddings: numpy.ndarray，嵌入向量矩阵，形状为 (n_samples, n_features)。
    - output_csv: str，输出 CSV 文件名，用于保存评估结果，默认为 'clustering_evaluation_results.csv'。

    返回：
    - average_silhouette: float，平均轮廓系数（Silhouette Coefficient）。
    - ch_score: float，Calinski-Harabasz 指数。
    - db_score: float，Davies-Bouldin 指数。
    """

    # 1. 读取 CSV 文件并提取聚类标签和文档标题
    df = pd.read_csv(csv_filename)

    labels = df['Topic'].tolist()
    titles = df['Document'].tolist()

    # 2. 确保嵌入和聚类标签对应
    # 转换嵌入为 numpy 数组（如果不是）
    if not isinstance(fused_embeddings, np.ndarray):
        fused_embeddings = np.array(fused_embeddings)

    # 3. 检查嵌入向量和标签的数量是否一致
    assert fused_embeddings.shape[0] == len(labels), "嵌入向量数量与标签数量不一致！"

    # 4. 计算聚类评估指标

    # 4.1 轮廓系数
    average_silhouette = silhouette_score(fused_embeddings, labels)
    print(f"平均轮廓系数（Silhouette Coefficient）：{average_silhouette:.4f}")

    # 4.2 Calinski-Harabasz 指数
    ch_score = calinski_harabasz_score(fused_embeddings, labels)
    print(f"Calinski-Harabasz 指数：{ch_score:.4f}")

    # 4.3 Davies-Bouldin 指数
    db_score = davies_bouldin_score(fused_embeddings, labels)
    print(f"Davies-Bouldin 指数：{db_score:.4f}")


    # 4.4 topic_diversity_value

    # 提取 'Representation' 列并解析关键词列表,'Topic' 列为主题编号
    topic_keywords = {}
    for index, row in df.iterrows():
        topic_id = row['Topic']
        representation = row['Representation']
        # 将字符串解析为列表
        words = ast.literal_eval(representation)
        topic_keywords[topic_id] = words

    # 计算主题多样性
    td_value = calculate_topic_diversity(topic_keywords,top_n=10)
    print(f"topic_diversity_value 指数：{td_value:.4f}")
    # 5. 将结果写入输出 CSV 文件
    # 创建一个 DataFrame 来保存结果
    results_df = pd.DataFrame({
        'Input_File': [os.path.basename(csv_filename)],
        'Silhouette_Coefficient': [average_silhouette],
        'Calinski_Harabasz_Score': [ch_score],
        'Davies_Bouldin_Score': [db_score],
        'topic_diversity_value': [td_value]
    })

    # 检查输出文件是否存在，如果存在则追加，否则创建新文件
    if os.path.exists(output_csv):
        # 读取已有的结果文件
        existing_results = pd.read_csv(output_csv)
        # 将新的结果追加到已有的结果中
        all_results = pd.concat([existing_results, results_df], ignore_index=True)
    else:
        # 如果文件不存在，则使用新的结果
        all_results = results_df

    # 将结果保存到输出 CSV 文件
    all_results.to_csv(output_csv, index=False)
    print(f"评估结果已保存到文件：{output_csv}")

    return average_silhouette, ch_score, db_score,td_value
