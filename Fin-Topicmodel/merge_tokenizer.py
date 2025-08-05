import jieba
from transformers import AutoTokenizer
    
jieba.load_userdict("userdict.txt")
def chinese_tokenizer(text):
    return jieba.lcut(text)

Retriever_model_path = "valuesimplex-ai-lab/Fin-Retriever-base"    #这里使用带有领域适配分词器的模型地址
tokenizer = AutoTokenizer.from_pretrained(Retriever_model_path)

def bert_tokenizer(text,tokenizer=tokenizer):
    """使用BERT tokenizer对文本进行分词,返回分词列表"""
    tokens = tokenizer.tokenize(text)
    merged_tokens = []
    for token in tokens:
        token=token.replace(" ","")
        if token.startswith("##"):
            merged_tokens[-1] += token[2:]
        else:
            merged_tokens.append(token)
    return merged_tokens
    
def merge_tokenizers(text):

    tokenized_text1 = chinese_tokenizer(text)
    tokenized_text2 = bert_tokenizer(text)
    # tokenized_text2 = tokenize_with_NER(text)
    # 首先，重建原始文本（假设seg1和seg2是基于同一文本的正确分词）
    # 我们可以任选一个分词列表来重建原始文本
    text = ''.join(tokenized_text1)

    # 定义一个函数，将分词列表转换为包含起止位置的token列表
    def get_tokens_with_spans(segmentation):
        tokens = []
        index = 0
        for word in segmentation:
            start = index
            end = index + len(word)
            tokens.append({'word': word, 'start': start, 'end': end})
            index = end
        return tokens

    # 获取两个分词结果的tokens及其位置
    tokens1 = get_tokens_with_spans(tokenized_text1)
    tokens2 = get_tokens_with_spans(tokenized_text2)

    # 合并两个tokens列表
    all_tokens = tokens1 + tokens2

    # 根据start和end对所有tokens进行排序，方便后续处理
    all_tokens.sort(key=lambda x: (x['start'], -len(x['word'])))  # 长的词排在前面

    # 使用贪心算法进行合并
    merged_tokens = []
    index = 0
    text_length = len(text)
    while index < text_length:
        # 在所有tokens中，找到从当前index开始的所有词
        candidates = [token for token in all_tokens if token['start'] == index]
        if not candidates:
            # 如果没有候选词，说明无法匹配，跳过一个字符
            index += 1
            continue
        # 选择覆盖范围最长的词（即长度最大的词）
        best_token = candidates[0]

        # 检查其他分词中是否存在相同范围但词数更少的分词
        same_span_tokens = [token for token in candidates if token['end'] == best_token['end']]
        if same_span_tokens:
            # 取词数更少的分词
            best_token = min(same_span_tokens, key=lambda x: x['word'].count(' ') + 1)

        # 将选定的词加入结果
        merged_tokens.append(best_token['word'])
        # 更新索引
        index = best_token['end']

    return merged_tokens