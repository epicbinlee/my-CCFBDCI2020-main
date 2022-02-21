# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/11/26 -*-

import jieba
import codecs
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


# tokenizer = BertTokenizer.from_pretrained('../pretrained_model/wobert_base')


# noinspection PyProtectedMember
def demo(text, do_basic_tokenize=True, never_split=None):
    """
    1.加载模型
    2.分词
    3.转换成token
    """
    # 使用Bert分词启器加载自己预训练或者第三方预训练的模型，问题：hugging face中没有该模型，是否是自己训练的？
    bert_tokenizer = BertTokenizer.from_pretrained('../pretrained_model/wobert_base', do_basic_tokenize=do_basic_tokenize, never_split=never_split)
    # forward
    res = bert_tokenizer(text)
    input_ids = res["input_ids"]
    input_tokens = []
    for id in input_ids:
        # 转换成token
        token = bert_tokenizer._convert_id_to_token(id)
        input_tokens.append(token)
    print(input_tokens)


def tokenize_(text):
    tokenize_text = ' '.join(jieba.cut(text))
    return tokenize_text


def read_(file_in, file_out):
    """
    读取数据，将原始数据转换成jieba分词，写入到文件
    """
    with codecs.open(filename=file_in, mode='r', encoding='utf-8') as f_in:
        for ctx in f_in.readlines():
            # 使用jieba分词进行分词
            tokenize_ctx = tokenize_(text=ctx)
            with codecs.open(file_out, mode='a', encoding='utf-8') as f_out:
                f_out.write(tokenize_ctx)
    print('finished')


def _tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


# noinspection PyChainedComparisons
def _is_chinese_char(cp):
    f = (cp >= 0x4E00 and cp <= 0x9FFF) \
        or (cp >= 0x3400 and cp <= 0x4DBF) \
        or (cp >= 0x20000 and cp <= 0x2A6DF) \
        or (cp >= 0x2A700 and cp <= 0x2B73F) \
        or (cp >= 0x2B740 and cp <= 0x2B81F) \
        or (cp >= 0x2B820 and cp <= 0x2CEAF) \
        or (cp >= 0xF900 and cp <= 0xFAFF) \
        or (cp >= 0x2F800 and cp <= 0x2FA1F)
    return f


if __name__ == '__main__':
    # --------------
    # 测试用例：
    # 使用预训练的bert模型，转换成token
    # sentence = '我爱北京天安门。'
    # sentence = "我 爱 北京 天安门 ， 天安门 上 太阳升 。 人民 领袖 毛主席 ，指引 我们 向前进 。 "
    # demo(text=sentence, do_basic_tokenize=False)

    # --------------
    # 测试用例2：带空格的句子
    # sentence = "我 爱 北京 天安门 ， 天安门 上 太阳升 。 人民 领袖 毛主席 ，指引 我们 向前进 。 I love you."
    # print(f'sentence: {sentence}')
    # 测试方法3：
    # print(_tokenize_chinese_chars(text=sentence))

    # 读取测试到文件
    with codecs.open(filename=r'D:\CCFClassfication\all_dataset\test.txt', mode='r', encoding='utf-8') as f:
        for sentence in f.readlines():
            # 测试方法1：使用jieba分词进行切分
            tokenize_sentence = tokenize_(text=sentence)
            print(f'tokenize_sentence: {tokenize_sentence}')
            demo(text=sentence, do_basic_tokenize=False)
            print('==' * 15)
    # 测试方法2：三个步骤：1.加载模型 2.分词 3.转换成token
    # demo(text=sentence, do_basic_tokenize=False)

    # --------------
    # 测试用例：测试jieba分词函数+读数据函数
    # read_(file_in=r"D:\CCFClassfication\labeled_data.csv", file_out=r'D:\CCFClassfication\output/all_content_tokenized.txt')
    # 分词文件
    # read_(file_in='../data/train/labeled_data.txt', file_out='/labeled_data_tokenized.txt')
    # read_(file_in='../data/train/unlabeled_data.txt', file_out='../data/train/unlabeled_data_tokenized.txt')
    # read_(file_in='../data/test_data.txt', file_out='../data/test_data_tokenized.txt')
    # read_(file_in='../data/train/all_content.txt', file_out='../data/train/all_content_tokenized.txt')

    # --------------
    # 测试用例：测试jieba分词
    # test_sentence = '将设计融于人性，将家居带入悠闲自在的情境。'
    # print(' '.join(jieba.cut(sentence=test_sentence)))
