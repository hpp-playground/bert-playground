from pyknp import Juman
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from copy import copy

model = BertModel.from_pretrained(
    'bert/Japanese_L-24_H-1024_A-16_E-30_BPE_WWM_transformers')
bert_tokenizer = BertTokenizer(
    "bert/Japanese_L-24_H-1024_A-16_E-30_BPE_WWM_transformers/vocab.txt",
    do_lower_case=False,
    do_basic_tokenize=False)

model.eval()

jumanpp = Juman()
text = "僕は友達とサッカーをすることが好きだ"
result = jumanpp.analysis(text)
tokenized_text = [mrph.midasi for mrph in result.mrph_list()]
# ['僕', 'は', '友達', 'と', 'サッカー', 'を', 'する', 'こと', 'が', '好きだ']


tokenized_text.insert(0, '[CLS]')
tokenized_text.append('[SEP]')

for masked_index in range(1, len(tokenized_text) - 1):
    tmp_tokenized_text = tokenized_text.copy()
    tmp_tokenized_text[masked_index] = '[MASK]'
    print(tmp_tokenized_text)
    # ['[CLS]', '僕', 'は', '友達', 'と', '[MASK]', 'を', 'する', 'こと', 'が', '好きだ', '[SEP]']
    tokens = bert_tokenizer.convert_tokens_to_ids(tmp_tokenized_text)
    tokens_tensor = torch.tensor([tokens])

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    _, predict_indexes = torch.topk(predictions[0, masked_index], k=5)
    predict_tokens = bert_tokenizer.convert_ids_to_tokens(
        predict_indexes.tolist())
    print(predict_tokens)
    # ['話', '仕事', 'キス', 'ゲーム', 'サッカー']
