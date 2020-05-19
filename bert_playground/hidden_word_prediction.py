import torch
from copy import copy
from transformers import BertTokenizer, BertForMaskedLM


text = "How many lakes are there in Japan."

# ライブラリをimportする段階で、BertTokenizerも合わせて読み込んでいる
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_text = tokenizer.tokenize(text)

tokenized_text.insert(0, "[CLS]")
tokenized_text.append("[SEP]")
# ['[CLS]', 'how', 'many', 'lakes', 'are', 'there', 'in', 'japan', '.', '[SEP]']

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

for masked_index in range(1, len(tokenized_text) - 1):
    tmp_tokenized_text = tokenized_text.copy()
    print("[MASK] is", tmp_tokenized_text[masked_index])
    tmp_tokenized_text[masked_index] = '[MASK]'

    # テキストのままBERTに渡すのではなく、辞書で変換し、idになった状態にする。
    tokens = tokenizer.convert_tokens_to_ids(tmp_tokenized_text)
    tokens_tensor = torch.tensor([tokens])
    # BERTを読み込む。ここで少し時間がかかるかもしれない

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
    # masked_indexとなっている部分の単語の予測結果を取り出し、その予測結果top5を出す

    _, predict_indexes = torch.topk(predictions[0, masked_index], k=5)
    predict_tokens = tokenizer.convert_ids_to_tokens(predict_indexes.tolist())
    print(predict_tokens)
    # ['are', 'were', 'lie', 'out', 'is']
