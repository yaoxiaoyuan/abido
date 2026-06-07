"""
Prepare Chinese wiki data for pretrain word embeddings
Original data see https://github.com/brightmart/nlp_chinese_corpus
"""

import json
import zipfile
from collections import Counter
import jieba

input_path = "data/wiki_zh_2019/wiki_zh_2019.zip"
output_path = "data/wiki_zh_2019/wiki_zh_2019_preprocessed.jsonl"
vocab_path = "data/wiki_zh_2019/vocab.json"
min_freq = 100

counter = Counter()
z = zipfile.ZipFile(input_path)
with open(output_path, "w", encoding="utf-8") as fo:
    for f in z.namelist():
        print(f"processing {f} now.")
        for line in z.open(f):
            data = json.loads(line)
            words = list(jieba.cut(data["text"]))
            fo.write(json.dumps({"content":words}, ensure_ascii=False) + "\n")
            counter.update(words)
           
vocab = []
for word in counter:
    if counter[word] >= min_freq:
        vocab.append(word)
with open(vocab_path, "w", encoding="utf-8") as fo:
    json.dump(vocab, fo, ensure_ascii=False, indent=4)