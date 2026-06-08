"""
Prepare Chinese baike data for pretrain word embeddings
Original data see https://github.com/BIT-ENGD/baidu_baike
"""

import json
import zipfile
from collections import Counter
import jieba


input_path = "data/baike_2019/563w_baidubaike.json"
output_path = "data/baike_2019/563w_baidubaike_preprocessed.jsonl"
vocab_path = "data/baike_2019/vocab.json"
min_freq = 100
max_records = 3000000

counter = Counter()
cnt = 0
with open(output_path, "w", encoding="utf-8") as fo:
    for i,line  in enumerate(open(input_path, "r", encoding="utf-8")):
        data = json.loads(line)

        text = ""
        try:
            text = text + data.get("title") + "\n"
            for section in data.get("sections", []):
                text = text + section.get("title", "") + "\n"
                text = text + section.get("content", "") + "\n"
        except:
            pass
        text = text.strip()
        if len(text) == 0:
            continue

        words = list(jieba.cut(text))
        cnt += len(words)
        fo.write(json.dumps({"content":words}, ensure_ascii=False) + "\n")
        counter.update(words)
        if (i + 1) % 10000 == 0:
            print(f"processed {i+1} lines, total {cnt} words")
        
        if i >= 3000000:
            break
           
vocab = []
for word in counter:
    if counter[word] >= min_freq:
        vocab.append(word)
with open(vocab_path, "w", encoding="utf-8") as fo:
    json.dump(vocab, fo, ensure_ascii=False, indent=4)
