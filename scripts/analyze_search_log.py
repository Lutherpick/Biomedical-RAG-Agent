# scripts/analyze_search_log.py
# pip install matplotlib
from pathlib import Path
import json
from collections import Counter
import matplotlib.pyplot as plt

LOG = Path("logs/search_log.jsonl")
if not LOG.exists():
    print("No log file found:", LOG)
    raise SystemExit(0)

docs = []
with LOG.open(encoding="utf-8") as f:
    for line in f:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        for h in rec.get("hits_sample", []):
            doc = h.get("doc_id")
            if doc:
                docs.append(doc)

cnt = Counter(docs)
top = cnt.most_common(10)

print("Top doc_ids by appearances in top hits:")
for doc, n in top:
    print(f"{doc}: {n}")

labels = [d for d,_ in top]
values = [n for _,n in top]
plt.figure()
plt.bar(labels, values)
plt.xticks(rotation=45, ha="right")
plt.title("Top doc_ids (from search_log samples)")
plt.tight_layout()
plt.show()
