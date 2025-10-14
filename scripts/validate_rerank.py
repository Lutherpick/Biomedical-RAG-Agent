# scripts/validate_rerank.py
import requests, json

Q = [
    "glioblastoma treatment with temozolomide",
    "preclinical mouse model of Parkinson's disease",
    "zebrafish toxicity study liver damage",
    "rat model neuropathic pain",
    "canine osteosarcoma chemotherapy",
    "porcine cardiac ischemia model",
    "rabbit corneal wound healing",
    "zebrafish developmental toxicity",
    "in vivo xenograft tumor growth",
    "mouse CRISPR knockout phenotype"
]

url = "http://127.0.0.1:8000/search"
ok = 0
for q in Q:
    r = requests.post(url, json={"query": q, "k": 5, "n_retrieve": 30})
    if r.status_code == 200 and "passages" in r.json():
        ok += 1
    print(q, "->", r.status_code, "hits:", len(r.json().get("passages", [])))
print(f"{ok}/{len(Q)} queries succeeded.")
