# run_nl2fol_from_articles.py
import os, re, pandas as pd, transformers, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nl_to_fol import NL2FOL

inp = r"e:\Coding\Research\NL2FOL\test_res\wiki1_articles.csv"
out = r"e:\Coding\Research\NL2FOL\test_res\Final_CacheDelete\BaseGPT5Mini.csv"
model_name = "gpt-5-mini"
nli_model_name = "roberta-large-mnli"

df = pd.read_csv(inp)

model_type, pipeline, tokenizer = "gpt", None, None
nli_tok = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

def split_into_sentences(text: str):
    # Simple sentence splitter on ., !, ? followed by whitespace/newline
    # Keeps punctuation attached to sentence.
    text = str(text).strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

agg = {}
for _, row in df.iterrows():
    title = str(row.get("title", ""))
    article = str(row.get("articles", ""))
    if title not in agg:
        agg[title] = {"FOL1": [], "FOL2": []}
    for sentence in split_into_sentences(article):
        n = NL2FOL(sentence, model_type, pipeline, tokenizer, nli_model, nli_tok, debug=False)
        f1, f2 = n.convert_to_first_order_logic()
        if isinstance(f1, str) and f1:
            agg[title]["FOL1"].append(f1)
        if isinstance(f2, str) and f2:
            agg[title]["FOL2"].append(f2)

out_rows = []
for title, fols in agg.items():
    fol1_text = " || ".join(fols["FOL1"]).strip()
    fol2_text = " || ".join(fols["FOL2"]).strip()
    out_rows.append({"title": title, "FOL1": fol1_text, "FOL2": fol2_text})

pd.DataFrame(out_rows, columns=["title", "FOL1", "FOL2"]).to_csv(out, index=False)
print("Wrote:", out)