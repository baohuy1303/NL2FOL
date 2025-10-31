import os, pandas as pd
inp = r"e:\Coding\Research\NL2FOL\test_data\wiki1.csv"
out_dir = r"e:\Coding\Research\NL2FOL\test_res"
out = os.path.join(out_dir, "wiki1_articles.csv")
os.makedirs(out_dir, exist_ok=True)
#read from /test_data/wiki1.csv
#write to /test_res/wiki1_articles.csv

df = pd.read_csv(inp)
if not {"title", "text"}.issubset(df.columns):
    raise ValueError("Input CSV must contain 'title' and 'text' columns")

# Keep title, turn text column into articles column
df_out = df[["title", "text"]].rename(columns={"text": "articles"})
df_out.to_csv(out, index=False)

print("Wrote:", out)