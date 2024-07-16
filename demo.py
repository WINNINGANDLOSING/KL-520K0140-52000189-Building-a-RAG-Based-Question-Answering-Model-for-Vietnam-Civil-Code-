import pandas as pd

df = pd.read_csv("qa_data.csv")
df = df.rename(columns={"query": "queries", "content": "relevance_docs", })
df.to_json("qa_data.json", index=False)
