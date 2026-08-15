import streamlit as st
import pandas as pd
import json

st.title("RAG Pipeline — Live Monitoring")

records = [json.loads(line) for line in open("query_log.jsonl")]
df = pd.DataFrame(records)
df["timestamp"] = pd.to_datetime(df["timestamp"])

col1, col2, col3 = st.columns(3)
col1.metric("Total Queries", len(df))
col2.metric("Total Cost", f"${df['cost'].sum():.4f}")
col3.metric("Avg Cost/Query", f"${df['cost'].mean():.5f}")

st.subheader("Cost over time")
st.line_chart(df.set_index("timestamp")["cost"])

st.subheader("Recent queries")
st.dataframe(df[["timestamp", "question", "cost"]].sort_values("timestamp", ascending=False).head(20))