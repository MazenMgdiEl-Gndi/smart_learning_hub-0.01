import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Smart Learning Hub", layout="wide")

st.title("📚 Smart Learning Hub")

# Load data
data_path = "data/student_performance.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    st.subheader("📊 Data Overview")
    st.dataframe(df.head())
else:
    st.error("Dataset not found!")

st.subheader("📈 Project Status")
st.write("All ML pipelines are already trained via run_all.py")

st.subheader("🤖 Prediction Module (Placeholder)")
st.write("You can connect models from /models folder here later")

st.subheader("🧠 AI Module")
st.write("Ollama / LLM integration can be plugged in here")