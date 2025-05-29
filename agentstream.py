import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents import create_csv_agent
from langchain_openai import OpenAI
from openai import OpenAI as RawOpenAI  # native SDK

# --- SETUP ---
st.set_page_config(page_title="AI Agent Strategy & Planning", layout="wide")

# Logo and title
st.image("https://images.squarespace-cdn.com/content/v1/66082d505fe6d9359fc9cbfa/7fcf823b-3d29-479d-99c3-43a83007c4de/Frame+72.png?format=2500w", width=250)
st.title("🧠 AI Agent Strategy & Planning")

# --- LOAD ENV VARIABLES ---
load_dotenv()

# --- API KEY SETUP ---
user_api_key = st.text_input("🔐 Enter your OpenAI API key", type="password")

if not user_api_key:
    # Try to load from environment
    user_api_key = os.getenv("OPENAI_API_KEY")

if not user_api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = user_api_key
raw_openai = RawOpenAI(api_key=user_api_key)

# --- CSV UPLOAD ---
uploaded_file = st.file_uploader("📂 Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.to_csv("temp.csv", index=False)

    # LangChain agent with fixed temperature
    llm = OpenAI(temperature=0)
    agent = create_csv_agent(
        llm,
        "temp.csv",
        verbose=False,
        allow_dangerous_code=True
    )

    st.success("✅ File loaded successfully. Ask any question about your data.")

    user_question = st.text_input("❓ What would you like to know?")

    if user_question:
        with st.spinner("Thinking..."):
            try:
                # Get raw response from agent
                raw_answer = agent.invoke(user_question)

                # Rephrase answer using GPT
                completion = raw_openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that makes data answers sound friendly and natural."},
                        {"role": "user", "content": f"Question: {user_question}\nAnswer: {raw_answer}\nCan you rephrase the answer in a natural, conversational way?"}
                    ],
                    temperature=0.7
                )
                friendly_answer = completion.choices[0].message.content

                st.markdown("🗨️ **Here's what I found:**")
                st.markdown(f"> {friendly_answer}")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
else:
    st.info("⬆️ Upload a CSV file to begin.")
