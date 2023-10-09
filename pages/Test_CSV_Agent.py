import os
import streamlit as st
import openai
import pandas as pd
import streamlit as st
from langchain.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_types import AgentType
from typing import TextIO
import tempfile
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from langchain.chat_models import ChatOpenAI

load_dotenv()

def get_answer_csv(file: TextIO, query: str) -> str:
   
    """
    Returns the answer to the given query by querying a CSV file.

    Args:
    - file (str): the file path to the CSV file to query.
    - query (str): the question to ask the agent.

    Returns:
    - answer (str): the answer to the query from the CSV file.
    """
    file_content = file.read()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name 
    # Load the CSV file as a Pandas dataframe
    # df = pd.read_csv(file)
    #df = pd.read_csv("titanic.csv")


    df = pd.read_csv(temp_file_path)
    llm2 = ChatOpenAI()
    llm = OpenAI(api_token=os.getenv("OPENAI_API_KEY"))

    sdf = SmartDataframe(df, config={"llm": llm2,"verbose":True})
    answer = sdf.chat(query)


    return answer

st.header("CSV Agent Example")
uploaded_file = st.file_uploader("Upload a csv file", type=["csv"])

if uploaded_file is not None:
    query = st.text_area("Ask any question related to the document")
    button = st.button("Submit")
    if button:
        st.write(get_answer_csv(uploaded_file, query))