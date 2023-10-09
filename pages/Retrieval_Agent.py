import streamlit as st

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.document_loaders import Docx2txtLoader
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from langsmith import Client
from langchain.tools import BaseTool, StructuredTool, Tool, tool
import pandas as pd

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from langchain.agents import create_csv_agent
from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

load_dotenv()

client = Client()

st.set_page_config(
    page_title="ChatDDW",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

"# Chat DDW"

@st.cache_resource(ttl="1h")
def configure_retriever():
    loader = Docx2txtLoader("./new data/DDW FAQ.docx")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    )
    documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


tool_ddw_and_tickets_queries = create_retriever_tool(
    configure_retriever(),
    "answer_ddw_and_tickets_queries",
    "use this tool whenever you need to answer anything about DDW (Dutch Design Week) including sub-topics of queries related to tickets, buying, selling, availing discounts, invoice etc.",
)

class GetAnswerForAnythingYouDontKnowAbout(BaseTool):
    name="DDW_location_details_with_services"
    description="Use this tool whenever you need to answer anything about Places_Name, Location Photo, Google Maps link, Places_Latitude, Places_Longitude, Places_Address, Places_City, Places_PostalCode, Opening times, Services	Dogs allowed, Fully Wheelchair Accessible, Partially Wheelchair Accessible, Toilets available, Wheelchair Friendly Toilet, Wifi available"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
            df=pd.read_csv("./new data/DDW_Location_details_with_services_FINAL.csv"),
            return_intermediate_steps=True,
            # ["./new data/DDW_Location_details_with_services_FINAL.csv", "./new data/Participants_FINAL.csv", "./new data/Programme_details_with_Narratives_and_Discipline_FINAL.csv"],
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
        return agent.run(query)
        # return ans

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

csv_tool = Tool(
    name="DDW_location_details_with_services",
    description="Use this tool whenever you need to answer anything about Places_Name, Location Photo, Google Maps link, Places_Latitude, Places_Longitude, Places_Address, Places_City, Places_PostalCode, Opening times, Services	Dogs allowed, Fully Wheelchair Accessible, Partially Wheelchair Accessible, Toilets available, Wheelchair Friendly Toilet, Wifi available",
    func=create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
            df=pd.read_csv("./new data/DDW_Location_details_with_services_FINAL.csv"),
            return_intermediate_steps=True,
            # ["./new data/DDW_Location_details_with_services_FINAL.csv", "./new data/Participants_FINAL.csv", "./new data/Programme_details_with_Narratives_and_Discipline_FINAL.csv"],
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
)
tools = [tool_ddw_and_tickets_queries,csv_tool]

llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-3.5-turbo-16k")

message = SystemMessage(
    content=(
        "You are a helpful chatbot who is tasked with answering questions about Dutch Desing Week (DDW). "
        "Unless otherwise explicitly stated, it is probably fair to assume that questions are about Dutch Design Week (DDW). "
        "If there is any ambiguity, you probably assume they are about that."
    )
)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
)

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)

memory = AgentTokenBufferMemory(llm=llm)
starter_message = "Ask me anything about DDW!"
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]


def send_feedback(run_id, score):
    pass
    # client.create_feedback(run_id, "user_score", score=score)


for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    memory.chat_memory.add_message(msg)


if prompt := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor(
            {"input": prompt, "history": st.session_state.messages},
            callbacks=[st_callback],
            include_run_info=True,
        )
        st.session_state.messages.append(AIMessage(content=response["output"]))
        st.write(response["output"])
        memory.save_context({"input": prompt}, response)
        st.session_state["messages"] = memory.buffer
        run_id = response["__run"].run_id

        col_blank, col_text, col1, col2 = st.columns([10, 2, 1, 1])
        with col_text:
            st.text("Feedback:")

        with col1:
            st.button("👍", on_click=send_feedback, args=(run_id, 1))

        with col2:
            st.button("👎", on_click=send_feedback, args=(run_id, 0))