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

load_dotenv()

client = Client()

st.set_page_config(
    page_title="ChatLangChain",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

"# Chat🦜🔗"

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

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from langchain.agents import create_csv_agent

agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
    ["./new data/DDW_Location_details_with_services_FINAL.csv", "./new data/Participants_FINAL.csv", "./new data/Programme_details_with_Narratives_and_Discipline_FINAL.csv"],
    verbose=True,
)

tool_csv_agent = Tool(
    func=agent.run,
    name="answer_everything_and_anything",
    description="useful for when you need to answer any question you are asked for which you don't know which tool you need to use"
)

#  = Tool.from_function(
#         func=agent.run,
#         name="answer_everything_and_anything",
#         description="useful for when you need to answer any question you are asked for which you don't know which tool you need to use"

#     ),

tools = [tool_ddw_and_tickets_queries,tool_csv_agent]

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
starter_message = "Ask me anything about langchain!"
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