import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.vectorstores import Pinecone
# For generating embeddings with OpenAI's embedding model
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from pinecone_text.sparse import BM25Encoder
from langchain.retrievers import PineconeHybridSearchRetriever
from dotenv import load_dotenv  # For loading environment variables from .env file
import os

# Load environment variables from .env file
load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENVIRONMENT"))

st.set_page_config(
    page_title='Directions Bot',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title("Directions Bot 🤖")


@st.cache_resource(ttl="1h")
def configure_retriever():
    import boto3
    # Initialize the S3 client
    s3 = boto3.client('s3',
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))

    # Specify your S3 bucket name and the filename you want to retrieve from S3
    bucket_name = os.getenv("AWS_BUCKET_NAME")
    file_name = os.getenv("BM25_FILEPATH_S3")

    # Specify the local file path where you want to save the downloaded file
    local_file_path = os.getenv("BM25_FILEPATH_S3")

    # Download the file from S3
    try:
        s3.download_file(bucket_name, file_name, local_file_path)
        print(f"File '{file_name}' downloaded from S3 to '{local_file_path}'")
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
    
    # load to your BM25Encoder object
    bm25_encoder = BM25Encoder().load(local_file_path)
    index = pinecone.Index(os.getenv("PINECONE_INDEX"))
    embeddings = OpenAIEmbeddings()
    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings, sparse_encoder=bm25_encoder, index=index, top_k = 5
    )

    llm = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo-16k")
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm
    )

    # from langchain.retrievers import ContextualCompressionRetriever
    # from langchain.retrievers.document_compressors import LLMChainExtractor

    # llm = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo-16k")
    # compressor = LLMChainExtractor.from_llm(llm)
    # compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever_from_llm)

    return retriever_from_llm

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            # source = os.path.basename(doc.metadata["path"])
            self.status.write(f"**Document: {idx}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


retriever = configure_retriever()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True, input_key='question', output_key='answer')
# memory = ConversationSummaryBufferMemory(llm=llm, input_key='question', output_key='answer')
# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True,return_source_documents=True
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain(user_query, callbacks=[retrieval_handler, stream_handler])
        # st.error(response)