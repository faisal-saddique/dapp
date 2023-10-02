import utils
import openai
from dotenv import load_dotenv
import os
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from utilities.sidebar import sidebar
from streaming import StreamHandler
from utilities.utils import load_existing_index_pinecone

from langchain.callbacks.base import BaseCallbackHandler

# Import required libraries for different functionalities
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

import pinecone

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENVIRONMENT"))

st.set_page_config(
    page_title='Directions Bot',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title("Directions Bot 🤖")



if "session_chat_history" not in st.session_state:
    st.session_state.session_chat_history = []

if "Knowledgebase" not in st.session_state:
    st.session_state["Knowledgebase"] = load_existing_index_pinecone()

embeddings = OpenAIEmbeddings()

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

class CustomDataChatbot:

    def __init__(self):
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def create_qa_chain(self):

        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, politely say that you don't know, don't try to make up an answer. Always end your answer by asking the user if he needs more help.

        ---------------------------
        {context}
        ---------------------------

        Question: {question}
        Friendly Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = {"prompt": PROMPT}

        # vectorstore = st.session_state.Knowledgebase
        
        # load to your BM25Encoder object
        bm25_encoder = BM25Encoder().load("bm25_values_for_ddw.json")
        index = pinecone.Index(os.getenv("PINECONE_INDEX"))

        retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings, sparse_encoder=bm25_encoder, index=index, top_k = 5
        )

        llm = ChatOpenAI(temperature=0)
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=retriever, llm=llm
        )

        # from langchain.retrievers import ContextualCompressionRetriever
        # from langchain.retrievers.document_compressors import LLMChainExtractor

        # llm = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo-16k")
        # compressor = LLMChainExtractor.from_llm(llm)
        # compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever_from_llm)

        return RetrievalQA.from_chain_type(llm=ChatOpenAI(streaming=True,model_name="gpt-3.5-turbo-16k"), chain_type="stuff", retriever=retriever_from_llm, return_source_documents=True,chain_type_kwargs=chain_type_kwargs)
        

    @utils.enable_chat_history
    def main(self):

        user_query = st.chat_input(placeholder="Ask me anything!")

        if user_query:

            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant", avatar="https://e7.pngegg.com/pngimages/139/563/png-clipart-virtual-assistant-computer-icons-business-assistant-face-service-thumbnail.png"):
                retrieval_handler = PrintRetrievalHandler(st.container())
                st_callback = StreamHandler(st.empty())
                qa = self.create_qa_chain()
                result = qa({"query": user_query}, callbacks=[retrieval_handler,st_callback])
                with st.expander("See sources"):
                    for doc in result['source_documents']:
                        st.success(f"Filename: {doc.metadata['source']}")
                        st.info(f"\nPage Content: {doc.page_content}")
                        st.json(doc.metadata, expanded=False)

                st.session_state.messages.append(
                    {"role": "assistant", "content": result['result'], "matching_docs": result['source_documents']})
                
                st.session_state.session_chat_history.append(
                    (user_query, result["result"]))


if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()
    sidebar()