from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_experimental.sql import SQLDatabaseChain

from streaming import StreamHandler

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

load_dotenv()

# Update the database URI to point to your enhanced_database.db
db_uri = "sqlite:///database.db"
db = SQLDatabase.from_uri(db_uri)


st.title("DDW Event")

query = st.text_area("Enter your query here:")

placeholder = st.empty()
placeholder.write("*[Agent Chatter will appear here]*")
st_cb = StreamHandler(placeholder)
chat = ChatOpenAI(streaming=True, callbacks=[st_cb])

db_chain = SQLDatabaseChain.from_llm(chat, db, verbose=True)

tools = [
    Tool(
        name="DDW_event_details",
        func=db_chain.run,
        description="useful for when you need to answer anything about DDW (Dutch Design Week). All the queries related to Places_Name,Location Photo,Google Maps link,Places_Latitude,Places_Longitude,Places_Address,Places_City,Places_PostalCode,Opening times,Services,Dogs allowed,Fully Wheelchair Accessible,Partially Wheelchair Accessible,Toilets available,Wheelchair Friendly Toilet,Wifi available"
    )
]

agent = initialize_agent(tools, chat, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

if st.button("Get Insights"):
    response = agent.run(query)
    st.subheader("Response:")
    st.markdown(response)