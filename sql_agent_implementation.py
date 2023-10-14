from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

db_uri = "sqlite:///E:/DESKTOP/FreeLanceProjects/audio8/dapp/database.db"

db = SQLDatabase.from_uri(db_uri)
toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

agent_executor = create_sql_agent(
    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS
)

agent_executor.run("where is the toilet?")