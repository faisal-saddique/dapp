import streamlit as st
import pandas as pd
import json
import openai
import os
import re
import matplotlib.pyplot as plt
from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv

load_dotenv()

def csv_agent_func(file_path, user_message):
    """Run the CSV agent with the given file path and user message."""
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
        file_path, 
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    try:
        # Properly format the user's input and wrap it with the required "input" key
        tool_input = {
            "input": {
                "name": "python",
                "arguments": user_message
            }
        }
        
        response = agent.run(tool_input)
        return response
    except Exception as e:
        st.write(f"Error: {e}")
        return None

def display_content_from_json(json_response):
    """
    Display content to Streamlit based on the structure of the provided JSON.
    """
    
    # Check if the response has plain text.
    if "answer" in json_response:
        st.write(json_response["answer"])

    # Check if the response has a bar chart.
    if "bar" in json_response:
        data = json_response["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    # Check if the response has a table.
    if "table" in json_response:
        data = json_response["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

def extract_code_from_response(response):
    """Extracts Python code from a string response."""
    # Use a regex pattern to match content between triple backticks
    code_pattern = r"```python(.*?)```"
    match = re.search(code_pattern, response, re.DOTALL)
    
    if match:
        # Extract the matched code and strip any leading/trailing whitespaces
        return match.group(1).strip()
    return None

def csv_analyzer_app():
    """Main Streamlit application for CSV analysis."""

    st.title('CSV Assistant')
    st.write('Please upload your CSV file and enter your query below:')
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)
        
        # Create a directory to store the uploaded files if it doesn't exist
        upload_dir = "/tmp"
        # Save the uploaded file to disk
        os.makedirs(upload_dir, exist_ok=True)
        # Construct the full file path
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        df = pd.read_csv(file_path)
        st.dataframe(df)
        
        user_input = st.text_input("Your query")
        if st.button('Run'):
            response_main = csv_agent_func(file_path, user_input)
            
            # Extracting code from the response
            code_to_execute = extract_code_from_response(response_main)
            
            if code_to_execute:
                try:
                    print(f"got the code")
                    # Making df available for execution in the context
                    from contextlib import redirect_stdout
                    from io import StringIO
                    io_buffer = StringIO()
                    try:
                        with redirect_stdout(io_buffer):
                            ret = eval(code_to_execute, globals(), {"df": df, "plt": plt})
                            if ret is None: 
                                st.write(io_buffer.getvalue())
                            else:
                                st.write(ret)
                    except Exception:
                        print("inside exception")
                        with redirect_stdout(io_buffer):
                            exec(code_to_execute, globals(), {"df": df, "plt": plt})
                        print(f"hello {io_buffer.getvalue()}")
                        st.write(io_buffer.getvalue())
        
                    # ans = exec(code_to_execute, globals(), {"df": df, "plt": plt})

                    # from langchain.agents.agent_toolkits import create_python_agent
                    # from langchain.tools.python.tool import PythonREPLTool
                    # from langchain.python import PythonREPL
                    # from langchain.llms.openai import OpenAI
                    # from langchain.tools.python.tool import PythonAstREPLTool
                    # from langchain.agents.agent_types import AgentType
                    # from langchain.chat_models import ChatOpenAI

                    # agent_executor = create_python_agent(
                    #     llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
                    #     tool=PythonREPLTool(),
                    #     verbose=True,
                    #     agent_type=AgentType.OPENAI_FUNCTIONS,
                    #     agent_executor_kwargs={"handle_parsing_errors": True,"df": df, "plt": plt},
                    # )

                    # response = agent_executor.run(response_main)
                    # st.success(response)
                    fig = plt.gcf()  # Get current figure
                    st.pyplot(fig)  # Display using Streamlit
                except Exception as e:
                    st.write(f"Error executing code: {e}")
            else:
                st.write(response_main)

    st.divider()

csv_analyzer_app()