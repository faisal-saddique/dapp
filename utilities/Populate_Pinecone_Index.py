import streamlit as st
from utilities.utils import (
    refined_docs,
    parse_docx,
    parse_readable_pdf,
    parse_xlsx,
    parse_csv,
    parse_json,
    parse_txt,
    num_tokens_from_string,
    add_vectors_to_pinecone
)

from dotenv import load_dotenv

from utilities.sidebar import sidebar

st.set_page_config(
    page_title='Populate Pinecone',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='expanded'
)

sidebar()

# Load environment variables from .env file
load_dotenv()

st.title("Populate Pinecone Index")

accepted_file_types = ["pdf", "csv", "docx", "xlsx", "json", "txt"]

uploaded_files = st.file_uploader("Upload one or more files", accept_multiple_files=True, type=accepted_file_types)

try:
    if st.button("Begin", use_container_width=True):
        if uploaded_files:
            docs = None
            csv_docs=None
            tot_len = 0

            for file in uploaded_files:
                file_extension = file.name.split(".")[-1].upper()
                st.write(f'File: {file.name}, Extension: {file_extension}')
                file_content = file.read()  # Read the content of the uploaded file
                
                if "files_for_download" not in st.session_state:
                    st.session_state.files_for_download = {}
                if "uploaded_files_history" not in st.session_state:
                    st.session_state.uploaded_files_history = {}
                st.session_state.uploaded_files_history[f"{file.name}"] = round(file.size / 1024, 2)
                st.session_state.files_for_download[f"{file.name}"] = file_content

                if file_extension == 'PDF':
                    if docs is None:
                        docs = parse_readable_pdf(file_content,filename=file.name)
                    else:
                        docs = docs + parse_readable_pdf(file_content,filename=file.name)

                elif file_extension == 'JSON':
                    if docs is None:
                        docs = parse_json(file_content,filename=file.name)
                    else:
                        docs = docs + parse_json(file_content,filename=file.name)

                elif file_extension == 'TXT':
                    if docs is None:
                        docs = parse_txt(file_content,filename=file.name)
                    else:
                        docs = docs + parse_txt(file_content,filename=file.name)

                elif file_extension == 'DOCX':
                    if docs is None:
                        docs = parse_docx(file_content,filename=file.name)
                    else:
                        docs = docs + parse_docx(file_content,filename=file.name)

                elif file_extension == 'XLSX':
                    if docs is None:
                        docs = parse_xlsx(file_content,filename=file.name)
                    else:
                        docs = docs + parse_xlsx(file_content,filename=file.name)
                
                elif file_extension == 'CSV':
                    
                    csv_docs = parse_csv(file_content,filename=file.name)
                else:
                    raise ValueError("File type not supported!")

            if docs:
                chunked_docs = refined_docs(docs)
                chunked_docs += csv_docs
            else:
                if csv_docs:
                    chunked_docs = csv_docs
                else:
                    raise Exception("Nothing to index.")
                
            no_of_tokens = num_tokens_from_string(chunked_docs)
            st.write(f"Number of tokens: \n{no_of_tokens}")

            if no_of_tokens:
                with st.spinner("Populating Pinecone..."):
                    add_vectors_to_pinecone(chunked_docs=chunked_docs)
                    st.success("Done! Please headover to chatbot to start interacting with your data.")
                    st.session_state["start_fresh"] = False
            else:
                st.error("No text found in the docs to index. Please make sure the documents you uploaded have a selectable text.")
                if "uploaded_files_history" in st.session_state:
                    st.session_state.uploaded_files_history = {}
                st.session_state["start_fresh"] = True
        else:
            st.error("Please add some files first!")
except Exception as e:
    st.error(f"An error occured while indexing your documents: {e}\n\nPlease fix the error and try again.")