import os
from langchain.document_loaders import Docx2txtLoader
from typing import List
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd
import tempfile
import tiktoken
import re
import json
from langchain.docstore.document import Document
import unicodedata
import pinecone
import streamlit as st
from dotenv import load_dotenv

import os

# Load environment variables from .env file
load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENVIRONMENT"))

def load_existing_index_pinecone():
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(index_name=os.getenv("PINECONE_INDEX"), embedding=embeddings)
    return docsearch

def add_vectors_to_pinecone(chunked_docs):
    """Embeds a list of Documents and adds them to a Pinecone Knowledgebase"""
    
    # Embed the chunks
    embeddings = OpenAIEmbeddings()  # type: ignore

    # docsearch = Pinecone.from_existing_index(index_name=os.getenv("PINECONE_INDEX"), embedding=embeddings)
    # Knowledgebase = Pinecone.add_documents(chunked_docs,embeddings)
    Knowledgebase = Pinecone.from_documents(
        chunked_docs, embeddings, index_name=os.getenv("PINECONE_INDEX"))
    return Knowledgebase

def convert_filename_to_key(input_string):
    # Normalize string to decomposed form (separate characters and diacritics)
    normalized_string = unicodedata.normalize('NFKD', input_string)
    # Convert non-ASCII characters to ASCII
    ascii_string = normalized_string.encode('ascii', 'ignore').decode('utf-8')
    # Replace spaces, hyphens, and periods with _ underscores
    replaced_string = ascii_string.replace(' ', '_').replace('-', '_').replace('.', '_')
    return replaced_string

def get_all_filenames_and_their_extensions(source_folder):
    """  
    # Usage example
    source_folder = './raw_data/01 - Setor Imobiliario'

    files = get_all_filenames_and_their_extensions(source_folder)
    for file in files:
        print(f'File: {file[0]}, Extension: {file[1]}')
    """


    file_list = []  # Initialize an empty list to store the file information
    
    # Traverse through the source folder and its subfolders
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            file_path = os.path.abspath(os.path.join(root, file))
            file_name, file_extension = os.path.splitext(file_path)
            
            # Add the file information to the list as a tuple
            file_list.append((file_path, file_extension[1:].upper()))

    return file_list

def parse_txt(content,filename):
    # Assuming the content is in bytes format, save it temporarily
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name 
    loader = TextLoader(temp_file_path, autodetect_encoding=True)
    data = loader.load()
    for doc in data:
        doc.metadata["source"] = filename
    return data

def parse_docx(content,filename):
    # Assuming the content is in bytes format, save it temporarily
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name

    loader = Docx2txtLoader(file_path=temp_file_path)
    data = loader.load()
    # data = [re.sub(r"\n\s*\n", "\n\n", obj.page_content) for obj in data]
    for d in data:
        d.page_content = re.sub(r"\n\s*\n", "\n\n", d.page_content)
        d.metadata["source"] = filename
    return data

def parse_xlsx(content,filename):
    # Assuming the content is in bytes format, save it temporarily
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name

    # Read the Excel file
    df = pd.read_excel(temp_file_path)

    # Convert DataFrame to a CSV string
    csv_string = df.to_csv(index=False, encoding='utf-8')

    # Create a temporary file in memory
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        # Write the CSV string to the temporary file
        temp_file.write(csv_string.encode())
        temp_file.flush()

        # Step 2: Load the data using CSVLoader
        loader = CSVLoader(file_path=temp_file.name,encoding='utf-8')
        data = loader.load()
        for doc in data:
            doc.metadata["source"] = filename
            
    return data

def parse_csv(content,filename):
    # Create a temporary file in memory
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        # Write the CSV string to the temporary file
        temp_file.write(content)
        temp_file.flush()

        # Step 2: Load the data using CSVLoader
        loader = CSVLoader(file_path=temp_file.name,encoding='utf-8')
        data = loader.load()
        for doc in data:
            doc.metadata["source"] = filename
    return data

def refined_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800, # You can play around with this parameter to adjust the length of each chunk
        chunk_overlap  = 10,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function = len,
    )

    for doc in docs:
        doc.metadata["filename_key"] = convert_filename_to_key(doc.metadata["source"])

    print(f"Lenght of docs is {len(docs)}")
    return text_splitter.split_documents(docs)

def num_tokens_from_string(chunked_docs: List[Document]) -> int:

    string = ""
    print(f"Number of vectors: \n{len(chunked_docs)}")
    for doc in chunked_docs:
        string += doc.page_content

    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def parse_readable_pdf(content,filename):
    # Assuming the content is in bytes format, save it temporarily
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name  

    pdf_loader = PyMuPDFLoader(temp_file_path)
    pdf_data = pdf_loader.load()  # Load PDF file

    for doc in pdf_data:
        # Merge hyphenated words
        doc.page_content = re.sub(r"(\w+)-\n(\w+)", r"\1\2", doc.page_content)
        # Fix newlines in the middle of sentences
        doc.page_content = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", doc.page_content.strip())
        # Remove multiple newlines
        doc.page_content = re.sub(r"\n\s*\n", "\n\n", doc.page_content)

        doc.metadata["source"] = filename

    return pdf_data

# def parse_json(content, filename):
#     # Assuming the content is in bytes format, save it temporarily
#     with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
#         temp_file.write(content)
#         temp_file_path = temp_file.name

#     data_list = []
#     # Open and read the JSON file with 'utf-8' encoding
#     with open(temp_file_path, 'r', encoding='utf-8') as json_file:
#         for line in json_file:
#             try:
#                 data = json.loads(line)
#                 data_list.append(data)
#             except json.JSONDecodeError as e:
#                 st.error(f"Error decoding JSON: {str(e)}")

#     # Create a DataFrame from the list of dictionaries
#     df = pd.DataFrame(data_list)
#     res = json.loads(df.to_json(orient='records'))

#     if len(res):
#         docs = []
#         for i in res:
#             docs.append(Document(page_content=json.dumps(i),metadata={"source":filename}))
#         return docs
#     else:
#         return []

def parse_json(content, filename):
    # Assuming the content is in bytes format, save it temporarily
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    data = json.load(open(temp_file_path,'r'))
    df = pd.DataFrame(data)
    res = json.loads(df.to_json(orient='records'))
    if len(res):
        docs = []
        for i in res:
            docs.append(Document(page_content=json.dumps(i),metadata={"source":filename}))
    
        return docs
    else:
        return []
    
def remove_files_from_pinecone(path):
    
    pinecone.init(api_key=os.environ["PINECONE_API_KEY"],environment=os.environ["PINECONE_ENVIRONMENT"])

    index = pinecone.Index(os.environ["PINECONE_INDEX_NAME"])

    files = get_all_filenames_and_their_extensions(path)
    for file in files:
        filename_key = convert_filename_to_key(os.path.split(file[0])[-1])
        
        index.delete(
            filter={
                "filename_key": {"$eq": f"{filename_key}"},
            }
        )

        print(f"Removed file from Pinecone: {filename_key}")
        # index.describe_index_stats()