# The Directions Bot

## Introduction

The Directions Bot is a powerful chatbot application designed to provide directions and information based on user queries. It utilizes AI and various data sources to offer intelligent responses to user questions.

## Features

- Interactive chat interface.
- Natural language understanding.
- Knowledgebase integration.
- Pinecone Index population.
- Pinecone Index statistics.
- Destructive operations for managing indices.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/faisal-saddique/dapp.git
   cd dapp
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on the provided `.env.template` and fill in the required API keys and environment variables.

## Usage

To run the Directions Bot, execute the following command:

```bash
streamlit run ChatBot.py
```

This will start the chatbot application. You can access it through the localhost URL: [http://localhost:8501](http://localhost:8501)

### Adding Documents to the Pinecone Index

To expand the knowledgebase and improve the capabilities of the Directions Bot, follow these steps to add more documents to the Pinecone Index:

1. **Prepare Your Documents**: Ensure that the documents you want to add to the index are in a compatible format, such as PDF, CSV, DOCX, XLSX, JSON, or TXT.

2. **Access the Pinecone Index Page**: Open the Pinecone Index page in the Directions Bot application. You can access it through the following URL when the bot is running: [http://localhost:8501/pinecone](http://localhost:8501/Populate_Pinecone_Index) 

3. **Upload Documents**: On the Pinecone Index page, you'll find an option to upload one or more documents. Click the "Upload" button.

4. **Select Document Files**: A file upload dialog will appear. Select the documents you want to add from your local file system. You can upload multiple files at once.

5. **Begin Indexing**: After selecting the documents, click the "Begin" button. The bot will start processing and indexing the uploaded documents.

6. **Monitor Progress**: You can monitor the progress of the indexing process on the Pinecone Index page. Depending on the number and size of the documents, it may take some time to complete.

7. **Indexing Completion**: Once the indexing process is complete, you will receive a confirmation message. The new documents are now part of the Pinecone Index and can be used for answering user queries.

8. **Interact with the Bot**: You can now interact with the Directions Bot using the newly added documents for enhanced knowledge and responses.

By following these steps, you can continuously expand the Pinecone Index with relevant documents to improve the bot's ability to answer user queries effectively.