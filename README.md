
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

This will start the chatbot application. You can access it through the localhost URL: http://localhost:8501