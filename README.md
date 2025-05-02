# HumanliQA: A Human-Centric Web Q&A Tool

HumanliQA is a Streamlit application that uses RAG (Retrieval Augmented Generation) to provide detailed answers to questions based on web content.

## Features

- Process content from multiple web URLs
- Chat interface for natural interaction
- Powered by Google's Gemini model
- Vector-based search for relevant information

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/jaidh01/RAG-chatbot.git
   cd RAG-chatbot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set your Google API key:
   - Create a `secrets.toml` file in `.streamlit` folder
   - Add your API key: `GOOGLE_API_KEY = "your-api-key"`

## Usage

Run the Streamlit app:
```
streamlit run chatbot.py
```

## Development

- Follow Git Flow workflow for feature development
- Create feature branches from `develop`
- Submit pull requests for code review
