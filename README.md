# 🧠 RAG Web Content Q&A Tool – Humanli.AI Assignment

This is my submission for the ML Engineer assignment at **Humanli.AI**. The application uses a Retrieval-Augmented Generation (RAG) architecture to answer questions based only on the content of provided web URLs.

## 🔗 Live App
Access the deployed app here:  
👉 [https://rag-chatbot.app](https://rag-chatbot-huamnliai.streamlit.app)

---

## 📌 Features
- 📝 Enter one or more webpage URLs
- 🔍 Scrape and chunk content
- 📚 Embed using HuggingFace models
- 🗂 Store & retrieve with FAISS vector DB
- 🤖 Generate answers using Gemmini (OpenAI-compatible LLM)
- 💡 Ask multiple questions after one ingestion (session-based)

---

## 🧱 Tech Stack
- `LangChain`
- `FAISS`
- `HuggingFace Embeddings`
- `Streamlit`
- `Gemmini API` (OpenAI-compatible)

---

## 🚀 How to Run Locally

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

## Repository Structure
```
RAG-chatbot/
├── chatbot.py               # Main Streamlit app
├── requirements.txt         # All required packages
├── .streamlit/
│   └── secrets.toml.example 
├── README.md                # Explains project
```

## Development

- Follow Git Flow workflow for feature development
- Create feature branches from `develop`
- Submit pull requests for code review
