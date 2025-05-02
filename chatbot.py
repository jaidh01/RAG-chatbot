import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import requests
from bs4 import BeautifulSoup
import json
import re
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()

# API key management function
def get_api_key():
    """Get API key from various secure sources with appropriate fallbacks"""
    # Try to get from Streamlit secrets (for Streamlit Cloud deployment)
    if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    
    # Try to get from environment variables (for Azure or other cloud deployments)
    elif os.environ.get("GOOGLE_API_KEY"):
        return os.environ.get("GOOGLE_API_KEY")
    
    # If we don't have an API key from secrets or environment, prompt the user
    elif 'user_api_key' in st.session_state and st.session_state['user_api_key']:
        return st.session_state['user_api_key']
    
    # No API key available
    return None

# Configure the page
st.set_page_config(page_title="HumanliQA: A Human-Centric Web Q&A Tool", layout="wide")

# Initialize conversation history in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Initialize processed status
if 'urls_processed' not in st.session_state:
    st.session_state.urls_processed = False

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_web_page_text(urls):
    """Extract text from web URLs"""
    all_text = ""
    
    for url in urls:
        if not url.strip():
            continue
            
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url.strip(), headers=headers, timeout=10)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            
            # Add URL as source reference
            all_text += f"\n\nSource: {url}\n{text}\n"
            
        except Exception as e:
            all_text += f"\n\nFailed to retrieve {url}: {str(e)}\n"
            
    return all_text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    api_key = get_api_key()
    if not api_key:
        st.error("API key not found. Please check your configuration or enter your API key.")
        return False
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return True

def get_conversational_chain():
    api_key = get_api_key()
    if not api_key:
        st.error("API key not found. Please check your configuration or enter your API key.")
        return None
        
    prompt_template = """
    You are Document Genie, a helpful and knowledgeable assistant that provides detailed information based on web content.
    
    Answer the question as detailed as possible using only the provided context. If the answer is not in the context, simply say 
    "I don't have enough information about that in the provided sources."
    
    Your response must be in plain text only. DO NOT return JSON format, DO NOT include thoughts, reasoning, or commands. 
    Just provide a direct, human-readable answer without any metadata or structural elements.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        api_key = get_api_key()
        if not api_key:
            return "API key not found. Please check your configuration or enter your API key."
            
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        output_text = response["output_text"]
        
        # Better JSON detection - checking for both backticks and curly braces
        is_json = False
        if (output_text.strip().startswith("{") or 
            output_text.strip().startswith("```json") or 
            output_text.strip().startswith("`{") or
            "thoughts" in output_text and "command" in output_text):
            is_json = True
        
        # Handle JSON extraction with improved error handling
        if is_json:
            try:
                # Clean the text to extract just the JSON part
                json_text = output_text
                
                # Remove markdown code block markers if present
                json_text = re.sub(r'```json|```|\n`', '', json_text)
                
                # Try to find and extract just the JSON portion
                json_match = re.search(r'(\{.*\})', json_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                    
                parsed = json.loads(json_text)
                
                # Extract the most relevant text from the parsed JSON
                if "thoughts" in parsed:
                    if "speak" in parsed["thoughts"]:
                        output_text = parsed["thoughts"]["speak"]
                    elif "text" in parsed["thoughts"]:
                        output_text = parsed["thoughts"]["text"]
            except Exception as e:
                # If JSON parsing fails, try to extract text between quotes as fallback
                try:
                    # Look for text between quotes in the "speak" section
                    speak_match = re.search(r'"speak"\s*:\s*"([^"]*)"', output_text)
                    if speak_match:
                        output_text = speak_match.group(1)
                    else:
                        # Look for text between quotes in the "text" section
                        text_match = re.search(r'"text"\s*:\s*"([^"]*)"', output_text)
                        if text_match:
                            output_text = text_match.group(1)
                except:
                    pass  # Keep original text if all extraction attempts fail
        
        return output_text
    except Exception as e:
        return f"Error processing your question: {str(e)}. Please make sure you've loaded web content first."

def process_urls(urls):
    with st.spinner("Processing web content..."):
        try:
            raw_text = get_web_page_text(urls)
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.session_state.urls_processed = True
                return True, f"Successfully processed {len(urls)} URLs"
            else:
                return False, "No content was retrieved from the provided URLs"
        except Exception as e:
            return False, f"Error processing web content: {str(e)}"

def main():
    # Sidebar for URL input
    with st.sidebar:
        st.title("HumanliQA: A Human-Centric Web Q&A Tool 🌐")
        st.write("Process web pages to answer your questions")
        
        # Web URL input section
        st.subheader("Enter Web URLs")
        urls_input = st.text_area(
            "Enter URLs (one per line)",
            height=150,
            help="Enter multiple web page URLs, each on a new line",
            key="urls_input"
        )
        
        if st.button("Process Web Content", key="process_button"):
            if urls_input.strip():
                urls = [url.strip() for url in urls_input.strip().split('\n')]
                success, message = process_urls(urls)
                if success:
                    st.success(message)
                    # Add system message to conversation history
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": "I've processed the web content. You can now ask me questions about it!"
                    })
                else:
                    st.error(message)
            else:
                st.warning("Please enter at least one URL")

    # Main chat interface
    st.header("HumanliQA: A Human-Centric Web Q&A Tool Chat")
    
    # Initialize the processing state
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    # Initial welcome message if no conversation yet
    if not st.session_state.conversation_history and not st.session_state.urls_processed:
        with st.chat_message("assistant"):
            st.markdown("👋 Hello team at Humanli.AI — I’m glad you’re here! ")

    # Display all past conversation history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # If we're processing, show a placeholder for the user question and a spinner for the response
    if st.session_state.processing and "temp_question" in st.session_state:
        with st.chat_message("user"):
            st.markdown(st.session_state.temp_question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # This is where the processing would happen if we didn't use the two-phase approach
                pass
    
    # Chat input
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    
    # Get user question
    user_question = st.chat_input("Ask a question about the web content...")
    
    # Process new question only if it's different from the current one and we're not already processing
    if user_question and user_question != st.session_state.current_question and not st.session_state.processing:
        # Store the question
        st.session_state.current_question = user_question
        st.session_state.temp_question = user_question
        st.session_state.processing = True
        
        # Force a rerun to display the user question immediately with a spinner
        st.rerun()
    
    # Handle the processing phase
    if st.session_state.processing:
        # Get the previously stored question
        user_question = st.session_state.temp_question
        
        # Add user message to conversation history
        st.session_state.conversation_history.append({"role": "user", "content": user_question})
        
        # Only process if URLs have been processed
        if st.session_state.urls_processed:
            # Generate the response
            response = user_input(user_question)
            
            # Add assistant response to conversation history
            st.session_state.conversation_history.append({"role": "assistant", "content": response})
        else:
            message = "Please process some web content first by adding URLs in the sidebar and clicking 'Process Web Content'."
            st.session_state.conversation_history.append({"role": "assistant", "content": message})
        
        # Reset processing state
        st.session_state.processing = False
        
        # Force a rerun to display the new messages
        st.rerun()

if __name__ == "__main__":
    main()
