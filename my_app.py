import os
import re
import gradio as gr
import requests
import fitz  # PyMuPDF
import google.generativeai as genai
import openai
from bs4 import BeautifulSoup
import tempfile


import shutil

# # Load environment variables from .env file (or api.env)
# load_dotenv("api.env")  

# # API key
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# MODEL = "google/gemini-2.0-flash-001" ## GEMINI 2.0 Flash model  
MODEL = "deepseek/deepseek-r1-0528-qwen3-8b:free" # Qwen-3 8B model


openai.api_key = "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"
openai.api_base = "https://openrouter.ai/api/v1"


conversation_history = []
def chatbot(message, history, api_key=None):

    """
    Process a chat message and generate a response using the LLMs
    
    Args:
        message: The user's message
        history: Chat history
        api_key: Optional API key provided by user
    """

    global conversation_history, genai, MODEL
    
    # Configure the client 
    if api_key and api_key.strip():
        genai.configure(api_key=api_key)

    # Initialize conversation history 
    if conversation_history is None:
        conversation_history = []

    # Add user message to conversation history
    user_message = {"role": "user", "content": message}
    conversation_history.append(user_message)

    # Prepare assistant message for incremental streaming
    assistant_message = {"role": "assistant", "content": ""}

    try:
        # Model creation
        
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=conversation_history,
            stream=True
        )

        for chunk in response:
            if "choices" in chunk and chunk["choices"][0]["delta"].get("content"):
                assistant_message["content"] += chunk["choices"][0]["delta"]["content"]
                yield history + [user_message, assistant_message]

        conversation_history.append(assistant_message)

    except Exception as e:
        error_message = {"role": "assistant", "content": f"Error: {str(e)}"}
        yield history + [user_message, error_message]





def clear_history():
    """Clear the conversation history"""
    global conversation_history
    conversation_history = []
    return None, []

def extract_text_from_url(url):
    """
    Extract and clean text content from a URL
    
    Args:
        url: The URL to extract content from
    """
    try:
        # Send a GET request to the URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
            script_or_style.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        return f"Error extracting text from URL: {str(e)}"

def summarize_url(url, api_key=None):
    """
    Summarize content from a URL using Gemini
    
    Args:
        url: The URL to summarize
        api_key: Optional API key provided by user
    """
    global genai
    
    # Update API key 
    if api_key and api_key.strip():
        genai.configure(api_key=api_key)
    
    # Extract text from URL
    text = extract_text_from_url(url)
    
    # Error handling
    if text.startswith("Error"):
        return text
    max_chars = 30000  # 30k tokens maximum
    if len(text) > max_chars:
        text = text[:max_chars] + "...[Content truncated due to length]"
    
    try:

        # Model creation
        prompt = f"Please summarize the following web article in a comprehensive way. Include key points, data, and insights:\n\n{text}"
        
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        answer = ""
        for chunk in response:
            if "choices" in chunk and chunk["choices"][0]["delta"].get("content"):
                answer += chunk["choices"][0]["delta"]["content"]
                yield answer
        
    except Exception as e:
        yield f"Error: {str(e)}"

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file
    
    Args:
        file_path: Path to the PDF file
    """
    try:
        text = ""
        with fitz.open(file_path) as pdf_document:
            # Extract text from each page
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text += page.get_text()
        
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def process_pdf(pdf_file, question, api_key=None):
    """
    Process a PDF file and answer questions about it
    Arguments:
        pdf_file: The uploaded PDF
        question: User's question about the PDF
        api_key: Optional new API key
    """
    global genai
    
    if api_key and api_key.strip():
        genai.configure(api_key=api_key)
    
    if pdf_file is None:
        return "Please upload a PDF file first."
    
    try:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfile(pdf_file.name, temp_file.name)
            temp_file_path = temp_file.name
        
        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(temp_file_path)
        # Delete the temporary file
        os.unlink(temp_file_path)
        
        # If extraction failed, return the error
        if pdf_text.startswith("Error"):
            return pdf_text
        max_chars = 30000  # 30k tokens
        if len(pdf_text) > max_chars:
            pdf_text = pdf_text[:max_chars] + "...[Content truncated due to length]"
        
        # Model creation
        prompt = f"Here is the content of a PDF document:\n\n{pdf_text}\n\nBased on this document, please answer the following question: {question}"
        
        # Streaming
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        answer = ""
        for chunk in response:
            if "choices" in chunk and chunk["choices"][0]["delta"].get("content"):
                answer += chunk["choices"][0]["delta"]["content"]
                yield answer
        
    except Exception as e:
        yield f"Error: {str(e)}"

def search_web(query, api_key=None):
    """
    Search the web for information using Gemini
    
    Args:
        query: Searching query
        api_key: Optional API key provided by user
    """
    global genai
    
    # Update API key 
    if api_key and api_key.strip():
        genai.configure(api_key=api_key)
    
    try:
        # Create model and generate response
        
        prompt = f"Search for information about: {query}. Provide the most up-to-date information you have. If the query is about events after February 2023, clearly state that limitation."
        
        # Generate response with streaming
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        answer = ""
        for chunk in response:
            if "choices" in chunk and chunk["choices"][0]["delta"].get("content"):
                answer += chunk["choices"][0]["delta"]["content"]
                yield answer
        
    except Exception as e:
        yield f"Error: {str(e)}"

# Build the Gradio interface
with gr.Blocks(title="Virtual Assistant") as demo:
    gr.Markdown("## 🔮 Chat with AI Assistant 🔮")
    gr.Markdown("### 💡 Tip: You can hit `Enter` to send a message.")
    
    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column(scale=4):
                chatbot_interface = gr.Chatbot(height=500, type="messages")
            with gr.Column(scale=1):
                api_key_input = gr.Textbox(
                    placeholder="Enter your API key (optional)", 
                    label="API Key", 
                    type="password"
                )
        
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Type your message here...",
                label="Message",
                lines=3
            )
        
        with gr.Row():
            submit_btn = gr.Button("Send")
            clear_btn = gr.Button("Clear Chat")
    
    with gr.Tab("URL Summarizer"):
        with gr.Row():
            url_input = gr.Textbox(
                placeholder="Enter URL to summarize",
                label="URL"
            )
            url_api_key = gr.Textbox(
                placeholder="Enter your API key (optional)", 
                label="API Key", 
                type="password"
            )
        
        with gr.Row():
            url_submit_btn = gr.Button("Summarize")
        
        with gr.Row():
            url_output = gr.Textbox(
                label="Summary",
                lines=10
            )
    
    with gr.Tab("PDF Answering"):
        with gr.Row():
            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"]
            )
            pdf_api_key = gr.Textbox(
                placeholder="Enter your API key (optional)", 
                label="API Key", 
                type="password"
            )
        
        with gr.Row():
            pdf_question = gr.Textbox(
                placeholder="What do you want to know about the PDF?",
                label="Question"
            )
        
        with gr.Row():
            pdf_submit_btn = gr.Button("Get Answer")
        
        with gr.Row():
            pdf_output = gr.Textbox(
                label="Answer",
                lines=10
            )
    
    with gr.Tab("Web Search"):
        with gr.Row():
            search_input = gr.Textbox(
                placeholder="What do you want to search for?",
                label="Search Query"
            )
            search_api_key = gr.Textbox(
                placeholder="Enter your API key (optional)", 
                label="API Key", 
                type="password"
            )
        
        with gr.Row():
            search_submit_btn = gr.Button("Search")
        
        with gr.Row():
            search_output = gr.Textbox(
                label="Search Results",
                lines=10
            )

    # Event handlers
    submit_btn.click(
        chatbot,
        inputs=[msg_input, chatbot_interface, api_key_input],
        outputs=chatbot_interface
    )
    
    msg_input.submit(
        chatbot,
        inputs=[msg_input, chatbot_interface, api_key_input],
        outputs=chatbot_interface
    )
    
    clear_btn.click(
        clear_history,
        inputs=[],
        outputs=[msg_input, chatbot_interface]
    )
    
    url_submit_btn.click(
        summarize_url,
        inputs=[url_input, url_api_key],
        outputs=url_output
    )
    
    pdf_submit_btn.click(
        process_pdf,
        inputs=[pdf_input, pdf_question, pdf_api_key],
        outputs=pdf_output
    )
    
    search_submit_btn.click(
        search_web,
        inputs=[search_input, search_api_key],
        outputs=search_output
    )

# App launching
if __name__ == "__main__":
    demo.queue().launch()


