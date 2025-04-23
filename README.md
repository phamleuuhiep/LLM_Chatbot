# LLM_Chatbot
This project is a multi-functional virtual assistant application built with Gradio that leverages Large Language Models (LLMs) through OpenRouter and Google's Gemini to interact with users in a natural, conversational manner.

## Install and Demonstrate
### WSL (ubuntu)
Step 1: Create a virtual environment 
python3 -m venv myenv
source myenv/bin/activate

Step 2: Install libraries
pip install gradio openai==0.28.0 python-dotenv requests beautifulsoup4 pymupdf

Step 3: Run the Python source code
python3 app.py

### CMD (windows)
Step 1: Create a virtual environment 
python3 -m venv myenv
myenv\Scripts\activate
If a security warning in PowerShell raises, try to run:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
myenv\Scripts\activate

Step 2: Install libraries
pip install gradio openai==0.28.0 python-dotenv requests beautifulsoup4 pymupdf

Step 3: Run the Python source code
python3 app.py



