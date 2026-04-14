# Chatbot_With_RAG

## Purpose Of Project 
This project has taught us how Chatbot works with Retrieval Augmented Generation. This means that we upload some documents and Chatbot answers our question based on uploaded documents . The response is not pure knowledge from vector database . It is mixed with knowledge and creativity coming from LLM .

## Libraries 
- Langchain
- Ollama (for Embeddings  and LLM)
- Streamlit (for UI)
- PyPDF (to read pdf docs)
- Dotenv (to read environment variables)

## Requirements For Usage
Whatever embeddings that you use , you have to run the embedding models first and if your embedding models work with API_KEY ,you have to create .env file and create API_KEY that you get from the model website online.\n\n

- I used Ollama model for embedding and LLM. In order to run that , first you have to run Ollama by typing "ollama serve" on your terminal
- After that , pull the embedding using "ollama pull nomic-embed-text".
- Finally , you can run streamlit server using "streamlit run {your main python file}". For example, streamlit run app.py .



