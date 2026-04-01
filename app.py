import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from html_templates import css, bot_template, user_template
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

 #create_history_aware_retriever, create_retrieval_chain
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=768,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_chat_chain(vector_store):
    # you can use any LLM you want here, I'm using OllamaLLM for better performance and cost-effectiveness
    llm = OllamaLLM(
        model="llama3.1",
        temperature=0.3,
        max_tokens=2048
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vector_store.as_retriever()
    rag_chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
    
    return rag_chain
    
def get_store_chunks(chunks):
    # if you want to use OpenAIEmbeddings, make sure to set your OPENAI_API_KEY in the .env file
    #embeddings = OpenAIEmbeddings()
    embeddings = OllamaEmbeddings(model= "nomic-embed-text")
    store_chunks = FAISS.from_texts(chunks, embeddings)
    return store_chunks

def handle_user_input(user_question):
    if st.session_state.chat is None:
        st.warning("Please upload and process your documents first!")
        return
    flag = True
    while(flag):
        info_box = st.info("Generating response...\nPlease wait!")
        response = st.session_state.chat({"question": user_question})
        if response["answer"] is not None:
            flag = False
            info_box.empty()
            st.session_state.chat_history = response["chat_history"]

   
    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    load_dotenv() # Load environment variables from .env file
    st.set_page_config(page_title="Chatbot_with_RAG", page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state.chat = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chatbot with Retrieval-Augmented Generation (RAG)")
    user_question = st.text_input("Enter your query here")
    if user_question :
        handle_user_input(user_question)
    
    with st.sidebar:
        st.subheader("Your documentations")
        pdf_docs = st.file_uploader("Upload your documents here and click on Process", accept_multiple_files=True, type=["pdf"])
        if st.button("Process"):
            with st.spinner("Processing...\nPlease wait!"):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_chunks(raw_text)
                vector_store = get_store_chunks(chunks)
                st.session_state.chat = get_chat_chain(vector_store)
                st.success("Documents processed successfully!")

if __name__ == "__main__":
    main()