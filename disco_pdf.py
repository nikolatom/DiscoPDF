import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile




def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hi, I am your PDF interpreter, what would you like to ask?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi there!"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(vector_store):
    # Create llm
    llm = LlamaCpp(
    streaming = True,
    model_path="model/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf",
    temperature=0.75,
    top_p=1, 
    verbose=True,
    n_ctx=4096,
    n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=-1, # Change this value based on your model and your GPU VRAM pool.
)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

def main():
    # Initialize session state
    initialize_session_state()
    st.title("DiscoPDF: Multi-PDF ChatBot & Interpreter")
    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
        progress_bar = st.progress(0)
        st.caption("Uploading files...")
        
        text = []
        for index, file in enumerate(uploaded_files):
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)
            
            progress_bar.progress((index + 1) / len(uploaded_files) * 0.2)  # Assuming file upload and initial processing is ~20% of the work

        st.caption("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(text)
        progress_bar.progress(0.4)  # 40% progress

        st.caption("Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})
        # Assume embedding creation is another 20% of the work
        progress_bar.progress(0.6)

        st.caption("Creating vector store...")
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        # Updating progress to 80%, assuming vector store creation is 20%
        progress_bar.progress(0.8)

        st.caption("Initializing conversational chain...")
        chain = create_conversational_chain(vector_store)
        # Complete the progress bar to indicate initialization is done
        progress_bar.progress(1.0)
        st.caption("Initialization complete.")

        # Display the chat history and interact with the model
        display_chat_history(chain)

if __name__ == "__main__":
    main()
