import streamlit as st
import os
from dotenv import load_dotenv
import model

# Load environment variables
load_dotenv()

# Set up the page layout
st.set_page_config(page_title="Chat and File Upload", layout="wide")
st.title("Doc QnA ChatBot")

# Initialize the bot
bot = model.RAGPDFBot()

# Function to display chat bubbles
def display_chat_bubbles(messages):
    for message in messages:
        if message['role'] == 'user':
            st.chat_message(message['role']).markdown(f"<div style='background-color:#DCF8C6;padding:10px;border-radius:10px;color:black'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.chat_message(message['role']).markdown(f"<div style='background-color:#f1f1f1;padding:10px;border-radius:10px;color:black'>{message['content']}</div>", unsafe_allow_html=True)

def initialize_model(filePath):
    top_k = 2
    chunk_size = 500
    overlap = 50
    max_length = 128
    temp = 0.7
    bot.load_model(max_length=max_length, repeat_penalty=1.50, top_k=top_k, temp=temp)
    st.write("Model Loaded!")
    
    # Build the vector database
    st.write("Building vector DB...")
    bot.build_vectordb(chunk_size=chunk_size, overlap=overlap, file_path=filePath)
    st.write("Model and Vector DB Build Done!!")

def retrieve(input):
    bot.retrieval(user_input=input)
    return bot.inference()

# First Column - File Upload
col1, spacer, col2 = st.columns([0.3, 0.05, 0.7])

with col1:
    st.header("Upload File")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        save_dir = "DocumentQnAChatBot/"
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the uploaded file
        file_path = os.path.join(save_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        initialize_model(file_path)

# Second Column - Chat Window
with col2:
    st.header("Chat Window")

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]
    
    display_chat_bubbles(st.session_state['messages'])
    
    if 'text_value' not in st.session_state:
        st.session_state.text_value = ''

    user_input = st.text_input("Your message:", value=st.session_state.text_value)
    send_button = st.button("Send")
    
    if send_button or user_input:
        st.session_state['messages'].append({"role": "assistant", "content": "I'm processing your input..."})
        st.session_state['messages'] = [{"role": "assistant", "content": retrieve(user_input)}]
        display_chat_bubbles(st.session_state['messages'])
        st.session_state['messages'].append({"role": "user", "content": user_input})
        st.session_state.text_value = ''
