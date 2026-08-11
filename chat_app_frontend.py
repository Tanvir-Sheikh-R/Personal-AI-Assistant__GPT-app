import streamlit as st
from ui import load_page_style
import numpy as np
from chat_app_backend import chat, checkpointer, get_summary_for_chatHead
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
import uuid

load_page_style()



# ****************************** Utility Functions ******************************
def generate_thread_id():
    id = uuid.uuid4()
    return id

def reset_chat():
    st.session_state['message'] = []

    new_id = generate_thread_id()
    st.session_state['thread_id'] = new_id
    add_thread(new_id)
    

def add_thread(thread_id):
    if thread_id not in st.session_state['thread_id_list']:
        st.session_state['thread_id_list'].append(thread_id)


def load_conversation(thread_id):
    conversation = chat.get_state({'configurable': {'thread_id': thread_id}}).values.get('message')
    return conversation



# ****************************** Session States ******************************
if 'message' not in st.session_state:
    st.session_state['message'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'thread_id_list' not in st.session_state:
    st.session_state['thread_id_list'] = []

add_thread(st.session_state['thread_id'])

if 'pdf_files' not in st.session_state:
    st.session_state['pdf_files'] = []



# ****************************** SideBar UI ******************************

with st.sidebar:
    st.title('AI Assistant')

    if st.button('New Chat', width='stretch', type='primary'):
        reset_chat()

    st.divider()
    st.header('Knowledge Base')

    uploaded_files = st.file_uploader(
        'Upload documents',
        type=['pdf', 'docx', 'txt', 'md'],
        accept_multiple_files=True,
    )

    if uploaded_files:
        all_docs = []
        for uploaded_file in uploaded_files:
            st.write(f"Processing: {uploaded_file.name}")
            # e.g. pass bytes to a PDF loader
            file_bytes = uploaded_file.read()
            # your_pdf_loader(file_bytes) -> chunks
            # all_docs.extend(chunks)
            st.session_state['pdf_files'].append(file_bytes)


    st.header('Chat history')
    st.divider()

    for id in st.session_state.thread_id_list[::-1]:
        conversation = load_conversation(id)

        if conversation:
            user = conversation[0].content
            summery = get_summary_for_chatHead(user)
            

            if st.button(summery, width='stretch', key = id): 
                response = load_conversation(id)
                temp_message = []

                for msg in response:
                    if isinstance(msg, HumanMessage):
                        role = 'user'
                    else:
                        role = 'assistant'

                    temp_message.append({'role': role, 'msg': msg.content})

                st.session_state['message'] = temp_message
            


st.image("src/logo_green.svg", width=80)
st.markdown("""# <h1>Personal AI Assistant</h1>""", unsafe_allow_html=True)

st.markdown('<p style="color: #6B8E55">Hi, Whats your agenda today?</p>', unsafe_allow_html=True)


for messages in st.session_state['message']:
    if messages["role"] == 'assistant':
        with st.chat_message('assistant' , avatar=":material/asterisk:"):
            st.write(messages['msg'])

    if messages["role"] == 'user':
         with st.chat_message('user'):
            st.write(messages['msg'])
         

user_input = st.chat_input(
                        "Type here", 
                        # accept_file=True, 
                        # file_type=['pdf', 'docx', 'txt', 'md']
                        )

# if user_input:
#     text = user_input.text          
#     files = user_input.files        

#     if files:
#         for f in files:
#             st.write(f"Attached: {f.name}")
#             file_bytes = f.getvalue()

CONFIG = {'configurable': {'thread_id': st.session_state.thread_id}}

if user_input:
    st.session_state.message.append({'role': 'user', 'msg': user_input})

    with st.chat_message('user'):
            st.write(user_input)
    
    with st.chat_message('assistant' , avatar=":material/asterisk:"):
        response = st.write_stream(
            message_chunk.content for message_chunk, metadata in chat.stream(
                {'message': [HumanMessage(user_input)]}, 
                config=CONFIG, 
                stream_mode='messages'
            ))
    st.session_state.message.append({'role': 'assistant', 'msg':  response})
    st.rerun()


# print()
# print(st.session_state)
# print(list(checkpointer.list(config={'thread_id': '1'})))