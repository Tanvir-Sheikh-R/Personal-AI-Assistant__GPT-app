import streamlit as st
from chat_app_backend_rag import add_documents_to_store
from ui import load_page_style
from chat_app_backend import chat, get_summary_for_chatHead
from langchain_core.messages import HumanMessage
import uuid
from pathlib import Path
from langchain_core.messages import AIMessageChunk

load_page_style()




UPLOAD_DIR = Path('.uploaded_files')
UPLOAD_DIR.mkdir(exist_ok=True)


def save_uploaded_files(files) -> list[str]:
    saved = []
    for f in files:
        target = UPLOAD_DIR / f.name
        target.write_bytes(f.getvalue())
        saved.append(str(target))
    return saved

def cleanup_uploaded_files():
    if UPLOAD_DIR.exists():
        for p in UPLOAD_DIR.iterdir():
            if p.is_file():
                p.unlink(missing_ok=True)


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

if 'indexed_files' not in st.session_state:
    st.session_state['indexed_files'] = set()

if 'kb_id' not in st.session_state:
    st.session_state['kb_id'] = str(uuid.uuid4())

if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = {}


# ****************************** SideBar UI ******************************

with st.sidebar:
    st.title('AI Assistant')

    if st.button('New Chat', width='stretch', type='primary'):
        reset_chat()


    st.header('Chat history')
    # st.divider()
    # st.markdown("""
    #         <hr style="margin-top: -1rem, margin-bottom: -0.5rem;">
    #     """,
    #     unsafe_allow_html=True
    # )


    for id in st.session_state.thread_id_list[::-1]:
        conversation = load_conversation(id)

        if conversation:
            if id not in st.session_state['thread_titles']:
                user = conversation[0].content
                st.session_state['thread_titles'][id] = get_summary_for_chatHead(user)
            summery = st.session_state['thread_titles'][id]

            if st.button(summery, width='stretch', key = str(id)): 
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
                        accept_file=True, 
                        file_type=['pdf', 'docx', 'txt', 'md']
                        )

if user_input:
    text = user_input.text
    files = user_input.files

    if files:
        indexing_failed = False
        with st.spinner("Indexing uploaded documents..."):
            saved_paths = save_uploaded_files(files)
            for p in saved_paths:
                st.write(f"Uploading file: {Path(p).name}")

            new_paths = [
                p for p in saved_paths
                if Path(p).name not in st.session_state['indexed_files']
            ]

            if new_paths:
                try:
                    add_documents_to_store(new_paths, collection_name=st.session_state['kb_id'])
                    with st.spinner("Loading embeddings and processing document..."):
                        st.session_state['indexed_files'].update(
                            Path(p).name for p in new_paths
                        )
                except Exception as e:

                    indexing_failed = True
                    st.warning(f"Couldn't index one or more files ({e}). Continuing without them.")
                finally:
                    cleanup_uploaded_files()

        attached_note = "Attached (indexing failed): " if indexing_failed else "Attached: "
        text = f"{attached_note}{', '.join(Path(p).name for p in saved_paths)}\n\n {text}"

CONFIG = {'configurable': {'thread_id': st.session_state.thread_id}}

NODE_LABELS = {
    'chat_message': 'Thinking',
    'tools': 'Using tools',
}


# ---------------- Message Streaming----------------

if user_input:
    st.session_state.message.append({'role': 'user', 'msg': text})

    with st.chat_message('user'):
            st.write(text)
    
    with st.chat_message('assistant', avatar=":material/asterisk:"):
        placeholder = st.empty()
        def render_placeholder(label: str):
            placeholder.markdown(
                f"""
                <style>
                @keyframes pulse {{
                    0%   {{ opacity: 0.3; }}
                    50%  {{ opacity: 1; }}
                    100% {{ opacity: 0.3; }}
                }}
                .thinking {{
                    animation: pulse 1.4s ease-in-out infinite;
                    font-style: italic;
                    color: gray;
                }}
                </style>
                <span class="thinking">{label}...</span>
                """,
                unsafe_allow_html=True,
            )

        render_placeholder("Thinking")

# ----------------- output buffer for thinking and remove duplicate ----------------------

        def stream_wrapper():
            buffer = ""
            current_id = None
            has_tool_call = False
            first_flush = True

            for message_chunk, metadata in chat.stream(
                {'message': [HumanMessage(text)], 'kb_id': st.session_state['kb_id']},
                config=CONFIG,
                stream_mode='messages'
            ):
                node = metadata.get('langgraph_node')
                label = NODE_LABELS.get(node, "Thinking")

                # Not the chat_message node (e.g. tools, check_answer) -> just update the label
                if node != 'chat_message':
                    render_placeholder(label)
                    continue

                if not isinstance(message_chunk, AIMessageChunk):
                    continue

                if message_chunk.id != current_id:
                    if buffer and not has_tool_call:
                        if first_flush:
                            placeholder.empty()
                            first_flush = False
                        yield buffer
                    buffer = ""
                    has_tool_call = False
                    current_id = message_chunk.id

                if message_chunk.tool_call_chunks:
                    has_tool_call = True

                if message_chunk.content:
                    buffer += message_chunk.content
                else:
                    render_placeholder(label)

            if buffer and not has_tool_call:
                if first_flush:
                    placeholder.empty()
                    first_flush = False
                yield buffer

        response = st.write_stream(stream_wrapper())
    st.session_state.message.append({'role': 'assistant', 'msg':  response})

    st.rerun()
    
