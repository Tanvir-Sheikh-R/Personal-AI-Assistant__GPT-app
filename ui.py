import streamlit as st

def load_page_style():

    st.set_page_config(page_title="Personal AI Assistant", page_icon=":material/asterisk:")

    st.markdown("""
    <style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Newsreader:ital,wght@0,400;1,400&display=swap');
        html,
        body,
        [class*="css"] {
            font-family: 'Comfortaa', 'Inter', sans-serif;
        }

        h1,h2,h3 {
            font-family: 'Newsreader', serif;
        }

        h1 {
            padding: 0px !important;
        }

        /* Reverse the user row: avatar-slot moves to the right of the bubble */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse;
        }

        /* Hide the user avatar icon */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] {
            display: none;
        }


        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            width: fit-content !important;
            margin-left: auto;
            display: flex;
            align-items: flex-start;
            gap: 0.5rem;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: rgba(240, 242, 246, 0.5);
        }

        /* hide the face icon since you don't want a logo on user messages */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] {
            display: none;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"]) [data-testid="stChatMessageAvatarCustom"] {
            border: 1px solid #7D936F;
            background-color: #7D936F;
            color: #7D936F !important;
        }

        [data-testid="stIconMaterial"] {
            color: white !important;
        }

        [data-testid="stIconMaterial"] {
            display: inline-block;
            transition: transform 0.3s ease;
        }

        [data-testid="stIconMaterial"]:hover {
            transform: rotate(45deg);
        }

        [data-testid="stHeaderActionElements"] {
            display: none;
        }

        [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
        [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {
            color: #7D936F !important;
            visibility: visible !important;
        }

        [data-testid="stMarkdownContainer"]{
            margin-bottom: 0rem;
        }

    </style>
    """, unsafe_allow_html=True)