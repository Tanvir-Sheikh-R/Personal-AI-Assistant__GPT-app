# Personal AI Assistant

A conversational AI chatbot built with LangGraph and Streamlit, powered by Groq's Llama 3.1 model.

🔗 **Live Demo:** [ai-chatbot-groqllm.streamlit.app](https://ai-chatbot-groqllm.streamlit.app/)

## Features

- Real-time streaming responses (token-by-token)
- Multi-thread conversation history — create, switch between, and resume chats
- Auto-generated conversation titles using the LLM
- Persistent session state via in-memory SQLite checkpointing
- Custom-styled chat UI with role-based message alignment
- Graceful error handling for API rate limits and connectivity issues

## Tech Stack

- **Language:** Python
- **Orchestration:** LangGraph, LangChain
- **LLM:** Groq (Llama 3.1)
- **Frontend:** Streamlit
- **Storage:** SQLite (in-memory checkpointer)

## Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/keys)

### Installation

```bash
git clone https://github.com/Tanvir-Sheikh-R/Personal-AI-Assistant__GPT-app
cd Personal-AI-Assistant__GPT-app
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_api_key_here
```

### Run Locally

```bash
streamlit run chat_app_frontend.py
```

## Project Structure

```
├── chat_app_frontend.py    # Streamlit UI
├── chat_app_backend.py     # LangGraph agent + LLM logic
├── ui.py                   # Page styling / CSS
├── requirements.txt
└── README.md
```

## License

This project is open source and available under the [MIT License](LICENSE).
