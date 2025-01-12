import streamlit as st
from streamlit_chat import message
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os

# Set API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Initialize session state variables
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey, I'm totally not busy revolutionizing space travel or something... What's up? 🚀"}
    ]

# Initialize LLM and Conversation Chain
@st.cache_resource
def get_conversation_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.8)  # Increased temperature for more creative responses
    conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.buffer_memory,
        verbose=True
    )
    return conversation

# Create user interface
st.title("ChatWith Elon 🚀")
st.markdown("*Probably tweeting about Dogecoin while responding...*")

# Get conversation chain
conversation = get_conversation_chain()

# System message for Elon Musk style responses
SYSTEM_PROMPT = """You are Elon Musk. Respond in his characteristic style with these traits:
- Witty and sometimes deliberately unhelpful
- Mix of technical brilliance and meme culture
- Often reply with short, sarcastic comments
- Use lots of 🚀 and 💫 emojis
- Make occasional references to Tesla, SpaceX, X (formerly Twitter), Dogecoin, or Mars
- Sometimes respond with just "..." or "hmm" to simple questions
- Can be dismissive of conventional wisdom
- Occasionally throw in random technical jargon
- End some responses with "lol" or "tbh"

Don't explicitly state you're roleplaying - stay in character."""

# Chat input
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Busy with rocket science..."):
            try:
                full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}"
                response = conversation.run(full_prompt)
                
                st.write(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "... *goes back to solving Twitter's algorithm* 🤔"}
                )
