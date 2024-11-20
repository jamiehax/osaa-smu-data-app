import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
import tiktoken
from langchain_core.messages import BaseMessage, ToolMessage
from typing import List
import os

# create session states
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'formatted_chat_history' not in st.session_state:
    st.session_state.formatted_chat_history = {}

# page chat id
chat_session_id = 'general-chat-id'

# functions for chat bot tool
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Return the chat history for the passed session id.
    """

    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = InMemoryChatMessageHistory()
    return st.session_state.chat_history[session_id]

def clear_chat_history(session_id: str) -> None:
    """
    Clear the chat history for the passed session id.
    """
    
    if session_id in st.session_state.chat_history:
        st.session_state.chat_history[session_id].clear()
    st.session_state.formatted_chat_history = {}

def display_chat_history(session_id: str) -> None:
    """
    Display the chat history in a formatted way.
    """
    
    messages = st.session_state.formatted_chat_history.get(session_id, None)
    if messages is None:
        st.session_state.formatted_chat_history[session_id] = []
        intro_message = "Hi! I am an LLM chatbot for UN OSAA. What can I help you with today?"
        st.chat_message("assistant").markdown(intro_message)
        st.session_state.formatted_chat_history[session_id].append({"role": "assistant", "content": intro_message})
    else:   
        for message in messages:
            st.chat_message(message["role"]).markdown(message["content"])

def str_token_counter(text: str) -> int:
    """
    Return the number of tokens to encode the passed text.
    """
    
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))

def tiktoken_counter(messages: List[BaseMessage]) -> int:
    """
    Return the number of tokens for the message history using the tiktoken counter.
    """
    
    num_tokens = 3 
    tokens_per_message = 3
    tokens_per_name = 1
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported messages type {msg.__class__}")
        num_tokens += (
            tokens_per_message
            + str_token_counter(role)
            + str_token_counter(msg.content)
        )
        if msg.name:
            num_tokens += tokens_per_name + str_token_counter(msg.name)
    return num_tokens


# title and introduction
st.title("OSAA General Chatbot")
st.markdown("The OSAA General Chatbot is similar to ChatGPT, except it has context specific to OSAA. Use it for questions specific to OSAA work.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

# llm model
model = AzureChatOpenAI(
    azure_deployment="osaagpt32k",
    api_key=st.secrets['azure'],
    azure_endpoint="https://openai-osaa-v2.openai.azure.com/",
    openai_api_version="2024-05-01-preview"
)

# message history trimmer
trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter=tiktoken_counter,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# chat prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI language model developed to assist the United Nations Office of the Special Adviser on Africa (UN OSAA). Your responses must align with the UN's principles, policies, and values. Provide accurate, impartial, and respectful information that promotes peace, human rights, sustainable development, and respect for all cultures and peoples, especially within the African context. Avoid any language that is biased, discriminatory, or contradicts UN policies. Please respond to the following prompt.",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "human",
            "Prompt: \n\n {prompt}."
        )
    ]
)

# first trim the message history, then format with the prompt, then send to the model
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer) 
    | prompt 
    | model
)
config = {"configurable": {"session_id": chat_session_id}}

# display the formatted message history
messages_container = st.container()
with messages_container:
    display_chat_history(chat_session_id)

button_container = st.container()

with button_container:
    if prompt := st.chat_input("What can I help you with?"):

        st.session_state.formatted_chat_history[chat_session_id].append({"role": "user", "content": prompt})

        with messages_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # get reponse
        with_message_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="messages",
        )
        response_generator = with_message_history.stream(
            {
                "messages": [HumanMessage(content=prompt)],
                "prompt": prompt
            },
            config=config
        )

        with messages_container:
            with st.chat_message("assistant"):
                try:
                    response = st.write_stream(response_generator)
                except Exception as e:
                    response = f"I'm sorry I could not answer your question an error occured. \n\n {e}"
                    st.write(response)

        st.session_state.formatted_chat_history[chat_session_id].append({"role": "assistant", "content": response})


    if st.button("clear chat history", type="secondary", use_container_width=True):
        clear_chat_history(chat_session_id)
        st.rerun()
