import streamlit as st
import os
from helper_functions import get_retriever, make_vectorstore
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = st.secrets['langsmith']
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_PROJECT"] = "osaa-smu-contradictory-analysis"

# session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'formatted_chat_history' not in st.session_state:
    st.session_state.formatted_chat_history = {}

# chat hsitory id for check analysis page
chat_session_id = 'check-analysis-chat-id'


# functions for the chatbot tool
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
        intro_message = "Hi! I am an LLM chatbot assistant with access to OSAA publications. I can answer whether provided analysis contradicts any analysis present in these publications."
        st.chat_message("assistant").markdown(intro_message)
        st.session_state.formatted_chat_history[session_id].append({"role": "assistant", "content": intro_message})
    else:   
        for message in messages:
            st.chat_message(message["role"]).markdown(message["content"])

def format_docs(docs):
    """
    Join the retrieved documents together, along with title and Metadata, to pass to model.
    """
    
    formatted_contexts = []
    for doc in docs:
        content = doc.page_content
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "Unknown Page")
        
        formatted_context = f"Publication Name: {source}. Page Number: {page}. Content: {content}"
        formatted_contexts.append(formatted_context)
    
    return "\n\n".join(formatted_contexts)

# title and introduction
st.title("OSAA SMU's Contradictory Analysis Tool")
st.markdown("The Contradictory Analysis Tool allows you to check if your analysis contradicts any previous analysis in OSAA's publications. This tool uses large language models with retrieval augmented generation and therefore may provide wrong answers.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

llm = AzureChatOpenAI(
    azure_deployment="osaagpt32k",
    api_key=st.secrets['azure'],
    azure_endpoint="https://openai-osaa-v2.openai.azure.com/",
    openai_api_version="2024-05-01-preview"
)

# document retriever
retriever = get_retriever('content/vectorstore.duckdb')

# prompt tempalte
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant. Your task is to determine whether the provided analysis contradicts any content from our existing publications. If the analysis contradicts any content from our existing publications, clearly state that there is a contradiction. If the analysis does not directly contradict content from the existing publications, but contains data points or facts that are different from the publications, say so. If there is no contradiction, state that there is no contradiction. If you are unsure, state that you are unsure. Reference the publication name when making your decision."
        ),
        (
            "human",
            "provided analysis: {analysis}."
        ),
        (
            "human",
            "existing publication content: {context}."
        ),
        (
            "system",
            "Based on the Analysis and the existing publication content, does the analysis contradict anything in the existing publications? Always provide the publication name and page number of existing publication content, and if possible, provide a quote to support your reasoning. If you are unsure whether there is a contradiction, provide the existing publication content so the user can make their own decision."
        )
    ]
)

# get docs with retriever, format them, pass them to the model
chain = (
    {"context": retriever | format_docs, "analysis": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

messages_container = st.container()
with messages_container:
    display_chat_history(chat_session_id)

button_container = st.container()

with button_container:
    if analysis := st.chat_input("provide some analysis..."):

        st.session_state.formatted_chat_history[chat_session_id].append({"role": "user", "content": analysis})
        messages_container.chat_message("user").markdown(analysis)

        response_generator = chain.stream(analysis)

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

    # if st.button("remake vectorstore", use_container_width=True):
    #     make_vectorstore()