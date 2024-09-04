
import streamlit as st
import duckdb
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.duckdb import DuckDB
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = st.secrets['langsmith']
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_PROJECT"] = "osaa-smu-contradictory-analysis"


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'formatted_chat_history' not in st.session_state:
    st.session_state.formatted_chat_history = {}

chat_session_id = 'test_id'

def clear_chat_history(session_id):
    if session_id in st.session_state.chat_history:
        st.session_state.chat_history[session_id].clear()
    st.session_state.formatted_chat_history = {}

def display_chat_history(session_id):
    messages = st.session_state.formatted_chat_history.get(session_id, None)
    if messages is None:
        st.session_state.formatted_chat_history[session_id] = []
        intro_message = "Hi! I am a chatbot assistant with access to OSAA's publications. I am trained to find contradictions in analysis. Provide data points or analysis to me and I will see if it contradicts any previous publications!"
        st.chat_message("assistant").markdown(intro_message)
        st.session_state.formatted_chat_history[session_id].append({"role": "assistant", "content": intro_message})
    else:   
        for message in messages:
            st.chat_message(message["role"]).markdown(message["content"])

@st.cache_resource
def make_vectorstore():
    # create embedding model
    embedding_model = AzureOpenAIEmbeddings(
        model="smudataembed",
        api_key=st.secrets['azure'],
        azure_endpoint="https://openai-osaa-v2.openai.azure.com/",
    )

    # create DuckDB vectorstore
    conn = duckdb.connect(database=':memory:',
        config={
                "enable_external_access": "false",
                "autoinstall_known_extensions": "false",
                "autoload_known_extensions": "false"
            }
    )

    vectorstore = DuckDB(connection=conn, embedding=embedding_model)
    return vectorstore

@st.cache_resource
def get_splits():
    doc_path = "flagship report shortened.pdf"
    loader = PyPDFLoader(doc_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# title and introduction
st.title("OSAA SMU's Contradictory Analysis Tool")
st.markdown("The Contradictory Analysis Tool allows you to check if your analysis contradicts any previous analysis in OSAA's publications. This tool uses large language models with retrieval augmented generation and therefore may provide wrong answers.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

llm = AzureChatOpenAI(
    azure_deployment="gpt35osaa",
    api_key=st.secrets['azure'],
    azure_endpoint="https://openai-osaa-v2.openai.azure.com/",
    openai_api_version="2024-05-01-preview"
)

splits = get_splits()
vectorstore = make_vectorstore()
vectorstore.add_documents(splits)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant. Your task is to determine whether the provided analysis contradicts any context from our existing publications. If the analysis contradicts any context from our existing publications, clearly state that there is a contradiction. If there is no contradiction, state that there is no contradiction. If you are unsure, state that you are unsure."
        ),
        (
            "human",
            "Analysis: {analysis}."
        ),
        (
            "human",
            "Existing Publication Context: {context}."
        ),
        (
            "system",
            "Based on the Analysis and the Existing Publication Context, does the analysis contradict anything in the existing publications? Explain your reasoning and provide quotes demonstrating your reasoning if possible."
        )
    ]
)

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