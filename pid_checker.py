import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pypdf import PdfReader
from docx import Document
import textract

# Function to extract text from .pdf file
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from .docx file
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    # Extract paragraphs
    full_text = [para.text for para in doc.paragraphs]
    
    # Extract text from tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                full_text.append(cell.text)

    return "\n".join(full_text)

# function to extract text from .doc file
def extract_text_from_doc(doc_file):
    text = textract.process(doc_file)
    return text.decode("utf-8")

# title and introduction
st.title("OSAA SMU's Check PID Tool")
st.markdown("Use the PID Checker to upload a PID and see if it aligns with the PID criteria.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("")

# initiatlize model
llm = AzureChatOpenAI(
    azure_deployment="osaagpt32k",
    api_key=st.secrets['azure'],
    azure_endpoint="https://openai-osaa-v2.openai.azure.com/",
    openai_api_version="2024-05-01-preview"
)

# chat prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant. Your task is to determine whether the uploaded document, called a PID, aligns with criteria. If the document aligns with the criteria, clearly state that it does. If the document does not align with the criteria, cleary state that it does not and state the reason(s). If you are unsure, state that you are unsure."
        ),
        (
            "human",
            "uploaded document (PID): {document}."
        ),
        (
            "human",
            "criteria: \n\n"
            "**Coherence with Thematic Guidance and Priorities:**"
            "1. Alignment with Priorities: Does the PID clearly align with the USGs thematic guidance and priorities?"
            "2. Relevance to Clusters: How well does the PID address the specific needs and objectives of the relevant thematic cluster(s)?"
            "3. Consistency: Are the proposed activities, outputs, outcomes, and objectives consistent with the overall strategic guidance? \n"
            "**Application of Value Chain:**"
            "4. Value Chain Integration: Does the PID effectively integrate a value chain approach to achieve the desired outcomes and objectives (i.e., sum of deliverables leads to outputs, sum of outputs leads to outcomes, etc.)?"
            "5. Outcome Focus: Are the outcomes and objectives clearly defined and achievable through the proposed value chain?"
            "6. Activity-Outcome Linkage: How well do the proposed activities link to the expected outputs and outcomes?"
            "**Intentionality and Target Audience:**"
            "7. Intentionality: Is there a clear intention behind each proposed deliverable based on OSAA's deliverable definitions, and does it align with the overall objectives?"
            "8. Target Audience: Does the PID identify and address the needs of the target audience effectively?"
            "**Cross-Functional Collaboration:**"
            "9. Collaborative Efforts: Does the PID outline clear strategies for cross-functional collaboration (policy/monitoring-coordination-advocacy-advisory)?"
            "10. Cross-functional Coordination: How well does the PID facilitate coordination between different teams and functions?"
            "**Delivery of Activities (deliverables table):**"
            "11. Role Clarity: Are the roles and responsibilities of different teams and clusters leads clearly defined?"
            "12. Activity Planning: Are the planned activities (policy documents/webinars, advocacy documents/webinars, coordination function, intergovernmental function, advisory function) well-structured and feasible, and are they intentional in targeting a specific audience as per the proposed value chain? I.e., not just a sum of parts but well-structured and interlinked."
            "13. Costing of Activities: Are the costs associated with each activity included?"
            "14. Timeline and Milestones: Does the PID include a realistic timeline with clear start and end times and milestones for each activity?"
            "**Monitoring and Evaluation:**"
            "15. Performance Indicators: Are there clear performance indicators to measure success?"
            "**Gender mainstreaming, disability inclusion, multilingualism:**"
            "16. Does the PID clearly outline how these will be addressed"
        )
    ]
)


# upload PID
st.markdown("#### Upload a PID")
uploaded_file = st.file_uploader("Upload a PID", type=['pdf', 'doc', 'docx'], label_visibility="collapsed")
if uploaded_file is not None:

    # process file based on type
    if uploaded_file.type == "application/pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        extracted_text = extract_text_from_docx(uploaded_file)
    elif uploaded_file.type == "application/msword":
        extracted_text = extract_text_from_doc(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a DOC, DOCX, or PDF file type.")
        extracted_text = None

    

chain = (
    {"document": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if st.button("Check PID", use_container_width=True, type="primary"):

    with st.spinner("checking PID..."):
        response = chain.invoke(extracted_text)

    st.markdown("#### Evaluation")
    st.write(response)

    # if extracted_text:
    #     st.text_area("###### Extracted Text:\n\n", extracted_text, height=300)