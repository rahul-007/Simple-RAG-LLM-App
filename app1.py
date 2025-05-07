import streamlit as st
import PyPDF2
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

st.title("Simple RAG Chatbot using Hugging Face")

st.sidebar.write("Input needed before getting started")
hf_api_key = st.sidebar.text_input("Enter your Hugging Face API key", type = "password")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

if st.sidebar.button("Start Ingesting") and hf_api_key and uploaded_file:

    def extract_text_from_pdf(file):
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    def create_vector_embeddings(text):
        st.session_state.embeddings = HuggingFaceInferenceAPIEmbeddings(
                model_name="intfloat/multilingual-e5-large-instruct",
                api_key=hf_api_key,
                api_url = "https://api-inference.huggingface.co/models/intfloat/multilingual-e5-large-instruct"
                )
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
        st.session_state.final_docs = st.session_state.text_splitter.create_documents([text])
        print(st.session_state.embeddings)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)

    with st.spinner("Indexing the documents..."):
        text = extract_text_from_pdf(uploaded_file)
        create_vector_embeddings(text)
        st.write("Document Ingestion Complete")
elif not (hf_api_key and uploaded_file):
    st.warning("Please upload PDF file and API Key before getting started")


template = """Answer the question on the provided context only.
Please provide the most accurate response to the question.
<context>
{context}
<context>

Question: {input}
"""

prompt = ChatPromptTemplate.from_template(template)

user_input = st.text_input("Ask any question....")
if user_input and hf_api_key and uploaded_file:
    llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta", 
        max_new_tokens=300,temperature=0.7,
        huggingfacehub_api_token=hf_api_key,
        task="text-generation"
        )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    response = retrieval_chain.invoke({'input': user_input})
    st.success(response['answer']) 

