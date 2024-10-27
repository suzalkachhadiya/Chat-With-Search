import streamlit as st

from langchain_groq import ChatGroq
from langchain.tools import ArxivQueryRun,WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.document_loaders import PyMuPDFLoader
from langchain.tools import Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import os
import uuid

from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_amx=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_amx=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="search")

st.title("Langchain - Chat with search")

st.sidebar.title("Settings")
api_key=st.sidebar.text_input("enter your groq api key:",type="password")
uploaded_files=st.sidebar.file_uploader("choose a PDF file",type="pdf",accept_multiple_files=True)

if uploaded_files:
    documents=[]
    for uploaded_file in uploaded_files:
        temp_pdf=f"./assets/temp.pdf"
        with open(temp_pdf,"wb") as f:
            f.write(uploaded_file.getvalue())
            file_name=uploaded_file.name

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"hii, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="what is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)

    def create_pdf_retriever(folder_path: str):
        documents = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(folder_path, file_name)
                loader = PyMuPDFLoader(pdf_path)
                docs=loader.load()  # Load all documents from PDFs

                for idx, doc in enumerate(docs):
                    doc.metadata["id"] = f"{file_name}_{idx}_{uuid.uuid4()}"
                    documents.append(doc)

        # Step 2: Split documents into manageable chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        # Step 3: Create embeddings and store in Chroma vector database
        embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
        vector_store = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)
        return vector_store.as_retriever()

    # Step 4: Initialize PDF retriever tool
    pdf_retriever = create_pdf_retriever("C:/DataScience/Gen_AI/Tools_agents/assets")  

    pdf_tool = Tool(
        name="PDFRetriever",
        func=lambda query: "\n\n".join([doc.page_content for doc in pdf_retriever.get_relevant_documents(query)]),
        description="Tool for searching PDF files using a vector database of content embeddings"
    )

    tools=[search,arxiv,wiki,pdf_tool]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_error=True)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)

        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])

        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)