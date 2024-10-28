import streamlit as st
from langchain_groq import ChatGroq
from langchain.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.document_loaders import PyMuPDFLoader
from langchain.tools import Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
import uuid
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
print(list(AgentType))
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

def create_pdf_tool(folder_path: str):
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

        # Step 4: Initialize PDF retriever tool
        pdf_retriever = vector_store.as_retriever()  

        pdf_tool = Tool(
            name="PDFRetriever",
            func=lambda query: "\n\n".join([doc.page_content for doc in pdf_retriever.get_relevant_documents(query)]),
            description="Tool for searching PDF files using a vector database of content embeddings"
        )
        return pdf_tool

def get_tools_for_agent(agent_type, selected_tools):
    """Configure tools based on agent type and selection"""
    tools = []
    
    if agent_type == AgentType.SELF_ASK_WITH_SEARCH:
        # For SELF_ASK_WITH_SEARCH, only use DuckDuckGo as the search tool
        return [DuckDuckGoSearchRun(name="search")]
    
    # For other agent types, add selected tools
    if selected_tools.get('Arxiv'):
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        tools.append(arxiv)
    
    if selected_tools.get('Wikipedia'):
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
        tools.append(wiki)
    
    if selected_tools.get('Search'):
        search_tool = DuckDuckGoSearchRun()
        tools_1 = [
        Tool(
            name="Search",  # This must be exactly "Search" for SelfAskWithSearch
            func=search_tool.run,
            description="Useful for searching the internet to find answers to questions"
        ),
        Tool(
            name="Intermediate Answer",  # This must be exactly "Intermediate Answer"
            func=lambda x: x,  # Simple pass-through function
            description="Used to record intermediate answers in a chain of reasoning"
        )
    ]
        tools.extend(tools_1)
    
    if selected_tools.get('Documents'):
        pdf_tool = create_pdf_tool(folder_path="C:/DataScience/Gen_AI/Tools_agents/assets")  # Create this function based on your existing PDF tool code
        tools.append(pdf_tool)
    
    return tools

def create_agent_instance(llm, agent_type, selected_tools):
    """Create an agent instance with appropriate tools"""
    tools = get_tools_for_agent(agent_type, selected_tools)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return initialize_agent(
        tools,
        llm,
        agent=agent_type,
        memory=memory,
        handling_parsing_errors=True
    )

def main():
    st.set_page_config(layout="wide")
    st.title("Langchain - Chat with search")
    
    # Sidebar settings
    st.sidebar.title("Settings")
    api_key = st.sidebar.text_input("Enter your groq api key:", type="password")

    uploaded_files=st.sidebar.file_uploader("choose a PDF file",type="pdf",accept_multiple_files=True)

    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temp_pdf=f"./assets/temp.pdf"
            with open(temp_pdf,"wb") as f:
                f.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
    
    col1, col2=st.columns(2)
    with col1:
        # Tool selection
        popover = st.popover("Tools you want to go through for searching:")
        selected_tools = {
            'Documents': popover.checkbox("Search through PDFs", True),
            'Arxiv': popover.checkbox("Search through arxiv", True),
            'Wikipedia': popover.checkbox("Search through Wikipedia", True),
            'Search': popover.checkbox("Search through DuckDuckGo", True)
        }
    with col2:
        # Agent selection
        popover = st.popover("Agent type ")
        agent_types = {
            'ZERO_SHOT': popover.checkbox("ZERO_SHOT_REACT_DESCRIPTION", True),
            'CONVERSATIONAL_REACT': popover.checkbox("AgentType.CONVERSATIONAL_REACT_DESCRIPTION",True),
            'STRUCTURE_ZERO_SHOT': popover.checkbox("STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION", True)
        }
        
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
        ]
    
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input(placeholder="What is machine learning?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="Llama3-8b-8192",
            streaming=True
        )
        
        # Create columns based on selected agents
        selected_agents = [
            (name, agent_type) for name, agent_type in [
                ('ZERO_SHOT', AgentType.ZERO_SHOT_REACT_DESCRIPTION),
                ('STRUCTURE_ZERO_SHOT', AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION),
                ('CONVERSATIONAL_REACT',AgentType.CONVERSATIONAL_REACT_DESCRIPTION)
            ] if agent_types[name]
        ]
        
        if selected_agents:
            cols = st.columns(len(selected_agents))
            
            for col, (name, agent_type) in zip(cols, selected_agents):
                with col:
                    st.write(f"Agent: {name}")
                    with st.chat_message("assistant"):
                        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                        
                        agent = create_agent_instance(llm, agent_type, selected_tools)
                        response = agent.run(st.session_state.messages, callbacks=[st_cb])
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.write(response)

if __name__ == "__main__":
    main()