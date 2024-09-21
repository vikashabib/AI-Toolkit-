import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly as ply
import bs4
from streamlit_option_menu import option_menu
from langchain.vectorstores.faiss import FAISS
from langchain_community.document_loaders import TextLoader,PyPDFDirectoryLoader,PyPDFLoader,WebBaseLoader,WikipediaLoader,ArxivLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain.agents import AgentExecutor,AgentType,initialize_agent
from streamlit_js_eval import streamlit_js_eval
from langchain.callbacks import StreamlitCallbackHandler
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

os.environ['GROQ_API_KEY'] = st.secrets["general"]["GROQ_API_KEY"]
os.environ['HF_TOKEN'] = st.secrets["general"]["HF_TOKEN"]
os.environ['LANGCHAIN_API_KEY'] = st.secrets["general"]["LANGCHAIN_API_KEY"]
os.environ['LANGCHAIN_TRACING_V2'] = 'true'


st.set_page_config("Langchain AI App",page_icon="🐦")


##########################################################################################################################################
llm = ChatGroq(model="Gemma-7b-It")
rag_template = ChatPromptTemplate.from_template(
    """
    Answer all the questions accurately!.Please read the context from
    the file and give answer accurately.
    <context>
    {context}
    </context>
    
    Question:{input}
"""
)
#-----------------------------------------------------------RAG-------------------------------------------------------------------------------------
def retrieval_augmented_generation():
    
    def delete_files_directory(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory,file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    def vector_store_query():
        loader = PyPDFDirectoryLoader(dir_name)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=500)
        text_splitted = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings()
        vectors = FAISS.from_documents(embedding=embeddings,documents=text_splitted)
        return vectors
    
    
    
    
    st.markdown(
        """
        <style>
        .small-font-header {
            font-size: 24px;
        }
        .small-font-markdown {
            font-size: 14px;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # Use CSS classes for smaller font sizes
    st.markdown('<h1 class="small-font-header">🐦Langchain: RAG DOCUMENT QUERY AI APP</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="small-font-header">Note</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="small-font-markdown">
        This is a document query AI app. Please upload a PDF file to start.
        Once uploaded, you can enter a query, and the app will provide answers based on the document content.
        Use the **"Reset Files"** button to clear the uploaded files and start fresh.
        You can select multiple files to get context data from your files😉!
        </p>
        """, 
        unsafe_allow_html=True
        )

    st.markdown(
        """
        <p class="small-font-markdown">
        Please manually **cancel** the uploaded file or click on **Browse Files** to upload new files!       
        </p>
        """, 
        unsafe_allow_html=True
    )
    dir_name = 'trial'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    uploaded_file = st.file_uploader(label="Please Upload a File",accept_multiple_files=True)
    if 'vectors' not in st.session_state:
        st.session_state.vectors = None
    if 'count' not in st.session_state:
        st.session_state.count = 0
    if uploaded_file:
        
        for file in uploaded_file:
            with open(os.path.join(dir_name,file.name),'wb') as f:
                f.write(file.getbuffer())
                
        if st.session_state.count == 0 or st.session_state.vectors is None:
            st.session_state.vectors = vector_store_query()
            st.session_state.count += 1
            st.write("Vectors Embedded Successfully")
            
        user_prompt = st.text_input("Ask me your query")
        if st.button("Generate Response"):
            if user_prompt:
                try:
                    document_chain = create_stuff_documents_chain(llm=llm,prompt=rag_template)
                    retriever = st.session_state.vectors.as_retriever()
                    retrieval_chain = create_retrieval_chain(retriever,document_chain)
                    relevants_docs = retriever.get_relevant_documents(user_prompt)[:5]
                    context = "\n".join([docs.page_content for docs in relevants_docs])
                    response = retrieval_chain.invoke({'input':user_prompt,'context':context})
                    st.success(response["answer"])
                    st.toast("Response generated!",icon="🎉")
                    
                    with st.expander("Document Similarity Search"):
                        for i,doc in enumerate(response['context']):
                            st.write(doc.page_content)
                            st.write("-------------------------")
                        
                except Exception as e:
                    st.error(f"Exception is:{e}")
        if st.button("Reload page"):
            streamlit_js_eval(js_expressions="parent.window.location.reload()")      
        if st.button("Reset Files"):
            st.session_state.vectors = None
            st.session_state.count = 0
            delete_files_directory(dir_name)
            st.success("Deleted all files from the cache")
            
 #--------------------------------------------------Chat With SQL-------------------------------------------------------------------------------------         
def chat_with_sql():
    LOCALDB = "USE_LOCALDB"
    MYSQL ="USE_MYSQL"
    st.title("🐦Langchain: Chat With SQL")
    col1,col2 = st.columns([1,1])
    with col1:
        my_sqlhost = st.text_input("Provide your  localhost")
        my_sqlpass = st.text_input("Provide your SQL Password")
    with col2:
        my_sqlusername = st.text_input("Provide your SQL Username")
        my_sqldb =st.text_input("Provide your Database Name")
        
    if not (my_sqlhost and my_sqlusername and my_sqlpass and my_sqldb):
        st.error("Please fill all the details")
        st.stop()
        
    try:
        from langchain.agents import create_sql_agent
        from langchain.sql_database import SQLDatabase
        from langchain.agents.agent_types import AgentType
        from langchain.agents.agent_toolkits import SQLDatabaseToolkit
        from sqlalchemy import create_engine
        from sqlalchemy.exc import SQLAlchemyError
        
        engine = create_engine(f"mysql+mysqlconnector://{my_sqlusername}:{my_sqlpass}@{my_sqlhost}/{my_sqldb}")
        db = SQLDatabase(engine)
        llm = ChatGroq(model="Gemma2-9b-It")
        toolkit = SQLDatabaseToolkit(db=db,llm=llm)
        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            agent_type= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_executor_kwargs={'handle_parsing_errors':True},
            verbose =True
        )
        st.toast("Database Connected!",icon="🛅")
        if 'messages' not in st.session_state or st.sidebar.button("Clear Message History"):
            st.session_state.messages = [{"role":"assistant","content":"Hey👋 I am Your SQL friend. How can I help you?"}]
        
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
        user_prompt = st.chat_input(placeholder="Ask me your query")
        
        if user_prompt:
            st.session_state.messages.append({"role":"user","content":user_prompt})
            st.chat_message("user").write(user_prompt)
            
            with st.chat_message("assistant"):
                stc = StreamlitCallbackHandler(st.container())
                response = agent.run(user_prompt,callbacks=[stc])
                st.session_state.messages.append({"role":"assistant","content":response})
                st.success(response)
        
    except Exception as e:
        st.error(f"Exception error is:{e}")
 #--------------------------------------------------ChatBot-------------------------------------------------------------------------------------         
def chatbot():
    st.title("🐦Langchain: ChatBot")
    llm = ChatGroq(model="Gemma2-9b-It")
    output_parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system","You are a helpul assitant. Read Carefully and provide accurate answers!"),
            ("user","Question:{user_prompt}")
        ]
    )
    if 'chats' not in st.session_state:
        st.session_state.chats=[{'role':'assistant','content':"Hey User, How can I help you?"}]
        
    for msg in st.session_state.chats:
        st.chat_message(msg["role"]).write(msg["content"])
        
    user_prompt = st.chat_input(placeholder="Ask me your query")
    if user_prompt:
        st.session_state.chats.append({'role':'user','content':user_prompt})
        st.chat_message("user").write(user_prompt)
        
        with st.chat_message("assistant"):
            chain = prompt|llm|output_parser
            
            response = chain.invoke({'user_prompt':user_prompt})
            st.session_state.chats.append({'role':'assistant','content':response})
            st.success(response)
    
    
    pass
def url_summarization():
    from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
    from langchain_core.prompts import PromptTemplate
    import validators
    from langchain.chains.summarize import load_summarize_chain
    
    llm = ChatGroq(model="Gemma-7b-It")
    st.title("🐦Langchain: URL Text Summarization")
    st.markdown(
        """
        <style>
        .small-font-header {
            font-size: 24px;
        }
        .small-font-markdown {
            font-size: 18px;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    st.markdown('<h2 class="small-font-header">Note</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="small-font-markdown">
        Please Select **English** Language Youtube Video URLS for Summarization😉!
        </p>
        """, 
        unsafe_allow_html=True
        )
    generic_url = st.text_input(label="Provide Url for summarization")
    template = """
    Please Summarize this URL accurately. If any error of Language occur.Please tell it does not support this language
    Content:{text}
    """
    prompt= PromptTemplate(
        template=template,input_variables=["text"]
    )
    if st.button("Summarize Content"):
        with st.spinner("Generating Summary"):
            if not generic_url:
                st.error("Please Provide URL to summarize")
            elif not validators.url(generic_url):
                st.error("Please provide valid URL")
            else:
                try:
                    if 'youtube.com' in generic_url:
                        loader = YoutubeLoader.from_youtube_url(generic_url)
                    else:
                        loader = UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    documents = loader.load()
                    chain = load_summarize_chain(
                        llm=llm,
                        prompt=prompt,
                        chain_type="stuff",
                        verbose=True
                    )
                    output_summary = chain.run(documents)
                    st.success(output_summary)
                except Exception as e:
                    st.error(f"Exception is: {e}")  
    
    pass

#--------------------------------------------------LOGIN-------------------------------------------------------------------------------------
with st.sidebar:
    username = st.text_input(label="Username")
    password = st.text_input(label='Password',type="password")
if username=="admin" and password=="admin":
    with st.sidebar:
        selected = option_menu(
            menu_title="GEN AI",
            options=["RAG QUERY","Chat With SQL","ChatBot","YTube URL Summarize"],
            default_index=0,
            styles={
                "container": {
                "background-color": " #5F5F6E",  # Light grey background
                "padding": "5px 10px",         # Padding around the container
                "border-radius": "10px",       # Rounded corners
                "box-shadow": "0px 4px 10px rgba(0, 0, 0, 0.1)",  # Light shadow for depth
            },
                "nav-link":{
                    "font-size":"13px",
                    "text-align":"center",
                    "--hover-color":"#FFFFFF"

                }
            }
        )
    
    if selected=="RAG QUERY":
        retrieval_augmented_generation()
    elif selected == "Chat With SQL":
        st.session_state.messages=None
        chat_with_sql()
    elif selected=="ChatBot":
        chatbot()
    elif selected=="YTube URL Summarize":
        url_summarization()
    else:
        st.write("Select Option")
else:
    if username!="" and password!="":
        if username !="admin" and password=="admin":
            st.toast("Invalid Username",icon="🚫")
        elif username =="admin" and password!="admin":
            st.toast("Invalid Password",icon="🚫")
        else:
            st.toast("Invalid Username and Password",icon="🚫")
            
#-------------------------------------------------------------------------------------------------------------------------------------------

