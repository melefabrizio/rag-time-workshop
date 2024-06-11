
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.globals import set_debug
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader

load_dotenv()

# set_debug(True)

print("Preparing...")

# LOAD, SPLIT AND CHUNK


def load_text(file_path: str):
    with open(file_path, 'r') as f:
        text = f.read()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=100
    )
    text = text_splitter.create_documents([text])
    return text


def load_pdf(file_path: str):
    loader = PyMuPDFLoader(file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    return text_splitter.split_documents(loader.load())


def load_url(url: str):
    loader = WebBaseLoader(url)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    return text_splitter.split_documents(loader.load())


#source = 'https://gutenberg.org/cache/epub/59047/pg59047-images.html'
#chunks = load_url(source)

source = "../sources/artusi.txt"
chunks = load_text(source)

# EMBEDDING

embeddings = BedrockEmbeddings(  # Utilizziamo AWS Bedrock per ottenere gli embedding
    model_id="cohere.embed-multilingual-v3",
    model_kwargs={'input_type': "search_document"}
    # tipo di embedding generato, per cohere *deve* essere "search_document"
)

# FAISS è un vector store in-memory

vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"input_type": "search_query"})

# PROMPT

prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Do not answer using your own knowledge, only the context provided.

Question: {question} 

Context: {context} 

Answer:
"""
prompt = PromptTemplate.from_template(prompt_template)

# LET'S GO

llm = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

chain = (
        {"context": retriever, "question": RunnablePassthrough()}  # input e retrieval
        | prompt  # da input a prompt
        | llm  # da prompt a risultato
        | StrOutputParser()  # da risultato a stringa
)

while True:
    query = input("Ask a question: ")
    print(chain.invoke(query))
