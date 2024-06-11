
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


# LOAD, SPLIT AND CHUNK


def load_data(file_path: str):
    with open(file_path, 'r') as f:
        text = f.read()
    return text


def split_data(data_to_split):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )
    return text_splitter.create_documents([data_to_split])


source = '../sources/de_bello_gallico.txt'
data = load_data(source)
chunks = split_data(data)

# EMBEDDING

embeddings = BedrockEmbeddings(  # Utilizziamo AWS Bedrock per ottenere gli embedding
    model_id="cohere.embed-multilingual-v3",
    model_kwargs={'input_type': "search_document"}
    # tipo di embedding generato, per cohere *deve* essere "search_document"
)

# FAISS è un vector store in-memory che permette di effettuare ricerche di similarità tra vettori
vector_store = FAISS.from_documents(chunks, embeddings)

# RICERCA E RISPOSTA

prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:
"""

query = "Chi erano gli allobrogi?"

search_results = vector_store.search(query, k=5, search_type="similarity", input_type="search_query")

prompt = PromptTemplate.from_template(prompt_template)

llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")

context = "\n".join([result.page_content for result in search_results])


def return_search(_question: str):
    return context


chain = (
        {"context": return_search, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

print(chain.invoke(query))
