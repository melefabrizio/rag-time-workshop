from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

from boilerplate.loaders import load_url, load_pdf, load_text

from dotenv import load_dotenv

def load_files(sources: list[str]) -> list[Document]:
    chunks = []
    for source in sources:
        if source.startswith("http"):
            print("Loading from URL " + source)
            chunks += load_url(source)
        elif source.endswith(".pdf"):
            print("Loading from PDF " + source)
            chunks += load_pdf(source)
        else:
            print("Loading as text file " + source)
            chunks += load_text(source)
    return chunks


def ingest(sources: list[str]) -> VectorStoreRetriever:
    print("Ingesting sources...", end="")
    chunks = load_files(sources)
    print("done")
    embeddings = BedrockEmbeddings(  # Utilizziamo AWS Bedrock per ottenere gli embedding
        model_id="cohere.embed-multilingual-v3",
        model_kwargs={'input_type': "search_document"}
    )
    print("Creating vector store...", end="")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"input_type": "search_query"})
    # print on the previous line
    print("done")
    return retriever


def prompt() -> PromptTemplate:
    prompt_template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. Do not answer using your own knowledge, only the context provided.
    
    Question: {question} 
    
    Context: {context} 
    
    Answer:
    """
    return PromptTemplate.from_template(prompt_template)

def prepare_chain(sources: list[str] = []):
    retriever = ingest(sources)
    llm = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

    chain = (
            {"context": retriever, "question": RunnablePassthrough()}  # input e retrieval
            | prompt()  # da input a prompt
            | llm  # da prompt a risultato
            | StrOutputParser()  # da risultato a stringa
    )
    return chain


def run():
    chain = prepare_chain(['../sources/artusi.txt'])
    print("Chain ready!")

    while True:
        query = input("Ask a question: ")
        print(chain.invoke(query))


if __name__ == "__main__":
    load_dotenv()
    run()