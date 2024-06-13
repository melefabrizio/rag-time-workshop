import asyncio
import os

from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.globals import set_debug

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


def ingest(sources: list[str], overwrite_db=False) -> VectorStoreRetriever:
    print("Ingesting sources...")
    chunks = load_files(sources)
    print("done")
    embeddings = BedrockEmbeddings(  # Utilizziamo AWS Bedrock per ottenere gli embedding
        model_id="cohere.embed-multilingual-v3",
    )
    print("Creating vector store...  ", end="")
    persist_directory = os.path.join(os.path.dirname(__file__), "../chroma_data")

    if overwrite_db:
        vector_store = Chroma.from_documents(
            chunks, embeddings,
            persist_directory=persist_directory,
            collection_name='boilerplate',
        )
    else:
        vector_store = Chroma(
            persist_directory=persist_directory,
            collection_name='boilerplate',
            embedding_function=embeddings
        )

    # TODO
    # Questo è un semplice VectorStoreRetriever che restituisce i primi 7 risultati.
    # Ci sono retriever più avanzati che sono utili in contesti più complessi.
    # Abbiamo delle alternative? https://python.langchain.com/v0.2/docs/how_to/#retrievers
    retriever = vector_store.as_retriever(search_kwargs={"k": 7})

    print("done")
    return retriever

# TODO:
# Questo prompt funziona, ma non è molto intelligente.
# Come si può migliorare? Si può adattare al nostro caso d'uso?
# Per esempio: c'è modo di recuperare la pagina del documento da cui è stata estratta la risposta?
# https://www.promptingguide.ai/it/introduction/tips
# https://smith.langchain.com/hub/
def prompt() -> PromptTemplate:
    prompt_template = """
    Answer the question using the informations in the context.
    
    Question: {question} 
    
    Context: {context} 
    
    Answer:
    """
    return PromptTemplate.from_template(prompt_template)


def prepare_chain(sources: list[str] = [], overwrite_db=False):
    retriever = ingest(sources, overwrite_db)
    llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")

    # TODO
    # Questa catena di esecuzione è molto semplice, ma può essere personalizzata.
    # Per esempio, può essere utile usare un altro llm+prompt per "preparare" la domanda al RAG,
    # oppure usare un llm per fare un parsing a valle della generazione.
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}  # input e retrieval
            | prompt()  # da input a prompt
            | llm  # da prompt a risultato
            | StrOutputParser()  # da risultato a stringa
    )
    return chain


async def run():
    overwrite_db = input("Overwrite database? (y/n) ").lower() == "y"
    chain = prepare_chain(['../sources/viaggiare_sicuri_uk.pdf'], overwrite_db)
    print("Chain ready!")

    while True:
        query = input("Ask a question: ")
        async for r in chain.astream(query):
            print(r, end="")
        print()


if __name__ == "__main__":
    load_dotenv()
    set_debug(False)
    asyncio.run(run())
