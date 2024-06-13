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

from workshop.loaders import load_url, load_pdf, load_text

from dotenv import load_dotenv


# Questo funziona, non c'è bisogno di cambiarlo
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
    # In caso di problemi di ingestion elimina la directory
    # e seleziona l'overwrite_db al momento dell'avvio
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
# Questo prompt non è un prompt. È più un template per un prompt.
# Prova a crearne uno reale e funzionante.
# https://www.promptingguide.ai/it/introduction/tips
# https://smith.langchain.com/hub/
def prompt() -> PromptTemplate:
    prompt_template = """
    Question: {question} 
    
    Context: {context} 
    
    Answer:
    """
    return PromptTemplate.from_template(prompt_template)


def prepare_chain(sources: list[str] = [], overwrite_db=False):
    retriever = ingest(sources, overwrite_db)
    llm = ChatBedrock(
        # Ci sono tanti modelli da provare, consulta il readme per vedere quelli disponibili
        # Scegliere il modello giusto è importante, e può fare la differenza.
        model_id="meta.llama2-70b-chat-v1",
        # Temperatura 0.5? forse è troppo, o forse è troppo poco.
        # È un parametro su cui si può giocare
        model_kwargs={"temperature": 0.5}
    )

    # TODO: Crea la chain. Quella della demo è un buon punto di partenza
    # ma ci sono molte cose che possono essere migliorate.
    # Si possono inserire degli step di pre o di post processing,
    # per esempio per rielaborare il contesto prima di passarlo al retriever,
    # oppure formattare il risultato, sempre usando un llm.
    chain = ()
    return chain


async def run():
    overwrite_db = input("Overwrite database? (y/n) ").lower() == "y"
    chain = prepare_chain(['../sources/viaggiare_sicuri_uk.pdf'], overwrite_db)
    print("Chain ready!")

    while True:
        query = input("Ask a question: ")
        print(chain.invoke(query))


if __name__ == "__main__":
    load_dotenv()
    set_debug(False)  # FIXME: Settato a True è utile per capire cosa succede dietro le quinte
    asyncio.run(run())
