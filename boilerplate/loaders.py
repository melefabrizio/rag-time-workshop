from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
    # TODO
    # Non è detto che il RecursiveCharacterTextSplitter sia sempre la scelta migliore.
    # In alcuni casi, potrebbe essere meglio usare un altro splitter.
    # https://python.langchain.com/v0.2/docs/how_to/#text-splitters
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
