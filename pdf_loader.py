import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_and_split_pdf(pdf_path: str = "attention.pdf") -> List[Document]:
    """
    Load a PDF file and split its content into chunks.

    Args:
        pdf_path (str): Path to the PDF file. Defaults to "attention.pdf".

    Returns:
        List[Document]: List of document chunks.

    Raises:
        FileNotFoundError: If the PDF file is not found.
    """
    try:
        logger.info(f"Loading PDF from {pdf_path}")
        loader: PyPDFLoader = PyPDFLoader(pdf_path)
        docs: List[Document] = loader.load()
        logger.info(f"Loaded {len(docs)} pages from PDF")

        logger.info("Splitting documents into chunks")
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0
        )
        documents: List[Document] = text_splitter.split_documents(docs)
        logger.info(f"Split into {len(documents)} chunks")
        return documents
    except FileNotFoundError as e:
        logger.error(f"PDF file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise

if __name__ == "__main__":
    documents: List[Document] = load_and_split_pdf()
    logger.info(f"Preview of first 5 chunks: {documents[:5]}")