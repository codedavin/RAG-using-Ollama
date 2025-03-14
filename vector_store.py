import logging
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def create_vector_store(documents: List[Document]) -> Chroma:
    """
    Create a Chroma vector store from a list of documents.

    Args:
        documents (List[Document]): List of document chunks to embed.

    Returns:
        Chroma: Initialized Chroma vector store with embedded documents.

    Raises:
        Exception: If embedding or vector store creation fails.
    """
    try:
        logger.info("Extracting text from documents")
        text: List[str] = [doc.page_content for doc in documents]
        
        logger.info("Loading Hugging Face embedding model")
        embedding_model: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        logger.info("Creating Chroma vector store")
        db: Chroma = Chroma.from_texts(texts=text, embedding=embedding_model)
        logger.info(f"Vector store created with {len(db)} documents")
        return db
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise

if __name__ == "__main__":
    from pdf_loader import load_and_split_pdf
    documents: List[Document] = load_and_split_pdf()
    db: Chroma = create_vector_store(documents)