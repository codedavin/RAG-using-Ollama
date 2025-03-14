import logging
from typing import List
from pdf_loader import load_and_split_pdf
from vector_store import create_vector_store
from query_processor import process_query
from langchain.schema import Document
from langchain.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Main function to load a PDF, create a vector store, and process a query.

    Raises:
        Exception: If any step in the pipeline fails.
    """
    try:
        # Load and split PDF
        documents: List[Document] = load_and_split_pdf("attention.pdf")
        
        # Create vector store
        db: Chroma = create_vector_store(documents)
        
        # Process query
        query: str = "Can you explain me about the attention in transformers"
        response: str = process_query(db, query)
        logger.info(f"Final response: {response}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()