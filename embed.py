from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from lib.helpers import embeddings
from db.pinecone_client import index

# Load and chunk PDF using PyPDFLoader
def load_and_chunk_pdf(file_path):
    """Load text from a PDF using PyPDFLoader and split it into chunks."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()  # Load pages as separate documents

    # Extract text and split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Choose an appropriate chunk size
        chunk_overlap=50
    )
    chunks = []
    for page in pages:
        page_chunks = text_splitter.split_text(page.page_content)
        chunks.extend(page_chunks)
    return chunks

# Store embeddings in Pinecone
def store_embeddings(chunks,collection):
    """Generate embeddings for each chunk and store them in Pinecone."""
    for i, chunk in enumerate(chunks):
        # Generate embedding for the chunk
        embedding = embeddings.embed_query(chunk)
        
        # Store in Pinecone with unique ID
        metadata = {"id": f"{collection}_chunk_{i}", "text": chunk}
        index.upsert([(metadata['id'], embedding, metadata)])

# Main function
def process_pdf(file_path):
    """Load PDF, chunk text, and store embeddings."""
    chunks = load_and_chunk_pdf(file_path)
    store_embeddings(chunks,"problem-statements")

# Usage example
process_pdf('./data/problem_statements.pdf')
