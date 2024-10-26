from pinecone import ServerlessSpec,Pinecone
import os
from dotenv import load_dotenv # type: ignore


# Load environment variables
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')

# Initialize Pinecone with your API key and environment
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY")) # replace with your actual Pinecone API key


# Create or connect to an existing index in Pinecone
index_name = "conversational-embeddings"
# print(pc.list_indexes())
if not pc.has_index(index_name):
    pc.create_index(
                    name=index_name, 
                    dimension=3072,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                        ),
                    deletion_protection="disabled")  # using 1536 for OpenAI embeddings
index = pc.Index(index_name)
