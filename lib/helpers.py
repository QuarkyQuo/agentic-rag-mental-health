# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Pinecone
# from openai.embeddings_utils import cosine_similarity
from dotenv import load_dotenv
from db.pinecone_client import index
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
import os


# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Initialize LLM
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# Initialize the embeddings model
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-3-large")

def embed_text(text):
    """Generate embeddings for a text snippet using OpenAI embeddings."""
    return embeddings.embed_query(text)

# def detect_intent(user_input):
#     """Detects if input is a casual conversation or mental health-related query."""
#     casual_phrases = ["hello", "hi", "what's up", "how are you"]
#     mental_health_keywords = ["anxiety", "depression", "stress", "sad"]

#     # Simple keyword matching approach
#     if any(keyword in user_input.lower() for keyword in casual_phrases):
#         return "casual"
#     elif any(keyword in user_input.lower() for keyword in mental_health_keywords):
#         return "mental_health"
#     return "general"

def store_embedding(text, metadata):
    """Stores the embedding with metadata in Pinecone."""
    vector = embed_text(text)
    index.upsert([(metadata['id'], vector, metadata)])

def retrieve_similar(query, top_k=5):
    """Retrieve top_k similar embeddings from Pinecone for the query."""
    query_embedding = embed_text(query)
    return index.query(namespace="ns1",vector=query_embedding, top_k=top_k)

user_feedback = {}

def feedback_loop(response, user_id, feedback):
    """Stores feedback and adapts the agent based on feedback."""
    user_feedback[user_id] = feedback

    # Adjust conversational flow based on feedback
    if feedback == "positive":
        print("Great! Glad you found it helpful.")
    elif feedback == "negative":
        print("Thanks for the feedback. I'll adjust my approach.")
