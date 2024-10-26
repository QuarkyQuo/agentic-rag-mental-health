from langchain_core.graph import Graph
from langchain_core.agents import GraphAgent
from pathlib import Path
from langchain_core.prompts import PromptTemplate,MessagesPlaceholder,ChatPromptTemplate
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from lib.helpers import retrieve_similar, llm,model
from langchain.schema.output_parser import StrOutputParser
# Directory for storing chat history
history_dir = Path("./message_history")
history_dir.mkdir(parents=True, exist_ok=True)

# Retrieve session history
def get_session_history(session_id: str) -> FileChatMessageHistory:
    session_history_path = history_dir / f"{session_id}.json"
    return FileChatMessageHistory(str(session_history_path))

# Define prompt templates for different intents
casual_prompt = ChatPromptTemplate.from_messages(
    [
        ("You are a friendly conversational AI that talks in a friendly, regional English tone. Respond to '{input}' naturally and helpfully."),
        MessagesPlaceholder(variable_name="chat_history")
    ]
)

therapeutic_prompt = ChatPromptTemplate.from_messages(
    [
        ("You are a supportive assistant using '{framework}' to help users with mental health concerns. Respond to '{input}' empathetically and helpfully."),
        MessagesPlaceholder(variable_name="chat_history")
    ]
)

intent_prompt = ChatPromptTemplate.from_messages(
    [
        ("Determine if the following input is 'casual', 'mental_health', or 'general'. Input: '{input}'"),
        MessagesPlaceholder(variable_name="chat_history"),
        "Intent should be one word max. Something like 'casual', 'mental_health', or 'general'."
    ]
)

# Configure chains with RunnableWithMessageHistory
casual_chain_schema = casual_prompt | model | StrOutputParser()
casual_chain = RunnableWithMessageHistory(casual_chain_schema, get_session_history, input_messages_key="input", history_messages_key="chat_history")

therapeutic_chain_schema = therapeutic_prompt | model | StrOutputParser()
therapeutic_chain = RunnableWithMessageHistory(therapeutic_chain_schema, get_session_history, input_messages_key="input", history_messages_key="chat_history")

intent_chain_schema = intent_prompt | model | StrOutputParser()
intent_chain = RunnableWithMessageHistory(intent_chain_schema, get_session_history, input_messages_key="input", history_messages_key="chat_history")

# Define the nodes for Graph
nodes = {
    "intent_detection": intent_chain,
    "casual_response": casual_chain,
    "therapeutic_response": therapeutic_chain,
}

# Define edges for intent-based flow in the graph
edges = {
    # Routing to intent detection
    "start": {"intent_detection"},
    
    # Routing based on detected intent
    "intent_detection": {
        "casual": "casual_response",
        "mental_health": "therapeutic_response",
        "general": "casual_response",
    },
}

# Create the graph
graph = Graph(nodes=nodes, edges=edges, start_node="start")

# Initialize GraphAgent with history management
class ChatAgent:
    def __init__(self):
        self.graph_agent = GraphAgent(graph, get_session_history)

    def generate_response(self, user_input):
        """Generates a response using the GraphAgent based on LLM-determined intent."""
        session_id = "agent_chat_session"
        
        # Run the graph and get response based on input and session ID
        response = self.graph_agent.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return response

# Example usage
agent = ChatAgent()
user_input = "Hello! How's it going?"
response = agent.generate_response(user_input)
print(response)
