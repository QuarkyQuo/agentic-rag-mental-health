import os
from pathlib import Path
from langchain_core.prompts import PromptTemplate,MessagesPlaceholder,ChatPromptTemplate
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from lib.helpers import retrieve_similar, llm,model
from langchain.schema.output_parser import StrOutputParser

# Directory for storing chat history
history_dir = Path("./message_history")
history_dir.mkdir(parents=True, exist_ok=True)

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

# Configure RunnableWithMessageHistory with prompt templates and message history
casual_chain_schema= casual_prompt | model | StrOutputParser()
casual_chain = RunnableWithMessageHistory(casual_chain_schema, get_session_history, input_messages_key="input", history_messages_key="chat_history")

therapeutic_chain_schema = therapeutic_prompt | model | StrOutputParser()
therapeutic_chain = RunnableWithMessageHistory(therapeutic_chain_schema, get_session_history, input_messages_key="input", history_messages_key="chat_history")

intent_chain_schema = intent_prompt | model | StrOutputParser()
intent_chain = RunnableWithMessageHistory(intent_chain_schema, get_session_history, input_messages_key="input", history_messages_key="chat_history")

# Define ChatAgent class for managing different conversation intents
class ChatAgent:
    def __init__(self):
        self.casual_chain = casual_chain
        self.therapeutic_chain = therapeutic_chain
        self.intent_chain = intent_chain
        
# conv_chain.invoke(augmented_query, config={"configurable": {"session_id": session_id}})

    def detect_intent_with_llm(self, user_input):
        """Categorizes intent as 'casual', 'mental_health', or 'general'."""
        response = self.intent_chain.invoke({"input":user_input},config={"configurable": {"session_id": "agent_chat_session"}})
        response = response.strip().lower()
        return response if response in ["casual", "mental_health", "general"] else "general"

    def casual_response(self, user_input):
        """Handles casual conversation with a friendly tone."""
        return self.casual_chain.invoke({"input":user_input},config={"configurable": {"session_id": "agent_chat_session"}})

    def therapeutic_response(self, user_input, framework="CBT"):
        """Handles mental health queries with a specified framework."""
        return self.therapeutic_chain.invoke({"input":user_input,"framework":framework},config={"configurable": {"session_id": "agent_chat_session"}})

    def generate_response(self, user_input):
        """Generates response based on LLM-categorized intent."""
        intent = self.detect_intent_with_llm(user_input)
        
        if intent == "casual":
            return self.casual_response(user_input)
        elif intent == "mental_health":
            retrieved_docs = retrieve_similar(user_input, top_k=3)
            response = [match['metadata']['text'] for match in retrieved_docs['matches'] if 'metadata' in match]
            augmented_query=f"query:{user_input},context:{response}"
            return self.therapeutic_response(augmented_query)
        else:
            return self.casual_response(user_input)

# Example usage
agent = ChatAgent()
user_input = "Hello! How's it going?"
response = agent.generate_response(user_input)
print(response)
