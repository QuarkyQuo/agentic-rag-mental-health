from agents import ChatAgent
from lib.helpers import feedback_loop

def chatbot():
    """Simulates a chatbot interface."""
    agent = ChatAgent()
    user_id = "sudheer"  # Unique identifier for each user session

    while True:
        user_input = input("You: ")
        
        # Generate response based on user input
        response = agent.generate_response(user_input)
        print("Bot:", response)
        
        # # Collect feedback
        # feedback = input("Was this helpful? (yes/no): ")
        # feedback_loop(response, user_id, "positive" if feedback == "yes" else "negative")


# Main function
if __name__ == '__main__':
    chatbot()

