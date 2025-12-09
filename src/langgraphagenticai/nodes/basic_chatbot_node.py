from src.langgraphagenticai.state.state import State

class BasicChatbotNode:
    """
    Basic Chatbot login implementation
    """
    def __init__(self,model):
        self.llm = model
    
    def process(self, state: State) -> dict:
        """
        Process the input state and generate a chatbot response.
        """
        # Extract the last user message text
        last_message = state["messages"][-1].content
        
        # Generate response
        response = self.llm.invoke(last_message)

        # Must return a dict with list of messages
        return {
            "messages": [response]
        }