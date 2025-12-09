import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import json

class DisplayResultStreamlit:
    def __init__(self,usecase,graph,user_message):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message
    
    def display_result_on_ui(self):
        usecase = self.usecase
        graph = self.graph
        user_message = self.user_message

        if usecase == "Basic Chatbot":
            
            initial_state = {
                "messages": [HumanMessage(content=user_message)]
            }

            for event in graph.stream(initial_state):
                for value in event.values():
                    with st.chat_message('user'):
                        st.write(user_message)
                    with st.chat_message("assistant"):
                        st.write(value["messages"][0].content)