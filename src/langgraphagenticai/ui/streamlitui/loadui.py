import streamlit as st
import os

from src.langgraphagenticai.ui.uiconfigfile import Config

class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}
    
    def load_streamlit_ui(self):
        st.set_page_config(page_title=str(self.config.get_page_title()),layout="wide")
        st.header(self.config.get_page_title())

        with st.sidebar:
            llm_options = self.config.get_llm_options()
            usecase_options = self.config.get_usecase_options()

            self.user_controls['selected_llm'] = st.selectbox("Select LLM",llm_options)

            if self.user_controls['selected_llm'] == 'Groq':
                model_options = self.config.get_groq_model_options()
                self.user_controls['selected_groq_model'] = st.selectbox("Select Model",model_options)
                self.user_controls['GROQ_API_KEY'] =st.session_state['GROQ_API_KEY'] = st.text_input("API Key",type="password")
                
                if not self.user_controls['GROQ_API_KEY']:
                    st.warning(" Please enter your GROQ API Key to Proceed. Don't have? refer: https://console.groq.com/home ")
            
            ##UseCase Selection
            self.user_controls['selected_usecase'] = st.selectbox("Select Usecases",usecase_options)

            if self.user_controls['selected_usecase'] == "Chatbot with Web":
                os.environ['TAVILY_API_KEY'] = self.user_controls['TAVILY_API_KEY'] = st.session_state['TAVILY_API_KEY'] = st.text_input("TAVILY API KEY",type="password")

                if not self.user_controls['TAVILY_API_KEY']:
                    st.warning("Please enter your TAVILY API KEY to proceed.Don't have? refer: https://www.tavily.com/")
    
        return self.user_controls