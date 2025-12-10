from tavily import TavilyClient ##To interact with tavily api
from langchain_core.prompts import ChatPromptTemplate
from src.langgraphagenticai.state.state import State

class AINewsNode:
    def __init__(self,llm):
        """Initialize the AINewsNode with API keys for Tavily and GROQ"""
        self.tavily = TavilyClient()
        self.llm = llm
        #this is used to capture various steps in this file so that later can be use for steps shown
        self.state = {}
    
    def safe(self,text: str) -> str:
        """
        Safely encode text to ASCII, replacing unsupported characters.
        """
        return text.encode("ascii", errors="backslashreplace").decode("ascii")


    def fetch_news(self,state:dict) -> dict:
        """
        Fetch AI news based on the specified frequency.

        Args:
            state(dict): The state dictionary containing 'frequency'.
        Returns:
            dict: Updated state with 'news_data' key containing fetched news.
        """
        frequency = state['messages'][0].content.lower()
        self.state['frequency'] = frequency
        time_range_map = {"daily":'d',"weekly":'w',"monthly":'m',"year":'y'}
        days_map = {"daily":1,"weekly":7,"monthly":30,"year":366}

        response = self.tavily.search(
            query = "Top Artifical Intelligence (AI) technology news India and globally",
            topic = "news",
            time_range=time_range_map[frequency],
            include_answer="advanced",
            max_results=15,
            days=days_map[frequency]
        )

        cleaned_results = []
        for item in response.get("results", []):
            cleaned_results.append({
                "content": self.safe(item.get("content", "")),
                "url": self.safe(item.get("url", "")),
                "published_date": self.safe(item.get("published_date", "")),
            })

        state['news_data'] = cleaned_results
        self.state['news_data'] = cleaned_results
        return state

    def summarize_state(self,state:dict)->dict:
        """
        Summarize the fetched news using an LLM.

        Args:
            state(dict): The state dictionary containing 'news_data'.
        
        Returns:
            dict: Updated state with 'summary' key containing the summarized news.
        """

        news_items = self.state['news_data']
        prompt_template = ChatPromptTemplate.from_messages([
            ("system","""Summarize AI news articles into markdown format. For each item include:
            -Date in "YYYY-MM-DD" format in IST timezone
            -Concise sentences summary from latest news
            -Sort news by date wise (latest first)
            Use format:
            ###[Date]
            -[Summary](URL)"""),
            ("user","Articles:\n{articles}")
        ])

        articles_str = "\n\n".join([
            f"Content: {item.get('content','')}\nURL:{item.get('url','')}\nDate:{item.get('published_date','')}"
            for item in news_items
        ])

        response = self.llm.invoke(prompt_template.format(articles = articles_str))
        state['summary'] = response.content
        self.state['summary'] = state['summary']
        return self.state

    def save_result(self,state):
        frequency = self.state['frequency']
        summary = self.state['summary']
        filename = f"./AINews/{frequency}_summary.md"
        with open(filename,'w') as f:
            f.write(f"# {frequency.capitalize()} AI News Summary\n\n")
            f.write(summary)
        self.state['filename'] = filename
        return self.state