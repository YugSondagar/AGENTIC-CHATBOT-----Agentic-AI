from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


def get_tools():
    """
    Return the list of tools to be used in the chatbot
    """
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools = [TavilySearchResults(max_results=2),wikipedia]
    return tools

def create_tool_node(tools):
    """
    Create and return a tool node for the graph
    """
    return ToolNode(tools=tools)