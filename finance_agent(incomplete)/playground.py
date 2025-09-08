import os
from dotenv import load_dotenv
import streamlit as st
import inspect
import phi
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Patch inspect.getargspec for Python 3.11+
if not hasattr(inspect, "getargspec"):
    from collections import namedtuple
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")
    def getargspec(func):
        return ArgSpec(*inspect.getfullargspec(func)[:4])
    inspect.getargspec = getargspec

# Load keys
load_dotenv()
phi.api_key = os.getenv("PHI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Agents ---
web_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        )
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# --- Streamlit UI ---
st.set_page_config(page_title="Unified Finance AI", layout="wide")
st.title("Unified Finance & Web AI Assistant")

query = st.text_input(
    "Ask a finance or investment question (e.g., 'Between TSLA and NVDA, which should I invest $1000 in?'):"
)

if st.button("Get Answer") and query:
    st.info("Fetching information, please wait...")

    try:
        # Step 1: Use web agent to gather contextual info
        web_info = web_agent.run(query)

        # Step 2: Use finance agent to gather financial data
        finance_info = finance_agent.run(query)

        # Step 3: Combine both results and ask finance agent to give final recommendation
        combined_prompt = (
            f"User question: {query}\n\n"
            f"Web search information:\n{web_info}\n\n"
            f"Financial data:\n{finance_info}\n\n"
            f"Based on both, provide a clear, concise investment recommendation."
        )
        final_answer = finance_agent.run(combined_prompt)

        # Display final answer
        st.markdown("### Final Recommendation")
        st.write(final_answer)

    except Exception as e:
        st.error(f"Error generating answer: {e}")
