from crewai import Agent, Task, Crew, LLM, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from dotenv import load_dotenv, find_dotenv
from datetime import date
import os, sys, json
import warnings
warnings.filterwarnings('ignore') # Suppress unimportant warnings

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())
# Set the Gemini API key from environment variables
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
# Set Serper API key from environment variables
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# Load the JSON file with Vertex AI service account credentials
file_path = "vertex_ai_service_account.json"
with open(file_path, "r") as file:
    vertex_credentials = json.load(file)

# Convert the credentials to a JSON string
vertex_credentials_json = json.dumps(vertex_credentials)

llm = LLM(
        model="gemini/gemini-2.0-flash",
        temperature=0.5,
        max_tokens=5000, # might need to be increased since the tasks are more complex
        vertex_credentials=vertex_credentials_json
    )

# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Define your Agents, and provide them a role, goal and backstory
# Agent 1: Data analyst
data_analyst_agent = Agent(
    role="Data Analyst",
    goal="Monitor and analyze market data in real-time "
         "to identify trends and predict market movements.",
    backstory="Specializing in financial markets, this agent "
              "uses statistical modeling and machine learning "
              "to provide crucial insights. With a knack for data, "
              "the Data Analyst Agent is the cornerstone for "
              "informing trading decisions.",
    verbose=True,
    allow_delegation=False,
    tools = [scrape_tool, search_tool],
    llm=llm
)

# Agent 2: Trading strategist
trading_strategy_agent = Agent(
    role="Trading Strategy Developer",
    goal="Develop and test various trading strategies based "
         "on insights from the Data Analyst Agent.",
    backstory="Equipped with a deep understanding of financial "
              "markets and quantitative analysis, this agent "
              "devises and refines trading strategies. It evaluates "
              "the performance of different approaches to determine "
              "the most profitable and risk-averse options.",
    verbose=True,
    allow_delegation=True, # Might delegate parts to Analyst or Risk if needed
    tools = [scrape_tool, search_tool],
    llm=llm
)

# Agent 3: Execution agent
execution_agent = Agent(
    role="Trade Advisor",
    goal="Suggest optimal trade execution strategies "
         "based on approved trading strategies.",
    backstory="This agent specializes in analyzing the timing, price, "
              "and logistical details of potential trades. By evaluating "
              "these factors, it provides well-founded suggestions for "
              "when and how trades should be executed to maximize "
              "efficiency and adherence to strategy.",
    verbose=True,
    allow_delegation=False,
    tools = [scrape_tool, search_tool],
    llm=llm
)

# Agent 4: Risk manager
risk_management_agent = Agent(
    role="Risk Advisor",
    goal="Evaluate and provide insights on the risks "
         "associated with potential trading activities.",
    backstory="Armed with a deep understanding of risk assessment models "
              "and market dynamics, this agent scrutinizes the potential "
              "risks of proposed trades. It offers a detailed analysis of "
              "risk exposure and suggests safeguards to ensure that "
              "trading activities align with the firm’s risk tolerance.",
    verbose=True,
    allow_delegation=False,
    tools = [scrape_tool, search_tool],
    llm=llm
)

# Define your Tasks, and provide them a description, expected_output, agent, and tools
# Task 1: Analyze Market Data
data_analysis_task = Task(
    description=(
        "Continuously monitor and analyze market data for "
        "the selected stock ({stock_selection}). "
        "Use statistical modeling and machine learning to "
        "identify trends and predict market movements. "
        "Factor in real-time news sentiment and significant market events "
        "if news impact is considered ({news_impact_consideration}). "
        "Use up-to-date insights and market activity as of {current_date}."
    ),
    expected_output=(
        "Insights and alerts about significant market "
        "opportunities or threats for {stock_selection}."
    ),
    agent=data_analyst_agent,
)

# Task 2: Develop Trading Strategies
strategy_development_task = Task(
    description=(
        "Develop and refine trading strategies based on "
        "the insights from the Data Analyst Agent and "
        "user-defined risk tolerance ({risk_tolerance}). "
        "Consider trading preferences ({trading_strategy_preference}) "
        "and user's available capital ({initial_capital})."
    ),
    expected_output=(
        "A set of potential trading strategies for {stock_selection} "
        "that align with the user's risk tolerance."
    ),
    agent=trading_strategy_agent,
)

# Task 3: Plan Trade Execution
execution_planning_task = Task(
    description=(
        "Based on the trading strategies developed by the Trading Strategy Agent "
        "to determine the best execution methods for {stock_selection}, "
        "considering current market conditions and optimal pricing."
    ),
    expected_output=(
        "Detailed execution plans suggesting how and when to "
        "execute trades for {stock_selection}."
    ),
    agent=execution_agent,
)

# Task 4: Assess Trading Risks
risk_assessment_task = Task(
    description=(
        "Evaluate the risks associated with the proposed trading "
        "strategies and execution plans for {stock_selection}. "
        "Provide a detailed analysis of potential risks "
        "and suggest mitigation strategies."
    ),
    expected_output=(
        "A comprehensive risk analysis report detailing potential "
        "risks and mitigation recommendations for {stock_selection}."
    ),
    agent=risk_management_agent,
)

# Create your crew of Agents and pass the tasks to be performed by those agents.
# The Process class helps to delegate the workflow to the Agents hierarchically (by a LLM Manager).
# manager_llm (in Process.hierarchical) can infer task dependencies if you write task descriptions clearly and refer to prior task outputs.
# However, it is better to set output_file and read by the next task to avoid any ambiguity and better logging.
financial_trading_crew = Crew(
    agents=[data_analyst_agent, 
            trading_strategy_agent, 
            execution_agent, 
            risk_management_agent],
    tasks=[data_analysis_task, 
           strategy_development_task, 
           execution_planning_task, 
           risk_assessment_task],
    manager_llm=llm,
    process=Process.hierarchical,
    verbose=True
)

# Set the input parameters and run the crew
financial_trading_inputs = {
    'stock_selection': 'AAPL',
    'initial_capital': '100000',
    'risk_tolerance': 'Medium',
    'trading_strategy_preference': 'Day Trading',
    'news_impact_consideration': True,
    'current_date': str(date.today()) # using current date to improve time-sensitive analysis
}

# The value in result will be the output of the last task listed in the tasks=[...] array
result = financial_trading_crew.kickoff(inputs=financial_trading_inputs)