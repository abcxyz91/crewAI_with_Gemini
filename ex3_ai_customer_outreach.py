from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, DirectoryReadTool, FileReadTool
from crewai.tools import BaseTool
from dotenv import load_dotenv, find_dotenv
from helpers import pretty_print_result
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
        max_tokens=2000,
        vertex_credentials=vertex_credentials_json
    )

# Define your Agents, and provide them a role, goal and backstory
# A sale agent specialize in lead profiling
sales_rep_agent = Agent(
    role="Sales Representative",
    goal="Identify high-value leads that match "
         "our ideal customer profile",
    backstory=(
        "As a part of the dynamic sales team at CrewAI, "
        "your mission is to scour "
        "the digital landscape for potential leads. "
        "Armed with cutting-edge tools "
        "and a strategic mindset, you analyze data, "
        "trends, and interactions to "
        "unearth opportunities that others might overlook. "
        "Your work is crucial in paving the way "
        "for meaningful engagements and driving the company's growth."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# A lead sales representative focuses on nurturing leads
lead_sales_rep_agent = Agent(
    role="Lead Sales Representative",
    goal="Nurture leads with personalized, compelling communications",
    backstory=(
        "Within the vibrant ecosystem of CrewAI's sales department, "
        "you stand out as the bridge between potential clients "
        "and the solutions they need."
        "By creating engaging, personalized messages, "
        "you not only inform leads about our offerings "
        "but also make them feel seen and heard."
        "Your role is pivotal in converting interest "
        "into action, guiding leads through the journey "
        "from curiosity to commitment."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Setup the tools for the agents
directory_read_tool = DirectoryReadTool(directory='./instructions') # Tool allowed to read from this directory only
file_read_tool = FileReadTool() # Tool allowed to read from any file
search_tool = SerperDevTool() # Tool to search the web for information by using the Serper API

# Setup a custom tool inherting from BaseTool class 
# Every Tool needs to have a name and a description
class SentimentAnalysisTool(BaseTool):
    name: str ="Sentiment Analysis Tool"
    description: str = ("Analyzes the sentiment of text "
         "to ensure positive and engaging communication.")
    
    def _run(self, text: str) -> str:
        # Your custom code tool goes here
        return "positive"
    
sentiment_analysis_tool = SentimentAnalysisTool()
    
# Define your Tasks, and provide them a description, expected_output, agent, and tools
# A lead profiling task can use the tools
lead_profiling_task = Task(
    description=(
        "Conduct an in-depth analysis of {lead_name}, "
        "a company in the {industry} sector "
        "that recently showed interest in our solutions. "
        "Utilize all available data sources "
        "to compile a detailed profile, "
        "focusing on key decision-makers, recent business "
        "developments, and potential needs "
        "that align with our offerings. "
        "This task is crucial for tailoring "
        "our engagement strategy effectively.\n"
        "Don't make assumptions and "
        "only use information you absolutely sure about."
    ),
    expected_output=(
        "A comprehensive report on {lead_name}, "
        "including company background, "
        "key personnel, recent milestones, and identified needs. "
        "Highlight potential areas where "
        "our solutions can provide value, "
        "and suggest personalized engagement strategies."
    ),
    tools=[directory_read_tool, file_read_tool, search_tool],
    agent=sales_rep_agent,
)

# A personalized outreach task can use search tool and a dummy custom tool
personalized_outreach_task = Task(
    description=(
        "Using the insights gathered from "
        "the lead profiling report on {lead_name}, "
        "craft a personalized outreach campaign "
        "aimed at {key_decision_maker}, "
        "the {position} of {lead_name}. "
        "The campaign should address their recent {milestone} "
        "and how our solutions can support their goals. "
        "Your communication must resonate "
        "with {lead_name}'s company culture and values, "
        "demonstrating a deep understanding of "
        "their business and needs.\n"
        "Don't make assumptions and only "
        "use information you absolutely sure about."
    ),
    expected_output=(
        "A series of personalized email drafts "
        "tailored to {lead_name}, "
        "specifically targeting {key_decision_maker}."
        "Each draft should include "
        "a compelling narrative that connects our solutions "
        "with their recent achievements and future goals. "
        "Ensure the tone is engaging, professional, "
        "and aligned with {lead_name}'s corporate identity."
    ),
    tools=[sentiment_analysis_tool, search_tool],
    agent=lead_sales_rep_agent,
)

# Create your crew of Agents and pass the tasks to be performed by those agents.
crew = Crew(
    agents=[sales_rep_agent, lead_sales_rep_agent],
    tasks=[lead_profiling_task, personalized_outreach_task],
    verbose=True,
	memory=True,
    embedder={
        "provider": "google",
        "config": {
            "api_key": os.getenv("GEMINI_API_KEY"),
            "model": "models/text-embedding-004"
        }
    }
)

# Running the Crew
inputs = {
    "lead_name": "DeepLearningAI",
    "industry": "Online Learning Platform",
    "key_decision_maker": "Andrew Ng",
    "position": "CEO",
    "milestone": "product launch"
}

result = crew.kickoff(inputs=inputs)