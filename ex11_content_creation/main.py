from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from dotenv import load_dotenv, find_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field
import os, yaml, json, textwrap
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())
# Set the Gemini API key from environment variables
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Load the JSON file with Vertex AI service account credentials
file_path = "../vertex_ai_service_account.json"
with open(file_path, "r") as file:
    vertex_credentials = json.load(file)

# Convert the credentials to a JSON string
vertex_credentials_json = json.dumps(vertex_credentials)

# Create 2 LLMs depends on agent
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.5,
    max_tokens=2000,
    vertex_credentials=vertex_credentials_json
    )

lite_llm = LLM(
    model="gemini/gemini-2.0-flash-lite",
    temperature=0.5,
    max_tokens=2000,
    vertex_credentials=vertex_credentials_json
)

# Define file paths for YAML configurations
files = {
    'agents': 'config/agents.yaml',
    'tasks': 'config/tasks.yaml'
}

# Load configurations from YAML files
configs = {}
for config_type, file_path in files.items():
    with open(file_path, 'r') as file:
        configs[config_type] = yaml.safe_load(file)

# Assign loaded configurations to specific variables
agents_config = configs['agents']
tasks_config = configs['tasks']

# Create Pydantic Models for Structured Output
# The purpose is to transform the fuzzy output of the LLM into a structured format that can be input into another external system
class SocialMediaPost(BaseModel):
    platform: str = Field(..., description="The social media platform where the post will be published (e.g., Twitter, LinkedIn).")
    content: str = Field(..., description="The content of the social media post, including any hashtags or mentions.")

class ContentOutput(BaseModel):
    article: str = Field(..., description="The article, formatted in markdown.")
    social_media_posts: List[SocialMediaPost] = Field(..., description="A list of social media posts related to the article.")

# Initialize the tools
# Since WebsiteSearchTool uses RAG, we need an embedding model
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
web_search_tool = WebsiteSearchTool(
    config=dict(
        llm={
            "provider": "google",
            "config": {
                "model": "gemini/gemini-2.0-flash",
                "api_key": os.getenv("GEMINI_API_KEY")
            }
        },
        embedder={
            "provider": "google",
            "config": {
                "model": "models/text-embedding-004",
                "task_type": "retrieval_document"
            }
        }
    )
)

# Creating Agents
# This agent will use Google search for relevant news, reading entire article bodies, extracting headlines
# Basically, this one does a broad research, so we need a fast and low latency LLM
market_news_monitor_agent = Agent(
    config=agents_config['market_news_monitor_agent'],
    tools=[search_tool, scrape_tool],
    llm=lite_llm,
)

# This agent also uses Google search for relevant news, then use RAG to semantic search from specified URLs efficiently 
# This one does a specific research within a website, and we also need a fast and low latency LLM
data_analyst_agent = Agent(
    config=agents_config['data_analyst_agent'],
    tools=[search_tool, web_search_tool],
    llm=lite_llm,
)

content_creator_agent = Agent(
    config=agents_config['content_creator_agent'],
    tools=[search_tool, web_search_tool],
    llm=llm
)

quality_assurance_agent = Agent(
    config=agents_config['quality_assurance_agent'],
    llm=llm
)

# Creating Tasks
monitor_financial_news_task = Task(
    config=tasks_config['monitor_financial_news'],
    agent=market_news_monitor_agent
)

analyze_market_data_task = Task(
    config=tasks_config['analyze_market_data'],
    agent=data_analyst_agent
)

create_content_task = Task(
    config=tasks_config['create_content'],
    agent=content_creator_agent,
    context=[monitor_financial_news_task, analyze_market_data_task]
)

quality_assurance_task = Task(
    config=tasks_config['quality_assurance'],
    agent=quality_assurance_agent,
    output_pydantic=ContentOutput
)

# Creating Crew
content_creation_crew = Crew(
    agents=[
        market_news_monitor_agent,
        data_analyst_agent,
        content_creator_agent,
        quality_assurance_agent
    ],
    tasks=[
        monitor_financial_news_task,
        analyze_market_data_task,
        create_content_task,
        quality_assurance_task
    ],
    verbose=True
)

# Kicking off the Crew
result = content_creation_crew.kickoff(inputs={
  'subject': 'Inflation in the US and the impact on the stock market in 2024'
})

# Show the result
posts = result.pydantic.dict()['social_media_posts']
for post in posts:
    platform = post['platform']
    content = post['content']
    print(platform)
    wrapped_content = textwrap.fill(content, width=50)
    print(wrapped_content)
    print('-' * 50)