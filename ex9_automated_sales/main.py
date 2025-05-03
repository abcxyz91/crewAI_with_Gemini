from crewai import Agent, Task, Crew, LLM, Flow
from crewai.flow.flow import start, listen, and_, or_, router
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from dotenv import load_dotenv, find_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field
import os, yaml, json
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

llm = LLM(
        model="gemini/gemini-2.0-flash",
        temperature=0.5,
        max_tokens=5000, # might need to be increased since the tasks are more complex
        vertex_credentials=vertex_credentials_json
    )

# Loading Tasks and Agents in a config YAML files
# Define file paths for YAML configurations
files = {
    'lead_agents': 'config/lead_qualification_agents.yaml',
    'lead_tasks': 'config/lead_qualification_tasks.yaml',
    'email_agents': 'config/email_engagement_agents.yaml',
    'email_tasks': 'config/email_engagement_tasks.yaml'
}

# Load configurations from YAML files
configs = {}
for config_type, file_path in files.items():
    with open(file_path, 'r') as file:
        configs[config_type] = yaml.safe_load(file)

# Assign loaded configurations to specific variables
lead_agents_config = configs['lead_agents']
lead_tasks_config = configs['lead_tasks']
email_agents_config = configs['email_agents']
email_tasks_config = configs['email_tasks']

# Create Pydantic Models for Structured Output
# The purpose is to transform the fuzzy output of the LLM into a structured format that can be input into another external system
class LeadPersonalInfo(BaseModel):
    name: str = Field(..., description="The full name of the lead.")
    job_title: str = Field(..., description="The job title of the lead.")
    role_relevance: int = Field(..., ge=0, le=10, description="A score representing how relevant the lead's role is to the decision-making process (0-10).")
    professional_background: Optional[str] = Field(..., description="A brief description of the lead's professional background.")

class CompanyInfo(BaseModel):
    company_name: str = Field(..., description="The name of the company the lead works for.")
    industry: str = Field(..., description="The industry in which the company operates.")
    company_size: int = Field(..., description="The size of the company in terms of employee count.")
    revenue: Optional[float] = Field(None, description="The annual revenue of the company, if available.")
    market_presence: int = Field(..., ge=0, le=10, description="A score representing the company's market presence (0-10).")

class LeadScore(BaseModel):
    score: int = Field(..., ge=0, le=100, description="The final score assigned to the lead (0-100).")
    scoring_criteria: List[str] = Field(..., description="The criteria used to determine the lead's score.")
    validation_notes: Optional[str] = Field(None, description="Any notes regarding the validation of the lead score.")

class LeadScoringResult(BaseModel):
    personal_info: LeadPersonalInfo = Field(..., description="Personal information about the lead.")
    company_info: CompanyInfo = Field(..., description="Information about the lead's company.")
    lead_score: LeadScore = Field(..., description="The calculated score and related information for the lead.")

# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Create Lead Qualification Crew
# Creating Agents
lead_data_agent = Agent(
  config=lead_agents_config['lead_data_agent'],
  tools=[search_tool, scrape_tool],
  llm=llm
)

cultural_fit_agent = Agent(
  config=lead_agents_config['cultural_fit_agent'],
  tools=[search_tool, scrape_tool],
  llm=llm
)

scoring_validation_agent = Agent(
  config=lead_agents_config['scoring_validation_agent'],
  tools=[search_tool, scrape_tool],
  llm=llm
)

# Creating Tasks
lead_data_task = Task(
  config=lead_tasks_config['lead_data_collection'],
  agent=lead_data_agent
)

cultural_fit_task = Task(
  config=lead_tasks_config['cultural_fit_analysis'],
  agent=cultural_fit_agent
)

# The logic of scoring is not hardcoded but handled by a LLM, instructed through your YAML task description and guided by context
scoring_validation_task = Task(
  config=lead_tasks_config['lead_scoring_and_validation'],
  agent=scoring_validation_agent,
  context=[lead_data_task, cultural_fit_task],
  output_pydantic=LeadScoringResult
)

# Creating Crew
lead_scoring_crew = Crew(
  agents=[
    lead_data_agent,
    cultural_fit_agent,
    scoring_validation_agent
  ],
  tasks=[
    lead_data_task,
    cultural_fit_task,
    scoring_validation_task
  ],
  verbose=True
)

# Create Email Engagement Crew
# Creating Agents
email_content_specialist = Agent(
  config=email_agents_config['email_content_specialist'],
  llm=llm
)

engagement_strategist = Agent(
  config=email_agents_config['engagement_strategist'],
  llm=llm
)

# Creating Tasks
email_drafting = Task(
  config=email_tasks_config['email_drafting'],
  agent=email_content_specialist
)

engagement_optimization = Task(
  config=email_tasks_config['engagement_optimization'],
  agent=engagement_strategist
)

# Creating Crew
email_writing_crew = Crew(
  agents=[
    email_content_specialist,
    engagement_strategist
  ],
  tasks=[
    email_drafting,
    engagement_optimization
  ],
  verbose=True
)

# Create a Flow
# Flow allows you to run Python code in between the tasks and agents.
# This is useful for data transformation, validation, or any other processing that needs to happen between tasks.
class SalesPipeline(Flow):
    
  @start()
  def fetch_leads(self):
    # Pull our leads from the database
    # This is a mock, in a real-world scenario, this is where we would
    # fetch leads from a database
    leads = [
      {
        "lead_data": {
          "name": "João Moura",
          "job_title": "Director of Engineering",
          "company": "Clearbit",
          "email": "joao@clearbit.com",
          "use_case": "Using AI Agent to do better data enrichment."
        },
      },
    ]
    return leads

  @listen(fetch_leads)
  def score_leads(self, leads):
    # Run lead_scoring_crew for each lead in leads, then save the results in self.state
    # leads[0] will return scores[0] and so on
    scores = lead_scoring_crew.kickoff_for_each(leads)
    self.state["score_crews_results"] = scores
    print(scores)
    return scores

  @listen(score_leads)
  def store_leads_score(self, scores):
    # This is a mock, in a real-world scenario, we would store the scores in the database
    return scores

  @listen(score_leads)
  def filter_leads(self, scores):
    # Filters for leads with score > 70
    return [score for score in scores if score['lead_score'].score > 70]

  @listen(and_(filter_leads, store_leads_score))
  def log_leads(self, leads):
    # Logs leads only after both filtering and storing are done
    print(f"Leads: {leads}")

  # This router decorator in the tutorial may be deprecated and no longer accept keyword argument
  # Might need to re-wrote later
  @router(filter_leads)
  def count_leads(self, scores):
    # Routes leads based on how many were filtered
    if len(scores) > 10:
      return 'high'
    elif len(scores) > 5:
      return 'medium'
    else:
      return 'low'

  @listen('high')
  def store_in_salesforce(self, leads):
    # If score is high, store in salesforce (this is a mock)
    return leads

  @listen('medium')
  def send_to_sales_team(self, leads):
    # If score is medium, send to sales team (this is a mock)
    return leads

  @listen('low')
  def write_email(self, leads):
    # If few leads: converts results to dicts, then uses email_writing_crew to write emails for each lead in leads
    scored_leads = [lead.to_dict() for lead in leads]
    emails = email_writing_crew.kickoff_for_each(scored_leads)
    return emails

  @listen(write_email)
  def send_email(self, emails):
    # This is a mock code for sending email
    return emails
  
# Run the Flow  
flow = SalesPipeline()

# Optional: you can plot the flow to visualize it
flow.plot()
# Graph saved as crewai_flow_graph.html