from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv, find_dotenv
from typing import List
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
    'agents': 'config/agents.yaml',
    'tasks': 'config/tasks.yaml'
}

# Load configurations from YAML files
configs = {}
for config_type, file_path in files.items():
    with open(file_path, 'r') as file:
        configs[config_type] = yaml.safe_load(file) # reads and safely converts the YAML content into a Python dictionary

# Assign loaded configurations to specific variables
agents_config = configs['agents']
tasks_config = configs['tasks']

# Create Pydantic Models for Structured Output
# The purpose is to transform the fuzzy output of the LLM into a structured format that can be input into another external system
class TaskEstimate(BaseModel):
    task_name: str = Field(..., description="Name of the task")
    estimated_time_hours: float = Field(..., description="Estimated time to complete the task in hours")
    required_resources: List[str] = Field(..., description="List of resources required to complete the task")

class Milestone(BaseModel):
    milestone_name: str = Field(..., description="Name of the milestone")
    tasks: List[str] = Field(..., description="List of task IDs associated with this milestone")

class ProjectPlan(BaseModel):
    tasks: List[TaskEstimate] = Field(..., description="List of tasks with their estimates")
    milestones: List[Milestone] = Field(..., description="List of project milestones")

# Creating Agents
project_planning_agent = Agent(
    config=agents_config['project_planning_agent'],
    llm=llm
)

estimation_agent = Agent(
    config=agents_config['estimation_agent'],
    llm=llm
)

resource_allocation_agent = Agent(
    config=agents_config['resource_allocation_agent'],
    llm=llm
)

# Creating Tasks
task_breakdown = Task(
    config=tasks_config['task_breakdown'],
    agent=project_planning_agent
)

time_resource_estimation = Task(
    config=tasks_config['time_resource_estimation'],
    agent=estimation_agent
)

resource_allocation = Task(
    config=tasks_config['resource_allocation'],
    agent=resource_allocation_agent,
    output_pydantic=ProjectPlan # This is the structured output we want
)

# Creating Crew
crew = Crew(
    agents=[
        project_planning_agent,
        estimation_agent,
        resource_allocation_agent
    ],
    tasks=[
        task_breakdown,
        time_resource_estimation,
        resource_allocation
    ],
    verbose=True
)

# Set the input parameters and run the crew
project = 'Website'
industry = 'Technology'
project_objectives = 'Create a website for a small business'
team_members = """
- John Doe (Project Manager)
- Jane Doe (Software Engineer)
- Bob Smith (Designer)
- Alice Johnson (QA Engineer)
- Tom Brown (QA Engineer)
"""
project_requirements = """
- Create a responsive design that works well on desktop and mobile devices
- Implement a modern, visually appealing user interface with a clean look
- Develop a user-friendly navigation system with intuitive menu structure
- Include an "About Us" page highlighting the company's history and values
- Design a "Services" page showcasing the business's offerings with descriptions
- Create a "Contact Us" page with a form and integrated map for communication
- Implement a blog section for sharing industry news and company updates
- Ensure fast loading times and optimize for search engines (SEO)
- Integrate social media links and sharing capabilities
- Include a testimonials section to showcase customer feedback and build trust
"""

inputs = {
    'project_type': project,
    'project_objectives': project_objectives,
    'industry': industry,
    'team_members': team_members,
    'project_requirements': project_requirements
}

# Run the crew
result = crew.kickoff(inputs=inputs)

# Optional, measure how much it would cost each time if this crew runs at scale. (in case of ChatGPT 4o mini)
import pandas as pd

costs = 0.150 * (crew.usage_metrics.prompt_tokens + crew.usage_metrics.completion_tokens) / 1_000_000
print(f"Total costs: ${costs:.4f}")

# Convert UsageMetrics instance to a DataFrame
df_usage_metrics = pd.DataFrame([crew.usage_metrics.dict()])
df_usage_metrics

# Display the task result
tasks = result.pydantic.dict()['tasks']
df_tasks = pd.DataFrame(tasks)

# Display the DataFrame as an HTML table
df_tasks.style.set_table_attributes('border="1"').set_caption("Task Details").set_table_styles(
    [{'selector': 'th, td', 'props': [('font-size', '120%')]}]
)

# Display the milestone result
milestones = result.pydantic.dict()['milestones']
df_milestones = pd.DataFrame(milestones)

# Display the DataFrame as an HTML table
df_milestones.style.set_table_attributes('border="1"').set_caption("Task Details").set_table_styles(
    [{'selector': 'th, td', 'props': [('font-size', '120%')]}]
)