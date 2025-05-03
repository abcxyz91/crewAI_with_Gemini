from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool
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

# Initialize Tool use
csv_tool = FileReadTool(file_path='./support_tickets_data.csv')

# Creating Agents
suggestion_generation_agent = Agent(
  config=agents_config['suggestion_generation_agent'],
  tools=[csv_tool]
)

reporting_agent = Agent(
  config=agents_config['reporting_agent'],
  tools=[csv_tool]
)

chart_generation_agent = Agent(
  config=agents_config['chart_generation_agent'],
  allow_code_execution=True # If set to True, agent can write and execute code in a protected environment using Docker
)

# Creating Tasks
suggestion_generation = Task(
  config=tasks_config['suggestion_generation'],
  agent=suggestion_generation_agent
)

table_generation = Task(
  config=tasks_config['table_generation'],
  agent=reporting_agent
)

chart_generation = Task(
  config=tasks_config['chart_generation'],
  agent=chart_generation_agent
)

final_report_assembly = Task(
  config=tasks_config['final_report_assembly'],
  agent=reporting_agent,
  context=[suggestion_generation, table_generation, chart_generation]
)


# Creating Crew
support_report_crew = Crew(
  agents=[
    suggestion_generation_agent,
    reporting_agent,
    chart_generation_agent
  ],
  tasks=[
    suggestion_generation,
    table_generation,
    chart_generation,
    final_report_assembly
  ],
  verbose=True
)

# You can test the Crew by running in the terminal: "crewai test" to test over multiple times and was scored by a Judge LLM
# You can also train the Crew by running in the terminal: "crewai train"
# In training mode, you can provide sementic and sentiment feedback to the Crew to improve in the next iteration
# After training, you can compare the performance before vs after training by running "crewai test" again
# Or you can directly edit your .yaml files and validate through actual execution
# Finally, run the final version of your Crew with the following command
result = support_report_crew.kickoff()