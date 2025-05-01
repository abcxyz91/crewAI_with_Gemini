from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
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

# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Define a Pydantic model for venue details (demonstrating Output as Pydantic)
# The purpose is to transform the fuzzy output of the LLM into a structered format
class VenueDetails(BaseModel):
    name: str
    address: str
    capacity: int
    booking_status: str

# Define your Agents, and provide them a role, goal and backstory
# Agent 1: Venue Coordinator
venue_coordinator = Agent(
    role="Venue Coordinator",
    goal="Identify and book an appropriate venue "
        "based on event requirements",
    backstory=(
        "With a keen sense of space and "
        "understanding of event logistics, "
        "you excel at finding and securing "
        "the perfect venue that fits the event's theme, "
        "size, and budget constraints."
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    llm=llm
)

 # Agent 2: Logistics Manager
logistics_manager = Agent(
    role="Logistics Manager",
    goal="Manage all logistics for the event "
        "including catering and equipment",
    backstory=(
        "Organized and detail-oriented, "
        "you ensure that every logistical aspect of the event "
        "from catering to equipment setup "
        "is flawlessly executed to create a seamless experience."
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    llm=llm
)

# Agent 3: Marketing and Communications Agent
marketing_communications_agent = Agent(
    role="Marketing and Communications Agent",
    goal="Effectively market the event and "
         "communicate with participants",
    backstory=(
        "Creative and communicative, "
        "you craft compelling messages and "
        "engage with potential attendees "
        "to maximize event exposure and participation."
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    llm=llm
)

# Define your Tasks, and provide them a description, expected_output, agent, and tools
# After getting LLM response, transform it into a structered JSON format following the Pydantic model
# Task 1: Venue search and selection
venue_task = Task(
    description="Find a venue in {event_city} "
                "that meets criteria for {event_topic}.",
    expected_output="All the details of a specifically chosen"
                    "venue you found to accommodate the event.",
    human_input=True,                   # requires human input to confirm the venue
    output_json=VenueDetails,           # specify the structure of the output following the Pydantic model
    output_file="venue_details.json",   # output in a JSON file
    agent=venue_coordinator
)

# Task 2: Logistics coordination
logistics_task = Task(
    description="Coordinate catering and "
                 "equipment for an event "
                 "with {expected_participants} participants "
                 "on {tentative_date}.",
    expected_output="Confirmation of all logistics arrangements "
                    "including catering and equipment setup.",
    human_input=True,
    async_execution=True,               # allows for asynchronous execution of the task
    output_file="catering_report.md",   # outputs the report as a markdown file
    agent=logistics_manager
)

# Task 3: Make marketing plan
# CrewAI framework requires that at most one asynchronous task can be executed at the end of the crew's task list.
marketing_task = Task(
    description="Promote the {event_topic} "
                "aiming to engage at least"
                "{expected_participants} potential attendees.",
    expected_output="Report on marketing activities "
                    "and attendee engagement formatted as markdown.",
    async_execution=True,               # allows for asynchronous execution of the task
    output_file="marketing_report.md",  # outputs the report as a markdown file
    agent=marketing_communications_agent
)

# Create your crew of Agents and pass the tasks to be performed by those agents.
# Since async_execution=True for logistics_task and marketing_task tasks, now the order for them does not matter in the tasks list.
event_management_crew = Crew(
    agents=[venue_coordinator, 
            logistics_manager, 
            marketing_communications_agent],
    tasks=[venue_task, 
           logistics_task, 
           marketing_task],
    verbose=True
)

# Set the input parameters and run the crew
event_details = {
    'event_topic': "Tech Innovation Conference",
    'event_description': "A gathering of tech innovators "
                         "and industry leaders "
                         "to explore future technologies.",
    'event_city': "San Francisco",
    'tentative_date': "2025-09-15",
    'expected_participants': 500,
    'budget': 20000,
    'venue_type': "Conference Hall"
}
result = event_management_crew.kickoff(inputs=event_details)