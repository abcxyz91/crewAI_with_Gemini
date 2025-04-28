from crewai import Agent, Task, Crew, LLM
from crewai_tools import ScrapeWebsiteTool
from dotenv import load_dotenv, find_dotenv
import os, sys, json
import warnings
warnings.filterwarnings('ignore') # Suppress unimportant warnings

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())
# Set the Gemini API key from environment variables
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

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
support_agent = Agent(
    role="Senior Support Representative",
	goal="Be the most friendly and helpful "
        "support representative in your team",
	backstory=(
		"You work at crewAI (https://crewai.com) and "
        " are now working on providing "
		"support to {customer}, a super important customer "
        " for your company."
		"You need to make sure that you provide the best support!"
		"Make sure to provide full complete answers, "
        " and make no assumptions."
	),
	allow_delegation=False,
	verbose=True,
    llm=llm
)

# Support Quality Assurance Agent can delegate work back to the Support Agent, allowing for these agents to work together
# But it is not sure 100$ that the Support Quality Assurance Agent will delegate work back to the Support Agent
# Set allow_delegation to True, and the Support Quality Assurance Agent will decide if it wants to delegate work back to the Support Agent or not
support_quality_assurance_agent = Agent(
	role="Support Quality Assurance Specialist",
	goal="Get recognition for providing the "
    "best support quality assurance in your team",
	backstory=(
		"You work at crewAI (https://crewai.com) and "
        "are now working with your team "
		"on a request from {customer} ensuring that "
        "the support representative is "
		"providing the best support possible.\n"
		"You need to make sure that the support representative "
        "is providing full"
		"complete answers, and make no assumptions."
	),
	verbose=True,
    llm=llm
)

# Instantiate a document scraper tool. The tool will scrape a page (only 1 URL) of the CrewAI documentation
docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/"
)

# Define your Tasks, and provide them a description, expected_output and agent
# Agent Level: The Agent can use the Tool(s) on any Task it performs
# Task Level: The Agent will only use the Tool(s) when performing that specific Task. Below, we set the tool to be Task Level
inquiry_resolution = Task(
    description=(
        "{customer} just reached out with a super important ask:\n"
	    "{inquiry}\n\n"
        "{person} from {customer} is the one that reached out. "
		"Make sure to use everything you know "
        "to provide the best support possible."
		"You must strive to provide a complete "
        "and accurate response to the customer's inquiry."
    ),
    expected_output=(
	    "A detailed, informative response to the "
        "customer's inquiry that addresses "
        "all aspects of their question.\n"
        "The response should include references "
        "to everything you used to find the answer, "
        "including external data or solutions. "
        "Ensure the answer is complete, "
		"leaving no questions unanswered, and maintain a helpful and friendly "
		"tone throughout."
    ),
	tools=[docs_scrape_tool],
    agent=support_agent,
)

# quality_assurance_review is not using any Tool(s). It will only review the work of the Support Agent without accessing to external docs
quality_assurance_review = Task(
    description=(
        "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
        "Ensure that the answer is comprehensive, accurate, and adheres to the "
		"high-quality standards expected for customer support.\n"
        "Verify that all parts of the customer's inquiry "
        "have been addressed "
		"thoroughly, with a helpful and friendly tone.\n"
        "Check for references and sources used to "
        " find the information, "
		"ensuring the response is well-supported and "
        "leaves no questions unanswered."
    ),
    expected_output=(
        "A final, detailed, and informative response "
        "ready to be sent to the customer.\n"
        "This response should fully address the "
        "customer's inquiry, incorporating all "
		"relevant feedback and improvements.\n"
		"Don't be too formal, we are a chill and cool company "
	    "but maintain a professional and friendly tone throughout."
    ),
    agent=support_quality_assurance_agent,
)

# Create your crew of Agents and pass the tasks to be performed by those agents.
# Setting memory=True enables all Memories (Short Term, Long Term and Entity Memory) or response of each other
# It was implemented by embedding the response of each agent by Gemini embedding model and storing it in a vector database (ChromaDB)
crew = Crew(
    agents=[support_agent, support_quality_assurance_agent],
    tasks=[inquiry_resolution, quality_assurance_review],
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
    "customer": "DeepLearningAI",
    "person": "Andrew Ng",
    "inquiry": "I need help with setting up a Crew "
               "and kicking it off, specifically "
               "how can I add memory to my crew? "
               "Can you provide guidance?"
}
result = crew.kickoff(inputs=inputs)
print(result)