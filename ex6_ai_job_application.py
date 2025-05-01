from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileReadTool, MDXSearchTool
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
        max_tokens=5000, # might need to be increased since the tasks are more complex
        vertex_credentials=vertex_credentials_json
    )

# Initialize the tools
# SerperDevTool uses Serper API
# MDXSearchTool uses OpenAI API by default but in this example using Google Gemini embedding.
# Therefore, it needs google-generativeai package as a dependency
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
read_resume = FileReadTool(file_path='./fake_resume.md')
semantic_search_resume = MDXSearchTool(
    mdx="./fake_resume.md",
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

# Define your Agents, and provide them a role, goal and backstory
# Agent 1: Researcher
researcher = Agent(
    role="Tech Job Researcher",
    goal="Make sure to do amazing analysis on "
         "job posting to help job applicants",
    backstory=(
        "As a Job Researcher, your prowess in "
        "navigating and extracting critical "
        "information from job postings is unmatched."
        "Your skills help pinpoint the necessary "
        "qualifications and skills sought "
        "by employers, forming the foundation for "
        "effective application tailoring."
    ),
    tools = [scrape_tool, search_tool],
    verbose=True,
    llm=llm
)

# Agent 2: Profiler
profiler = Agent(
    role="Personal Profiler for Engineers",
    goal="Do increditble research on job applicants "
         "to help them stand out in the job market",
    backstory=(
        "Equipped with analytical prowess, you dissect "
        "and synthesize information "
        "from diverse sources to craft comprehensive "
        "personal and professional profiles, laying the "
        "groundwork for personalized resume enhancements."
    ),
    tools = [scrape_tool, search_tool,
             read_resume, semantic_search_resume],
    verbose=True,
    llm=llm
)

# Agent 3: Resume Strategist
resume_strategist = Agent(
    role="Resume Strategist for Engineers",
    goal="Find all the best ways to make a "
         "resume stand out in the job market.",
    backstory=(
        "With a strategic mind and an eye for detail, you "
        "excel at refining resumes to highlight the most "
        "relevant skills and experiences, ensuring they "
        "resonate perfectly with the job's requirements."
    ),
    tools = [scrape_tool, search_tool,
             read_resume, semantic_search_resume],
    verbose=True,
    llm=llm
)

# Agent 4: Interview Preparer
interview_preparer = Agent(
    role="Engineering Interview Preparer",
    goal="Create interview questions and talking points "
         "based on the resume and job requirements",
    backstory=(
        "Your role is crucial in anticipating the dynamics of "
        "interviews. With your ability to formulate key questions "
        "and talking points, you prepare candidates for success, "
        "ensuring they can confidently address all aspects of the "
        "job they are applying for."
    ),
    tools = [scrape_tool, search_tool,
             read_resume, semantic_search_resume],
    verbose=True,
    llm=llm
)

# Define your Tasks, and provide them a description, expected_output, agent, and tools
# Task 1: Extract Job Requirements
# This task will run asynchronously with task 2
research_task = Task(
    description=(
        "Analyze the job posting URL provided ({job_posting_url}) "
        "to extract key skills, experiences, and qualifications "
        "required. Use the tools to gather content and identify "
        "and categorize the requirements."
    ),
    expected_output=(
        "A structured list of job requirements, including necessary "
        "skills, qualifications, and experiences."
    ),
    agent=researcher,
    async_execution=True
)

# Task 2: Compile Comprehensive Profile
# This task will run asynchronously with task 1
profile_task = Task(
    description=(
        "Compile a detailed personal and professional profile "
        "using the GitHub ({github_url}) URLs, and personal write-up "
        "({personal_writeup}). Utilize tools to extract and "
        "synthesize information from these sources."
    ),
    expected_output=(
        "A comprehensive profile document that includes skills, "
        "project experiences, contributions, interests, and "
        "communication style."
    ),
    agent=profiler,
    async_execution=True
)

# Task 3: Align Resume with Job Requirements
# With context=[research_task, profile_task], task 3 takes into account the output of those tasks in its execution.
# Task 3 will not run until it has the output(s) from task 1 & 2.
resume_strategy_task = Task(
    description=(
        "Using the profile and job requirements obtained from "
        "previous tasks, tailor the resume to highlight the most "
        "relevant areas. Employ tools to adjust and enhance the "
        "resume content. Make sure this is the best resume even but "
        "don't make up any information. Update every section, "
        "inlcuding the initial summary, work experience, skills, "
        "and education. All to better reflrect the candidates "
        "abilities and how it matches the job posting."
    ),
    expected_output=(
        "An updated resume that effectively highlights the candidate's "
        "qualifications and experiences relevant to the job."
    ),
    output_file="tailored_resume.md",
    context=[research_task, profile_task],
    agent=resume_strategist
)

# Task 4: Develop Interview Materials
# Task 4 will not run until it has the output(s) from task 1, 2 & 3.
interview_preparation_task = Task(
    description=(
        "Create a set of potential interview questions and talking "
        "points based on the tailored resume and job requirements. "
        "Utilize tools to generate relevant questions and discussion "
        "points. Make sure to use these question and talking points to "
        "help the candiadte highlight the main points of the resume "
        "and how it matches the job posting."
    ),
    expected_output=(
        "A document containing key questions and talking points "
        "that the candidate should prepare for the initial interview."
    ),
    output_file="interview_materials.md",
    context=[research_task, profile_task, resume_strategy_task],
    agent=interview_preparer
)

# Create your crew of Agents and pass the tasks to be performed by those agents.
job_application_crew = Crew(
    agents=[researcher,
            profiler,
            resume_strategist,
            interview_preparer],
    tasks=[research_task,
           profile_task,
           resume_strategy_task,
           interview_preparation_task],
    verbose=True
)

# Set the input parameters and run the crew
job_application_inputs = {
    'job_posting_url': 'https://jobs.lever.co/AIFund/6c82e23e-d954-4dd8-a734-c0c2c5ee00f1?lever-origin=applied&lever-source%5B%5D=AI+Fund',
    'github_url': 'https://github.com/joaomdmoura',
    'personal_writeup': """Noah is an accomplished Software
    Engineering Leader with 18 years of experience, specializing in
    managing remote and in-office teams, and expert in multiple
    programming languages and frameworks. He holds an MBA and a strong
    background in AI and data science. Noah has successfully led
    major tech initiatives and startups, proving his ability to drive
    innovation and growth in the tech industry. Ideal for leadership
    roles that require a strategic and innovative approach."""
}
result = job_application_crew.kickoff(inputs=job_application_inputs)