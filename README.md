# AI Agents Use Cases by CrewAI and Google Gemini

This repository contains example implementations of multi-agent AI systems using the CrewAI framework, configured to work with Google's Gemini API instead of the default OpenAI integration.

## Why AI Agents vs Traditional Programming?

Traditional programming approaches often face several limitations:
- **Rigid Logic**: Conventional code requires explicit rules and conditions for every scenario
- **Maintenance Burden**: Changes in requirements need manual code updates across multiple modules
- **Limited Adaptability**: Cannot easily handle new situations without code modifications
- **Complex Integration**: Connecting multiple services requires extensive boilerplate code
- **Static Decision Making**: Decisions are based on pre-defined rules rather than contextual understanding

AI Agents solve these challenges by offering:
- **Natural Language Processing**: Instead of rigid logic, agents understand and process requirements in natural language
- **Minimal Maintenance**: Requirements changes are handled through prompt adjustments rather than code rewrites
- **Dynamic Adaptation**: Agents can handle new scenarios without code changes
- **Flexible Integration**: Easy integration with new tools and services through natural language instructions
- **Intelligent Decision Making**: Context-aware responses based on natural language understanding

While AI Agents offer significant advantages, they also come with important considerations:

- **Cost Considerations**: API calls to language models incur ongoing costs, unlike traditional code execution
- **Latency Issues**: Responses depend on API call speed and model processing time, which can be slower than traditional code
- **Determinism**: Outputs may vary between runs, making testing and debugging more challenging
- **Token Limitations**: Complex tasks may hit context window limits of language models
- **Security Concerns**: Sensitive information needs careful handling when passing through external AI services
- **Quality Control**: Outputs require human validation to ensure accuracy and appropriateness
- **API Dependency**: System reliability depends on external API availability and stability

## Overview

The project demonstrates various use cases for AI agent collaboration:

- **AI Writer**: Blog content creation pipeline with planning, writing, and editing agents
- **Customer Support**: Multi-agent system for handling customer inquiries with memory capabilities
- **Customer Outreach**: Lead profiling and personalized communication system
- **Event Planning**: Comprehensive event management system with venue coordination and marketing
- **Financial Analysis**: Real-time market analysis and trading strategy development
- **Job Application Assistant**: Automated job application process with resume tailoring

## Prerequisites

To run these examples, you'll need:

1. A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Google Cloud Vertex AI credentials from [Google Cloud Console](https://developers.google.com/workspace/guides/create-credentials)
3. A Serper API key from [Serper.dev](https://serper.dev/api-key)

## Setup

1. Clone this repository
2. Create a `.env` file in the root directory with your API keys:
```
GEMINI_API_KEY=your_gemini_api_key
SERPER_API_KEY=your_serper_api_key
```
3. Place your Vertex AI credentials JSON file in the root directory as `vertex_ai_service_account.json`
4. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `ex1_ai_writer.py`: Automates entire content creation workflow - from topic research to final editing, replacing manual content pipeline management
- `ex2_ai_customer_support.py`: Intelligent customer service system that handles inquiries, escalates complex issues, and maintains conversation context automatically
- `ex3_ai_customer_outreach.py`: Autonomous lead generation system that researches prospects, crafts personalized messages, and manages follow-ups
- `ex4_ai_event_planning.py`: End-to-end event automation platform handling venue selection, vendor coordination, and promotional activities
- `ex5_ai_financial_analysis.py`: Automated financial research system combining market data analysis, trend detection, and strategy formulation
- `ex6_ai_job_application.py`: Smart career assistant that analyzes job posts, customizes applications, and generates targeted resumes automatically
- `helpers.py`: Utility functions for agent coordination and task management
- `instructions/`: Templates and guidelines for agent behavior and task execution

## Dependencies

- crewai
- crewai_tools
- python-dotenv
