import os
import json
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.tools import tool

from models import (
    ObjectDetectionResult
)

load_dotenv()

# Initialize the search tool and LLM once
search_tool = DuckDuckGoSearchRun()
llm = ChatOpenAI(model="gpt-4-turbo")

# Define the research prompt template
research_prompt = ChatPromptTemplate.from_template("""
Based on the following object details and search results, identify 1-5 key historical milestones
in the evolution of this object. For each milestone, provide:
1. The approximate year or time period
2. A description of the innovation or change
3. The historical context or importance of this milestone
4. A brief visual description of the object at this milestone

Object Details:
- Name: {object_name}
- Category: {object_category}
- Distinguishing Features: {features}

Search Results:
{search_results}

Format your response as a JSON object with this structure:
{{
    "milestones": [
        {{
            "year": "year or time period",
            "title": "brief name of milestone",
            "description": "detailed description",
            "significance": "historical importance",
            "visual_description": "description for image generation"
        }},
        ...
    ]
}}
""")

# Create the research chain
research_chain = LLMChain(
    llm=llm,
    prompt=research_prompt
)

@tool("historical_research_tool")
def research_object_history(object_name: str, object_category: str, features: list) -> dict:
    """
    Research the historical evolution of an object based on its name, category, and features.
    Returns key milestones in the object's development throughout history.
    
    Args:
        object_name: The name of the object
        object_category: The category of the object
        features: List of distinguishing features of the object
        
    Returns:
        A dictionary containing historical milestones for the object
    """
    try:
        # Convert features list to string
        features_str = ", ".join(features) if isinstance(features, list) else features
        
        # Construct search queries based on object details
        search_queries = [
            f"history of {object_name}",
            f"evolution of {object_name} through history",
            f"{object_name} invention and development timeline",
            f"how has the {object_name} changed over time"
        ]
        
        # Combine search results
        all_search_results = ""
        for query in search_queries:
            search_result = search_tool.run(query)
            all_search_results += f"\nQuery: {query}\nResults: {search_result}\n\n"
        
        # Use the research chain to process search results
        response = research_chain.run(
            object_name=object_name,
            object_category=object_category,
            features=features_str,
            search_results=all_search_results
        )
        
        # Parse the response as JSON
        # First, try to parse it directly
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from text (in case model included extra text)
            import re
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                # Last resort - try eval (carefully!)
                result = eval(response)
        
        return result
    
    except Exception as e:
        print(f"Error in historical research: {str(e)}")
        # Return a minimal fallback response
        return {
            "error": str(e),
            "milestones": [
                {
                    "year": "Unknown",
                    "title": "Error retrieving historical data",
                    "description": f"An error occurred while researching the history of {object_name}",
                    "significance": "Please try again with more specific details",
                    "visual_description": "Error visualization"
                }
            ]
        }