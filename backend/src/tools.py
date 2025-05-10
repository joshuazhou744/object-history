import json
import os
import base64
import requests
from io import BytesIO
from models import ObjectDetectionResult
from langchain.tools import tool
from PIL import Image

from openai import OpenAI
from dotenv import load_dotenv

from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference

load_dotenv()

credentials = Credentials(
    url = "https://us-south.ml.cloud.ibm.com"
)
client = APIClient(credentials)
project_id = "skills-network"

class ObjectDetectionTool:
    @tool("Object Detection Tool")
    def detect_object(image_input=None):
        """
        Analyze the uploaded image and identify the primary object within it.
        Provide the object_name, object_category, and any distinguishing_features.
        
        Args:
            image_input: Path to the image file or URL of the image to analyze
        """
        print(f"Tool received input: {image_input}")  # Debug print
            
        try:
            # Handle case when no input is provided
            if image_input is None or image_input == '':
                # Use the direct URL to the hammer image
                image_input = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/U7sAx8ZP1dn8HpOcMO_wcg/test.jpg"
                print(f"Using default image URL: {image_input}")
                
            if isinstance(image_input, dict):
                # Try to get the actual path from various potential formats
                if "image_input" in image_input:
                    image_input = image_input["image_input"]
                elif "name" in image_input and image_input["name"] == "image_input":
                    # This might be the format CrewAI is using
                    raise ValueError("Tool received metadata instead of actual image path")
                else:
                    # Just use the first value
                    image_input = list(image_input.values())[0]
                    
            print(f"Using image path/URL: {image_input}")  # Debug print
                
            if image_input.startswith("http"):
                response = requests.get(image_input)
                response.raise_for_status()
                image_bytes = BytesIO(response.content)
            else:
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"File not found: {image_input}")
                with open(image_input, "rb") as image_file:
                    image_bytes = BytesIO(image_file.read())

            encoded_image = base64.b64encode(image_bytes.read()).decode("utf-8")

            model = ModelInference(
                model_id="meta-llama/llama-3-2-90b-vision-instruct",
                credentials=credentials,
                project_id=project_id,
                params={"max_tokens": 300},
            )
            response = model.chat(
                messages = [
                    {
                        "role":"user",
                        "content": [
                            {"type": "text", "text": """Analyze this image and identify the primary object within it. 
                            Return ONLY a JSON object with this exact format:
                            {
                                "object_name": "name of object",
                                "object_category": "category of object",
                                "distinguishing_features": ["feature 1", "feature 2", ...]
                            }
                            Do not include any other text or explanation."""},
                            {"type":"image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded_image}}
                        ]
                    }
                ]
            )
            
            # Parse the response from the correct structure
            print(f"Raw response: {response}")
            content_text = response['choices'][0]['message']['content']

            result = json.loads(content_text)
            print(result)
            
            formatted_result = {
                "object_name": result.get("object_name", "unknown"),
                "object_category": result.get("object_category", "unknown"),
                "distinguishing_features": result.get("distinguishing_features", [])
            }

            print("\n=== Object Detection Results ===")
            print(f"Object: {formatted_result['object_name']}")
            print(f"Category: {formatted_result['object_category']}")
            print("\nDistinguishing Features:")
            for feature in formatted_result['distinguishing_features']:
                print(f"- {feature}")
            print("==============================\n")
            
            return formatted_result
        except Exception as e:
            print(f"Error in object detection tool: {str(e)}")
            return {
                "object_name": "error",
                "object_category": "error",
                "distinguishing_features": [f"Error processing image: {str(e)}"]
            }