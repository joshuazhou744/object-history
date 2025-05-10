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

load_dotenv()

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
                # Try to use a default test image
                image_input = os.path.join(os.path.dirname(__file__), "test.jpg")
                print(f"Using default image: {image_input}")
                
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
                    
            print(f"Using image path: {image_input}")  # Debug print
                
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

            client = OpenAI()
            response = client.responses.create(
                model="gpt-4.1-nano",
                input=[   
                    {"role": "user", "content": f"Analyze this image and identify the primary object within it. Provide the object name, category, and any distinguishing features. Return the response in JSON format with fields: object_name, object_category, and distinguishing_features. Image: {encoded_image}"}         
                ]
            )
            
            # Parse the response from the correct structure
            content_text = response.output[0].content[0].text
            content_text = content_text.replace('```json\n', '').replace('\n```', '')
            result = json.loads(content_text)
            print(f"API result: {result}")
            
            return {
                "object_name": result.get("object_name", "unknown"),
                "object_category": result.get("object_category", "unknown"),
                "distinguishing_features": result.get("distinguishing_features", [])
            }
        except Exception as e:
            print(f"Error in object detection tool: {str(e)}")
            return {
                "object_name": "error",
                "object_category": "error",
                "distinguishing_features": [f"Error processing image: {str(e)}"]
            }