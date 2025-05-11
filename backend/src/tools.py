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
                # Use the direct URL to the hammer image
                image_input = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/U7sAx8ZP1dn8HpOcMO_wcg/test.jpg"
                print(f"Using default image URL: {image_input}")
                    
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

            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this image and identify the primary object within it. "
                                    "Provide the object name, category, and any distinguishing features. "
                                    "Return the response in JSON format with fields: "
                                    "object_name, object_category, and distinguishing_features."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            # Get the content from the response
            content = response.choices[0].message.content
            
            # Clean up the content string by removing markdown code block markers
            content = content.replace('```json\n', '').replace('\n```', '')
            
            # Parse the JSON string into a Python dictionary
            result = json.loads(content)
            print(f"Parsed result: {result}")
            
            # Convert distinguishing_features to a list if it's a string
            if isinstance(result.get('distinguishing_features'), str):
                result['distinguishing_features'] = [result['distinguishing_features']]
            
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