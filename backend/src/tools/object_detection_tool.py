import os
import base64
import requests
import json

from openai import OpenAI
from dotenv import load_dotenv
from io import BytesIO
from langchain.tools import tool

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
                        "role": "system",
                        "content": """You are a precise object detection system. You must respond with a valid JSON object containing exactly these fields:
                        - object_name: string
                        - object_category: string
                        - distinguishing_features: array of strings
                        
                        Example format:
                        {
                            "object_name": "Hammer",
                            "object_category": "Tool",
                            "distinguishing_features": ["Metal head", "Wooden handle", "16 oz marking"]
                        }
                        
                        Rules:
                        1. Response must be ONLY the JSON object, no other text
                        2. No markdown formatting or code blocks
                        3. distinguishing_features must be an array of strings
                        4. All fields are required
                        5. No trailing commas
                        6. Use double quotes for strings"""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image and identify the primary object within it. Return ONLY a JSON object with object_name, object_category, and distinguishing_features fields."
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

            content = response.choices[0].message.content

            print("\n=== Raw LLM Response ===")
            print(f"Content type: {type(content)}")
            print(f"Content: {repr(content)}")

            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            object_dict = json.loads(content)
            
            if isinstance(object_dict.get('distinguishing_features'), str):
                object_dict['distinguishing_features'] = [object_dict['distinguishing_features']]
            
            return object_dict
        
        except Exception as e:
            print(f"Error in object detection tool: {str(e)}")
            return {
                "object_name": "error",
                "object_category": "error",
                "distinguishing_features": [f"Error processing image: {str(e)}"]
            }