from crew import (
    ObjectAnalysisCrew
)
from models import (
    ObjectDetectionResult
)
import os
import sys
import json

def analyze_object(image_data: str):
    print(f"\n=== Starting Analysis ===")
    print(f"Initial image path: {image_data}")
    print(f"File exists: {os.path.exists(image_data)}")
    print(f"Absolute path: {os.path.abspath(image_data)}")
    
    # Make sure file exists
    if not os.path.exists(image_data):
        print(f"WARNING: Image file does not exist: {image_data}")
    
    crew_instance = ObjectAnalysisCrew(
        image_data=image_data,
    )
    crew_obj = crew_instance.crew()
    
    inputs = {
        "image_input": image_data
    }
    
    print(f"\n=== Passing to Crew ===")
    print(f"Inputs dictionary: {inputs}")
    
    result = crew_obj.kickoff(inputs=inputs)
    raw_json = result.raw

    object_dict = json.loads(raw_json)
    object_result = ObjectDetectionResult(**object_dict)
    
    print("\n=== Object Detection Result ===")
    print(f"Object Name: {object_result.object_name}")
    print(f"Object Category: {object_result.object_category}")
    print(f"Distinguishing Features: {object_result.distinguishing_features}")
    
    return object_result

if __name__ == "__main__":
    # relative path for testing
    test_image = "test.jpg"
    print(f"\n=== Test Setup ===")
    print(f"Test image path: {test_image}")
    
    analyze_object(test_image)