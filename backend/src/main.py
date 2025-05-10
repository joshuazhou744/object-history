from crew import (
    ObjectAnalysisCrew
)
import os
import sys

def analyze_object(image_data: str):
    print(f"\n=== Starting Analysis ===")
    print(f"Initial image path: {image_data}")
    print(f"File exists: {os.path.exists(image_data)}")
    print(f"Absolute path: {os.path.abspath(image_data)}")
    
    # Make sure file exists
    if not os.path.exists(image_data):
        print(f"WARNING: Image file does not exist: {image_data}")
    
    crew_instance = ObjectAnalysisCrew(image_data=image_data)
    crew_obj = crew_instance.crew()
    
    # Create inputs dictionary to pass at kickoff time
    inputs = {
        "image_input": None  # This is the key - passing image path at kickoff
    }
    
    print(f"\n=== Passing to Crew ===")
    print(f"Inputs dictionary: {inputs}")
    
    result = crew_obj.kickoff(inputs=inputs)
    return result

if __name__ == "__main__":
    # Use absolute path for testing
    test_image = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/U7sAx8ZP1dn8HpOcMO_wcg/test.jpg"
    print(f"\n=== Test Setup ===")
    print(f"Test image path: {test_image}")
    
    analyze_object(test_image)