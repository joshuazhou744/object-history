import os
import yaml
import base64
from pathlib import Path

from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task

from tools import ObjectDetectionTool
from models import ObjectDetectionResult


@CrewBase
class BaseObjectAnalysisCrew:
    agents_config_path = os.path.join(os.path.dirname(__file__), "config", "agents.yaml")
    tasks_config_path = os.path.join(os.path.dirname(__file__), "config", "tasks.yaml")

    def __init__(self, image_data: str):
        # Convert to absolute path if it's a relative path
        self.image_path = image_data
        print(f"Using image path: {self.image_path}")  # Debug print

        with open(self.agents_config_path, "r") as f:
            self.agents_config = yaml.safe_load(f)

        with open(self.tasks_config_path, "r") as f:
            self.tasks_config = yaml.safe_load(f)

    def __del__(self):
        # Clean up the temporary file when the object is destroyed
        if hasattr(self, 'temp_file'):
            os.unlink(self.image_path)

    @agent
    def object_detection_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["object_detection_agent"],
            tools=[ObjectDetectionTool.detect_object],
            allow_delegation=False,
            max_iter=1,
            verbose=True
        )
    
    @task
    def object_detection_task(self) -> Task:
        task_config = self.tasks_config["detect_object_task"]

        expected_output = task_config["expected_output"]
        if not isinstance(expected_output, str):
            expected_output = "Object detection result including name, category, and features"

        def input_handler(inputs):
            print(f"Task received inputs: {inputs}")  # Debug print
            image_path = inputs.get("image_input", self.image_path)
            print(f"Task using image path: {image_path}")  # Debug print
            return image_path

        # Add the image path to the task description to force the agent to use it
        task_description = task_config["description"] + f"\nUse this EXACT image path: {self.image_path}"

        return Task(
            description=task_description,
            agent=self.object_detection_agent(),
            expected_output=expected_output,
            input_data=input_handler
        )

@CrewBase
class ObjectAnalysisCrew(BaseObjectAnalysisCrew):
    @crew
    def crew(self) -> Crew:
        tasks = [
            self.object_detection_task()
        ]

        agents = [
            self.object_detection_agent()
        ]

        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )