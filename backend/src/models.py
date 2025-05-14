from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class ObjectDetectionResult(BaseModel):
    object_name: str = Field(..., description="The name of the object in the image")
    object_category: str = Field(..., description="The category of the object")
    distinguishing_features: List[str] = Field(..., description="Any distinguishing features of the object")

