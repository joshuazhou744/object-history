# Object History

An application that guides you from a simple photo of any object to a vivid, time-traveling showcase of its evolution.

## Overview

Object History uses AI to analyze an image of an object, research its historical evolution, and create an engaging timeline showcasing key milestones in the object's development through history.

When you upload an image, the system:

1. Recognizes what you've shown
2. Searches through scholarly articles, historical records, and crowd-sourced sources
3. Identifies the most significant moments when that object changed over time
4. Creates concise, user-friendly summaries of each milestone
5. Generates period-specific illustrations for each milestone
6. Presents everything in an interactive timeline

## Architecture

The application uses a CrewAI framework with four specialized agents:

- **Object Detection Agent**: Identifies objects in uploaded images using GPT-4 Vision
- **Historical Research Agent**: Uses RAG techniques to find information about the object's evolution
- **Content Formatting Agent**: Transforms raw historical data into engaging narratives and detailed visual descriptions
- **Image Generation Agent**: Creates historically accurate illustrations using DALL-E

## Setup

### Backend

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - On Windows: `.venv\Scripts\activate`
   - On macOS/Linux: `source .venv/bin/activate`

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the backend directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SERPER_API_KEY=your_serper_api_key
   ```

6. Run the server:
   ```
   python src/main.py
   ```

### Frontend

A React-based frontend will be created in a subsequent phase.

## API Endpoints

- **POST /process-image/**: Upload an image to analyze and create a timeline
- **POST /demo/**: Get a demo timeline (for testing without running the full AI pipeline)

## Project Structure

```
object-history/
├── backend/
│   ├── src/
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── object_detection.py
│   │   │   ├── historical_research.py
│   │   │   ├── content_formatting.py
│   │   │   └── image_generation.py
│   │   ├── config/
│   │   │   ├── agents.yaml
│   │   │   ├── tasks.yaml
│   │   │   └── models.yaml
│   │   └── main.py
│   ├── examples/
│   └── requirements.txt
├── frontend/
└── README.md
```

## Future Enhancements

- Add a frontend user interface with React
- Implement caching for object research to improve performance
- Expand the research sources and databases for historical information
- Add user accounts to save and share object histories
- Support comparing multiple objects in the same timeline