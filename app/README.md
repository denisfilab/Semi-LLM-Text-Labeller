# Backend Service (FastAPI)

The backend service for the Semi LLM Text Labeller project, built with FastAPI and Python.

## Directory Structure

```
.
├── api/                # API routes and endpoints
├── background/        # Background task processing
├── crud/             # Database operations
├── models/           # Database models and Pydantic schemas
├── services/         # Business logic and core services
├── logss/            # Application logs
├── pipeline_results/ # Processing results
└── uploads/         # Temporary file storage
```

## Setup Instructions

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Edit the `.env` file with your configuration:
```env
OPENAI_API_KEY=your-api-key
```

4. Initialize the database:
```bash
python -m models.database
```

5. Start the server:
```bash
uvicorn main:app --reload --port 8000
```

## API Documentation

### Services
- `labeling.py`: Text labeling logic using LLMs
- `embeddings.py`: Text embedding generation
- `training.py`: Model training and fine-tuning
- `pipeline.py`: Data processing pipeline
- `progress.py`: Progress tracking

