from fastapi import FastAPI, File, Form, Path, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import pandas as pd
import os
from typing import List, Optional
import json
import logging
import traceback
from services.training import TrainingService
from models import schemas, database as db
from services.pipeline import PipelineService
from background.task_manager import task_manager, TaskStatus
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pathlib import Path
import sys
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Create logs directory
LOG_DIR = Path("logss")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s\n%(levelname)s: %(message)s\n%(pathname)s:%(lineno)d\n',
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
    ]
)

logger = logging.getLogger(__name__)
app = FastAPI(
    title="Text Classification Pipeline API",
    # debug=True  # Enable debug mode
)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unexpected error occurred")
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc)
        }
    )


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database session
def get_db():
    database = db.SessionLocal()
    try:
        yield database
    finally:
        database.close()

@app.get("/")
async def root():
    return {"message": "welcome to the"}

from pydantic import BaseModel

class PipelineRequest(BaseModel):
    projectId: str
    columnName: str
    fileId: str
actual_rules = ""
# @app.post("/run-pipeline")
# async def run_pipeline(
#     project_id: str = Form(...),
#     column_name: str = Form(...),
#     file_id: str = Form(...),
#     db_session: Session = Depends(get_db)
# ):
#     """Start the pipeline processing"""
#     # Get the file from the db
#     data_record = crud.get_csv_file_as_df(db_session, file_id, project_id)
#     # Create pipeline job
#     pipeline_service = PipelineService(db_session=db_session)
#     job = pipeline_service.create_job(project_id, column_name)
    
#     # Start background task
#     print("test")
#     task_id = task_manager.submit_task(
#         pipeline_service.run_pipeline,
#         job_id=job.id,
#         rules=actual_rules, 
#         labels=["positive", "negative"],
#         data_record=data_record,
#         db_session=db_session,
#         file_id=file_id,
#         project_id=project_id,
#     )
#     print("Starting task")
#     return {
#         "status": "success",
#         "message": "Pipeline started",
#         "job_id": job.id,
#         "task_id": task_id
#     }
        
@app.post("/run-pipeline")
async def run_pipeline(
    project_id: str = Form(...),
    column_name: str = Form(...),
    file_id: str = Form(...),
    db_session: Session = Depends(get_db)
):
    """Start the pipeline processing"""
    try:
        # Get the file from the db
        data_record = crud.get_csv_file_as_df(db_session, file_id, project_id)
        
        # Create pipeline job
        pipeline_service = PipelineService(db_session=db_session)
        job = pipeline_service.create_job(project_id, column_name)
        
        # Start background task
        print("test")
        task_id = task_manager.submit_task(
            pipeline_service.run_pipeline,
            job_id=job.id,
            rules=actual_rules, 
            labels=["positive", "negative"],
            data_record=data_record,
            db_session=db_session,
            file_id=file_id,
            project_id=project_id,
        )
        print("Starting task")
        return {
            "status": "success",
            "message": "Pipeline started",
            "job_id": job.id,
            "task_id": task_id
        }
    except Exception as e:
        # Log the full exception stacktrace for debugging
        logger.exception("Error in /run-pipeline endpoint")
        # Raise an HTTPException with a 500 status code and the error detail
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline-status/{job_id}")
async def get_pipeline_status(
    job_id: str,
    db_session: Session = Depends(get_db)
):
    """Get current pipeline status"""
    pipeline_service = PipelineService(db_session=db_session)
    job = pipeline_service.get_job_status(job_id)
    return job.model_dump()


@app.get("/metrics/{project_id}/{file_id}")
async def get_metrics(
    project_id: str,
    file_id: str,
):
    """Get metrics from saved file"""
    try:
        metrics_path = f"pipeline_results/project_{project_id}/file_{file_id}/metrics.json"
        
        if not os.path.exists(metrics_path):
            raise HTTPException(
                status_code=404,
                detail="Metrics not found for this project/file combination"
            )
            
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        return metrics
        
    except Exception as e:
        logger.exception("Error fetching metrics")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/estimate-cost")
async def estimate_cost(
    column_name: str = Form(...),  
    project_id: str = Form(...),
    file_id: str = Form(...),
    rules: str = Form(...),
    labels: List[str] = Form(...),
    db_session: Session = Depends(get_db)

):
    """Estimate processing cost for a dataset"""

    # Get the file from the db
    data_record = crud.get_csv_file_as_df(db_session, file_id, project_id)
    print(data_record.columns)

    # Create an instance of PipelineService
    pipeline_service = PipelineService(db_session=db_session)
    if isinstance(labels, str):
        try:
            labels = json.loads(labels)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON for labels: {e}")

    if not isinstance(labels, list):
        raise HTTPException(status_code=400, detail="labels must be a list")
    
    global actual_rules
    actual_rules = rules
    # Pass the bound method to the task manager
    print("submitting task")
    print("rules", rules[:10])
    print("labels", labels)
    task_id = task_manager.submit_task(
        pipeline_service.estimate_cost, # Bound method with `self` already included
        data_record=data_record,
        column_name=column_name,
        labels=labels,
        rules=rules,

    )
        
    # Wait for result with timeout
    task_info = task_manager.wait_for_task(task_id, timeout=30)

    if not task_info or task_info.status != TaskStatus.COMPLETED:
        print("Cost estimation failed")
        raise HTTPException(status_code=500, detail="Cost estimation failed")
    print(task_info.result)
    # Convert to json response
    return task_info.result
    


@app.put("/labels/{row_id}")
async def update_label(
    row_id: int,
    label_update: schemas.LabelUpdate,
    db_session: Session = Depends(get_db)
):
    """Update human label for a row"""
    try:
        db_session.query(db.CsvData).filter(
            db.CsvData.id == row_id
        ).update({
            "human_label": label_update.label,
            "confidence": label_update.confidence
        })
        db_session.commit()
        
        return {"status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

from crud import crud
from models import database
import time

@app.get("/test-db")
def test_db(db: Session = Depends(get_db)):
    project_name = f"Test Project {int(time.time())}"
    project = crud.create_project(db, project_name)
    projects = crud.get_projects(db)
    return {
        "created_project": project.to_dict(),
        "all_projects": [proj.to_dict() for proj in projects]
    }

@app.post("/predict")
async def predict(
    request: Request,
    db_session: Session = Depends(get_db)
):
    try:
        body = await request.json()
        project_id = body.get("project_id")
        file_id = body.get("file_id")
        text = body.get("text")

        if not all([project_id, file_id, text]):
            raise HTTPException(
                status_code=400,
                detail="Missing required fields"
            )

        # Load the model and make prediction
        training_service = TrainingService()
        prediction = training_service.predict(text, project_id, file_id)
        
        # Convert numeric prediction to label string
        with open(f"pipeline_results/project_{project_id}/file_{file_id}/labels.json", "r") as f:
            label_mappings = json.load(f)
            prediction_label = label_mappings["reverse_mapping"][str(prediction["prediction"])]

        return {
            "prediction": prediction_label,
            "confidence": prediction["confidence"]
        }

    except Exception as e:
        logger.exception("Error in prediction endpoint")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        access_log=True
    )
