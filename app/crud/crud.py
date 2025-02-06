import time
from sqlalchemy.orm import Session
from models.database import CsvFile, Project

def create_project(db: Session, name: str):
    project_id = str(int(time.time()))
    project = Project(id=project_id, name=name)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project

def get_projects(db: Session):
    return db.query(Project).order_by(Project.created_at.desc()).all()

def get_project(db: Session, project_id: str):
    return db.query(Project).filter(Project.id == id).first()

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

def get_csv_file_as_df(db: Session, file_id: str, project_id: str) -> pd.DataFrame:
    # Retrieve the CsvFile record from the database.
    file_record = db.query(CsvFile).filter(
        CsvFile.id == file_id,
        CsvFile.project_id == project_id
    ).first()

    if not file_record:
        raise HTTPException(status_code=404, detail="CSV file not found")
    
    if not file_record.csv_data:
        raise HTTPException(status_code=404, detail="CSV data not found for the file")

    # Convert each CsvData row to a dictionary.
    data_rows = [csv_data_row.to_dict() for csv_data_row in file_record.csv_data]

    df = pd.DataFrame(data_rows)
    return df

def update_label_and_confidence(db: Session, row_id: int, label: str, confidence: float, label_type: str):
    """Update both LLM label and confidence score for a CSV data row."""
    from models.database import CsvData  # Import at function level
    
    csv_data = db.query(CsvData).filter(CsvData.id == row_id).first()
    if csv_data:
        if label_type in ['llm_label', 'model_label', 'human_label', 'final_label']:
            setattr(csv_data, label_type, label)
        csv_data.confidence_score = confidence
        db.flush()  # Flush changes but don't commit yet
    return csv_data