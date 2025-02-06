# progress.py
import logging
import traceback
from sqlalchemy.orm import Session
from models.database import PipelineJob

# Configure a module-level logger
logger = logging.getLogger(__name__)

def update_progress(db_session: Session, job_id: str, stage: str, progress: int) -> None:
    """Update progress for a specific stage"""
    try:
        # Get the job from the database
        job = db_session.query(PipelineJob).filter(PipelineJob.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            raise ValueError(f"Job {job_id} not found")
        print("Job found")

        # Update stage-specific progress
        match stage:
            case "embedding":
                job.embedding_progress = progress
                if progress == 100:
                    job.current_stage = "labeling"
            case "labeling":
                job.labeling_progress = progress
                if progress == 100:
                    job.current_stage = "training"
            case "training":
                job.training_progress = progress
                if progress == 100:
                    job.status = "completed"
                    
        # Update overall job status
        if progress < 100:
            job.status = "running"
            job.current_stage = stage
            
        db_session.commit()
        
        # Log the update for debugging
        logger.info(f"Updated job {job_id} - Stage: {stage}, Progress: {progress}, Current Stage: {job.current_stage}, Status: {job.status}")
        print("Progress updated")
    except Exception as e:
        db_session.rollback()  # Rollback any changes on error
        logger.error(f"Error updating progress for job {job_id}: {str(e)}")
        raise ValueError(f"Failed to update progress: {str(e)}")
