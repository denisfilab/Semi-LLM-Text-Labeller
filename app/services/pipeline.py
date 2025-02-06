import asyncio
import os
import json
import time
from fastapi import Depends
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from services.progress import update_progress
from models import database as db
from models import schemas
from services.embeddings import EmbeddingService
from services.labeling import LabelingService
from services.training import TrainingService

class PipelineService:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.embedding_service = EmbeddingService()
        self.labeling_service = LabelingService()
        self.training_service = TrainingService()
        
        # Create results directory
        self.results_dir = "pipeline_results"
        os.makedirs(self.results_dir, exist_ok=True)

    def create_job(self, project_id: str, column_name: str) -> schemas.PipelineJob:
        """Create a new pipeline job"""
        job = db.PipelineJob(
            id=datetime.now().strftime("%Y%m%d-%H%M%S"),
            project_id=project_id,
            status="pending",
            column_name=column_name,
            current_stage="embedding",
            # Initialize all progress fields to 0
            embedding_progress=0,
            labeling_progress=0,
            training_progress=0,
            error=None
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job
    def _save_token_usage(self, job_id: str, usage: dict) -> None:
        """Save token usage information"""
        token_usage = db.TokenUsage(
            job_id=job_id,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cost=usage.get("cost", 0.0)
        )
        self.db.add(token_usage)
        self.db.commit()

    def get_job_status(self, job_id: str) -> schemas.PipelineStatus:
        """Get current job status"""
        job = self.db.query(db.PipelineJob).filter(db.PipelineJob.id == job_id).first()
        if not job:
            raise Exception("Job not found")
        return schemas.PipelineStatus(
            status=job.status,
            current_stage=job.current_stage,
            progress=job.progress,
            error=job.error
        )
    
    def estimate_cost(
        self,
        data_record: pd.DataFrame,
        column_name: str,
        labels: List[str],
        rules: str,
        sample_size: int = 100,
    ) -> schemas.CostEstimate:
        """Estimate token usage and cost for processing the dataset"""        
        calculated_sample_size = int(round(0.3 * len(data_record)))
        sample_size = calculated_sample_size
        sample_size = 10
        sample = data_record[column_name].sample(n=min(sample_size, len(data_record)))
        prompt_tokens = self.labeling_service.get_prompt_tokens(sample, rules)

        completion_tokens = self.labeling_service.get_completion_tokens(labels, len(sample))
        prompt_token_cost, completion_tokens_cost = self.labeling_service._calculate_cost(prompt_tokens, completion_tokens)
        return schemas.CostEstimate(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=float(completion_tokens),
            estimated_cost=float(prompt_token_cost + completion_tokens_cost),
            total_tokens=int(prompt_tokens + completion_tokens),
            sample_size=len(sample),
        )

    

    def get_completion_token(self, labels: List[str], df: pd.DataFrame) -> int:
        """Get the number of completion token for a given df"""
        token_count = 0
        for label in labels:
            temp_response = f"""{"sentiment": {label}}"""
            # Calculate the token
            self.tokenizer.encode(temp_response)
            token_count += len(self.tokenizer.encode(temp_response))
        return token_count/len(labels) * len(df)
    
    def get_prompt_token(self, df: pd.DataFrame) -> int:
        """Get the number of prompt token for a given df"""
        token_count = 0
        for text in df:
            temp_response = f"""{"role": "user", "content": {text}}"""
            # Calculate the token
            self.tokenizer.encode(temp_response)
            token_count += len(self.tokenizer.encode(temp_response))
        return token_count/len(df)

    def run_pipeline(self, job_id: str, rules: str, labels: List[str], data_record: pd.DataFrame, db_session: Session, file_id: str, project_id:str ) -> None:
        """Run the complete pipeline synchronously and update progress"""

        job_status = self.get_job_status(job_id)
        job = self.db.query(db.PipelineJob).filter(db.PipelineJob.id == job_id).first()
        column = 'text'
        data_record = data_record.sample(n=min(10, len(data_record)))
        print(f"Running Embedding generation for column: {column}")
        # Step 1: Embedding generation
        update_progress(db_session, job_id, "embedding", 10)

        texts = data_record['text'].tolist()
        print('123')
        embeddings = asyncio.run(
            self.embedding_service.generate_embeddings(
                texts, lambda p: update_progress(db_session, job_id, "embedding", 10)  # from 10 to 100
            )
        )
        update_progress(db_session, job_id, "embedding", 100)
        
        print("Running Labeling")

        update_progress(db_session, job_id, "labeling", 0)
        # Step 2: Labeling
        _ , token_usage = asyncio.run(
            self.labeling_service.label_texts(
                job_id, data_record, rules, labels, db_session,
            )
        )
        
        update_progress(db_session, job_id, "labeling", 100)

        print("Running Training & Evaluation")
        update_progress(db_session, job_id, "training", 0)
        # Step 3: Training & Evaluation
        metrics = asyncio.run(
            self.training_service.train_and_evaluate(
                file_id, project_id, labels, db_session, job_id
            )
        )        
        update_progress(db_session, job_id, "training", 100)
        # Update job status to completed with metrics
        job.status = "completed"
        job.metrics = metrics
        self._save_token_usage(job_id, token_usage)
        self.db.commit()

    


    def _save_token_usage(self, job_id: str, usage: Dict[str, int]) -> None:
        """Save token usage information"""
        token_usage = db.TokenUsage(
            job_id=job_id,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cost=usage.get("cost", 0.0)
        )
        self.db.add(token_usage)
        self.db.commit()

    def get_job_status(self, job_id: str) -> schemas.PipelineStatus:
        """Get current job status"""
        job = self.db.query(db.PipelineJob).filter(db.PipelineJob.id == job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")

        def get_stage_status(progress: int) -> schemas.StageStatus:
            if progress == 100:
                return schemas.StageStatus.COMPLETED
            elif progress > 0:
                return schemas.StageStatus.IN_PROGRESS
            return schemas.StageStatus.PENDING

        # Transform the status into the required format with stage-specific progress
        return schemas.PipelineStatus(
            status=job.status,
            stages={
                "Embedding Generation": schemas.PipelineStage(
                    progress=job.embedding_progress,
                    status=get_stage_status(job.embedding_progress)
                ),
                "LLM Labeling": schemas.PipelineStage(
                    progress=job.labeling_progress,
                    status=get_stage_status(job.labeling_progress)
                ),
                "Model Training": schemas.PipelineStage(
                    progress=job.training_progress,
                    status=get_stage_status(job.training_progress)
                )
            },
            error=job.error
        )

    def get_metrics(self, job_id: str) -> schemas.ModelMetrics:
        """Get job metrics"""
        job = self.db.query(db.PipelineJob).filter(db.PipelineJob.id == job_id).first()
        if not job or not job.metrics:
            raise ValueError(f"No metrics found for job {job_id}")

        return schemas.ModelMetrics(
            accuracy=job.metrics["accuracy"],
            auc_score=job.metrics["auc_score"],
            confusion_matrix=job.metrics["confusion_matrix"],
            sample_counts=job.metrics["sample_counts"],
            token_usage=job.token_usage,
            flagged_count=job.metrics.get("flagged_count", 0)
        )

