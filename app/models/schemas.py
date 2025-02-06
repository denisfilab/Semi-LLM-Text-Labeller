from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class TokenUsageBase(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0

class TokenUsageCreate(TokenUsageBase):
    pass

class TokenUsage(TokenUsageBase):
    id: int
    job_id: str

    class Config:
        from_attributes = True

class PipelineJobBase(BaseModel):
    project_id: str
    column_name: str

class PipelineJobCreate(PipelineJobBase):
    pass

class PipelineJob(PipelineJobBase):
    id: str
    project_id: str
    status: str
    current_stage: Optional[str] = None
    embedding_progress: int = 0
    labeling_progress: int = 0
    training_progress: int = 0
    metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    error: Optional[str] = None
    column_name: str
    token_usage: Optional[TokenUsage] = None

    class Config:
        from_attributes = True

class CostEstimate(BaseModel):
    prompt_tokens: int
    estimated_cost: float
    completion_tokens: float
    sample_size: int
    total_tokens: int

class StageStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

class PipelineStage(BaseModel):
    progress: int = 0
    status: StageStatus = StageStatus.PENDING

class PipelineStatus(BaseModel):
    status: str
    stages: Dict[str, PipelineStage] = {
        "Embedding Generation": PipelineStage(),
        "LLM Labeling": PipelineStage(),
        "Model Training": PipelineStage()
    }
    error: Optional[str] = None

    class Config:
        from_attributes = True

class ModelMetrics(BaseModel):
    accuracy: float
    auc_score: float
    confusion_matrix: Dict[str, List[int]]
    sample_counts: Dict[str, int]
    token_usage: Optional[TokenUsage] = None
    flagged_count: int = 0

class LabelUpdate(BaseModel):
    label: str
    confidence: Optional[float] = None

class TextColumn(BaseModel):
    name: str = Field(..., description="Name of the column containing the text data to process")
