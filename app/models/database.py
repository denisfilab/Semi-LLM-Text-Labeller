from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import relationship

SQLALCHEMY_DATABASE_URL = "sqlite:///../database/data.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class PipelineJob(Base):
    __tablename__ = "pipeline_jobs"

    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False)
    status = Column(String, nullable=False)  # pending, running, completed, failed
    current_stage = Column(String)  # embedding, labeling, training, etc.
    embedding_progress = Column(Integer, default=0)
    labeling_progress = Column(Integer, default=0)
    training_progress = Column(Integer, default=0)
    metrics = Column(JSON)  
    created_at = Column(DateTime, default=datetime.utcnow)
    error = Column(String)
    column_name = Column(String, nullable=False)  # Store which column contains the text data

    token_usage = relationship("TokenUsage", back_populates="job", uselist=False)

class TokenUsage(Base):
    __tablename__ = "token_usage"

    id = Column(Integer, primary_key=True)
    job_id = Column(String, ForeignKey("pipeline_jobs.id"))
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    
    job = relationship("PipelineJob", back_populates="token_usage")

class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.current_timestamp())

    # Optional: Define a relationship to csv_files if needed
    csv_files = relationship("CsvFile", back_populates="project")

    def to_dict(self):
        return {"id": self.id, "name": self.name, "created_at": self.created_at.isoformat()}

class CsvFile(Base):
    __tablename__ = "csv_files"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.current_timestamp())

    project = relationship("Project", back_populates="csv_files")
    csv_data = relationship("CsvData", back_populates="csv_file")
    classification_rule = relationship("ClassificationRule", uselist=False, back_populates="csv_file")

    def to_dict(self):
        return {"id": self.id, "project_id": self.project_id, "filename": self.filename, "created_at": self.created_at.isoformat()}

class CsvData(Base):
    __tablename__ = "csv_data"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    csv_file_id = Column(Integer, ForeignKey("csv_files.id"), nullable=False)
    text = Column(String, nullable=False)
    llm_label = Column(String, nullable=True)
    model_label = Column(String, nullable=True)
    human_label = Column(String, nullable=True)
    confindence = Column(Float, nullable=True)
    final_label = Column(String, nullable=True)


    csv_file = relationship("CsvFile", back_populates="csv_data")

    def to_dict(self):
        return {
            "id": self.id,
            "csv_file_id": self.csv_file_id,
            "text": self.text,
            "llm_label": self.llm_label,
            "model_label": self.model_label,
            "human_label": self.human_label,
            "final_label": self.final_label,
        }

class ClassLabel(Base):
    __tablename__ = "class_labels"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    csv_file_id = Column(Integer, ForeignKey("csv_files.id"), nullable=False)
    label = Column(String, nullable=False)

    def to_dict(self):
        return {"id": self.id, "csv_file_id": self.csv_file_id, "label": self.label}

class ClassificationRule(Base):
    __tablename__ = "classification_rules"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    csv_file_id = Column(Integer, ForeignKey("csv_files.id"), nullable=False, unique=True)
    rules = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.current_timestamp())

    csv_file = relationship("CsvFile", back_populates="classification_rule")

    def to_dict(self):
        return {"id": self.id, "csv_file_id": self.csv_file_id, "rules": self.rules, "created_at": self.created_at.isoformat()}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all tables
Base.metadata.create_all(bind=engine)
