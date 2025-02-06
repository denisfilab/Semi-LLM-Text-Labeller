import logging
import os
import json
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
from sqlalchemy.orm import Session
from crud import crud
from services.progress import update_progress
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self, output_dir="pipeline_results"):
        """Initialize the training service with project and file specific directories"""
        self.base_output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.confidence_threshold = 0.7
        self.label_mapping = {}
        self.reverse_mapping = {}

    def _create_label_mapping(self, labels: List[str]) -> None:
        """Create mapping between label strings and integers"""
        self.label_mapping = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.reverse_mapping = {idx: label for label, idx in self.label_mapping.items()}

    def _encode_labels(self, labels: List[str]) -> np.ndarray:
        """Convert string labels to numeric"""
        return np.array([self.label_mapping[label] for label in labels])

    def _decode_label(self, label_idx: int) -> str:
        """Convert numeric label back to string"""
        return self.reverse_mapping[label_idx]
    
    def _get_run_dir(self, project_id: str, file_id: str) -> str:
        """Get the directory path for a specific project and file"""
        project_dir = os.path.join(self.base_output_dir, f"project_{project_id}")
        run_dir = os.path.join(project_dir, f"file_{file_id}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    
    def _save_artifacts(self, run_dir: str, model, vectorizer, metrics: Dict, y_test, test_preds) -> None:
        """Save all artifacts in the specified directory"""
        # Save model and vectorizer
        joblib.dump(model, os.path.join(run_dir, 'model.pkl'))
        joblib.dump(vectorizer, os.path.join(run_dir, 'vectorizer.pkl'))
        
        # Save label mappings
        with open(os.path.join(run_dir, 'labels.json'), 'w') as f:
            json.dump({
                "label_mapping": self.label_mapping,
                "reverse_mapping": self.reverse_mapping
            }, f, indent=2)
            
        # Save confusion matrix visualization
        self._save_confusion_matrix(y_test, test_preds, run_dir)
        
        # Save metrics
        with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

    async def train_and_evaluate(
        self,
        file_id: str,
        project_id: str,
        labels: List[str],
        db_session: Session,
        job_id: str
    ) -> Dict:
        """Train model and evaluate performance"""
        logger.info(f"Starting training for job {job_id}")
        try:
            self._create_label_mapping(labels)
            update_progress(db_session, job_id, "training", 10)
            
            data_record = crud.get_csv_file_as_df(db_session, file_id, project_id)
            if data_record.empty:
                raise ValueError("No data found in the record")
            
 
            data_record['target'] = data_record.apply(
                lambda row: row['human_label'] if pd.notna(row['human_label']) else row['llm_label'],
                axis=1
            )
            # Filter rows: only keep those with a target label in the allowed labels list.
            filtered_data = data_record[data_record['target'].isin(labels)]
            if filtered_data.empty:
                raise ValueError("No valid labels found after filtering")
            
            # Generate aligned label and text lists
            y = self._encode_labels(filtered_data['target'].tolist())
            texts = filtered_data['text'].tolist()
            
            X_train, X_test, y_train, y_test = train_test_split(
                texts, y, test_size=0.2, random_state=42, stratify=y
            )
            update_progress(db_session, job_id, "training", 20)

            vectorizer = TfidfVectorizer(max_features=1000)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            update_progress(db_session, job_id, "training", 40)

            model = LogisticRegression(random_state=42)
            model.fit(X_train_tfidf, y_train)
            update_progress(db_session, job_id, "training", 60)

            test_preds = model.predict(X_test_tfidf)
            test_probs = model.predict_proba(X_test_tfidf)
            metrics = self._calculate_metrics(y_test, test_preds, test_probs[:, 1], len(y))
            update_progress(db_session, job_id, "training", 80)

            # Get the run directory for this project and file
            run_dir = self._get_run_dir(project_id, file_id)
            
            # Save all artifacts
            self._save_artifacts(
                run_dir=run_dir,
                model=model,
                vectorizer=vectorizer,
                metrics=metrics,
                y_test=y_test,
                test_preds=test_preds
            )

            # Full Dataset Inference
            X_full = vectorizer.transform(texts)
            probabilities = model.predict_proba(X_full)
            predictions = model.predict(X_full)

            # Step 11: Update Database
            try:
                for i, (idx, row) in enumerate(data_record.iterrows()):
                    crud.update_label_and_confidence(
                        db_session, 
                        row['id'], 
                        self._decode_label(int(predictions[i])),
                        float(probabilities[i].max()),
                        'model_label'
                    )
                db_session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Database update failed: {str(e)}")
                db_session.rollback()
                raise

            update_progress(db_session, job_id, "training", 100)
            logger.info(f"Training completed successfully for job {job_id}")
            return metrics

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            db_session.rollback()
            raise
        
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_prob: np.ndarray, total_samples: int) -> Dict:
        """Calculate and return model performance metrics"""
        cm = confusion_matrix(y_true, y_pred)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "auc_score": float(roc_auc_score(y_true, y_prob)),
            "confusion_matrix": {
                "true_negative": int(cm[0][0]),
                "false_positive": int(cm[0][1]),
                "false_negative": int(cm[1][0]),
                "true_positive": int(cm[1][1])
            },
            "sample_counts": {
                "train": int(0.8 * total_samples),
                "test": int(0.2 * total_samples),
                "total": total_samples
            }
        }

    def _save_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                save_dir: str) -> None:
        """Generate and save confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xticks([0, 1], ['Negative', 'Positive'])
        plt.yticks([0, 1], ['Negative', 'Positive'])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()

    def load_model(self, run_dir: str) -> Tuple:
        """Load saved model and vectorizer"""
        model = joblib.load(os.path.join(run_dir, 'model.pkl'))
        vectorizer = joblib.load(os.path.join(run_dir, 'vectorizer.pkl'))
        return model, vectorizer
    
    def predict(self, text: str, project_id: str, file_id: str) -> Dict[str, any]:
        """Predict class for input text using saved model"""
        try:
            # Get the run directory for this project and file
            run_dir = self._get_run_dir(project_id, file_id)
            
            # Load models
            model_path = os.path.join(run_dir, 'model.pkl')
            vectorizer_path = os.path.join(run_dir, 'vectorizer.pkl')
            
            if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
                raise FileNotFoundError(f"Model files not found in directory: {run_dir}")
                
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)

            # Transform text and get prediction
            X = vectorizer.transform([text])
            probabilities = model.predict_proba(X)[0]
            prediction = model.predict(X)[0]
            
            confidence = float(max(probabilities))
            
            return {
                "prediction": int(prediction),
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise