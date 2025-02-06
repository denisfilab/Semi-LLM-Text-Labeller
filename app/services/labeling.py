from enum import Enum
import json
import logging
import pandas as pd
import tiktoken
from typing import List, Dict, Tuple, Callable
from openai import OpenAI
from threading import Lock
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel
from services.progress import update_progress
from crud import crud
from sqlalchemy.orm import Session

from enum import Enum
from pydantic import BaseModel
from typing import Type

class ResponseModel(BaseModel):
    sentiment: str
    confidence: float

class LabelingService:
    def __init__(
        self, 
        batch_size: int = 5, 
        max_workers: int = 5, 
        rate_limit: bool = True, 
        wait_time: int = 3, 
        max_retries: int = 1000
    ):
        load_dotenv()  # Load environment variables
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.wait_time = wait_time
        self.max_retries = max_retries
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # Token tracking
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.token_lock = Lock()

    async def label_texts(
        self, 
        job_id,
        data_record: pd.DataFrame,
        rules: str,
        labels: List[str],
        db_session: Session
    ) -> Tuple[List[int], Dict[str, int]]:
        """
        Label texts in data_record using LLM with progress tracking.
        
        Parameters:
          data_record: A DataFrame containing at least columns "id" (database row ID) and "text".
          on_progress: A callback that takes an integer progress (0-100) for this stage.
          rules: The system prompt/rules to use for classification.
          class_labels: A list of class labels (not used for labeling directly in this example).
          db_session: A database session to update records.
          
        Returns:
          A tuple of (labels, token_usage).
        """
        # Extract texts and corresponding database row IDs.
        texts = data_record["text"].tolist()
        row_ids = data_record["id"].tolist()
        total_texts = len(texts)

        # Create batches for texts and row IDs (keeping order)
        text_batches = [texts[i:i + self.batch_size] for i in range(0, total_texts, self.batch_size)]
        id_batches = [row_ids[i:i + self.batch_size] for i in range(0, total_texts, self.batch_size)]
        
        print('Starting labeling process...')
        all_labels: List[int] = []
        total_processed = 0
        
        # Use a thread pool for concurrent processing of batches.
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            # Submit each batch for processing.
            for batch in text_batches:
                futures.append(executor.submit(self._process_batch, batch, rules, labels))
            
            # Process results as they complete.
            # Use an index to know which batch's row IDs to update.
            batch_index = 0
           
            for future in tqdm(as_completed(futures), total=len(futures), desc="Labeling batches"):
                try:
                    batch_results, batch_usage = future.result()
                    # Unzip the results
                    batch_labels, batch_confidences = zip(*batch_results)
                    all_labels.extend(batch_labels)
                    
                    # Update token counts
                    with self.token_lock:
                        self.prompt_tokens += batch_usage["prompt_tokens"]
                        self.completion_tokens += batch_usage["completion_tokens"]
                    
                    # Update database for this batch with both labels and confidence scores
                    current_ids = id_batches[batch_index]
                    for rid, (label, confidence) in zip(current_ids, batch_results):
                        crud.update_label_and_confidence(db_session, rid, label, confidence, 'llm_label')
                    db_session.commit()  # Commit after each batch
                    
                    # Update progress
                    total_processed += len(batch_labels)
                    stage_progress = int((total_processed / total_texts) * 100)
                    update_progress(db_session, job_id, "labeling", stage_progress)
                    
                    if self.rate_limit:
                        time.sleep(self.wait_time)
                    
                    batch_index += 1
                except Exception as e:
                    print(f"Batch failed: {str(e)}")
                    failed_count = len(text_batches[batch_index])
                    all_labels.extend([None] * failed_count)
                    batch_index += 1
        
        # Ensure we have labels for all texts.
        if len(all_labels) != total_texts or any(label is None for label in all_labels):
            print(f"Failed to generate labels for all texts: {len(all_labels)} labels for {total_texts} texts")
            raise ValueError("Failed to generate labels for all texts")
            
        # Calculate token usage and cost.
        token_usage = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost": self._calculate_cost(self.prompt_tokens, self.completion_tokens)
        }
        
        return all_labels, token_usage

    def _process_batch(self, texts: List[str], rules: str, labels: List[str]) -> Tuple[List[Tuple[int, float]], Dict[str, int]]:
        """Process a batch and return labels with confidence scores."""
        batch_results: List[Tuple[int, float]] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        for text in texts:
            attempts = 0
            while attempts < self.max_retries:
                try:
                    response = self.client.beta.chat.completions.parse(
                        model="gpt-4o-mini-2024-07-18",  # Fixed model name
                        messages=[
                            {"role": "system", "content": rules},
                            {"role": "user", "content": text}
                        ],
                        response_format=ResponseModel
                    )
                    
                    result = json.loads(response.choices[0].message.content)
                    sentiment = str(result['sentiment'])
                    confidence = float(result['confidence'])
                    label = sentiment                    
                    batch_results.append((label, confidence))
                    total_usage["prompt_tokens"] += response.usage.prompt_tokens
                    total_usage["completion_tokens"] += response.usage.completion_tokens
                    break
                    
                except Exception as e:
                    attempts += 1
                    if "Rate limit" in str(e):
                        wait_time = self._extract_wait_time(str(e))
                        print(f"\nRate limit hit. Waiting {wait_time}s... (Attempt {attempts}/{self.max_retries})")
                        time.sleep(wait_time)
                    else:
                        logging.exception(f"Error processing text '{text}': {str(e)}")
                        if attempts == self.max_retries:
                            raise
                        time.sleep(1)
        
        return batch_results, total_usage

    def _extract_wait_time(self, error_msg: str) -> float:
        """Extract wait time from rate limit error message."""
        import re
        match = re.search(r'Please try again in (\d+\.?\d*)s', error_msg)
        return float(match.group(1)) if match else 15

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> Tuple[float, float]:
        """Calculate cost based on token usage."""
        prompt_rate = 0.15 / 1000000  # $0.15 per 1M tokens
        completion_rate = 0.60 / 1000000  # $0.60 per 1M tokens
        return (prompt_tokens * prompt_rate, completion_tokens * completion_rate)

    def get_completion_tokens(self, labels: List[int], df_length: int) -> int:
        """Estimate the number of completion tokens for the given labels and dataset size."""
        try:
            token_count = 0
            for label in labels:
                temp_response = f"""{{"sentiment": "{label}"}}"""
                token_count += len(self.tokenizer.encode(temp_response))
            return int(token_count / len(labels) * df_length)
        except Exception as e:
            print(f"Error calculating completion tokens: {e}")
            raise

    def get_prompt_tokens(self, texts: pd.Series, rules: str) -> int:
        """Estimate the number of prompt tokens for the given texts and rules."""
        token_count = 0
        for text in texts:
            temp_response = f"""PROMPT: {rules} {text}"""
            token_count += len(self.tokenizer.encode(temp_response))
        return int(token_count / len(texts))

    def count_tokens(self, texts: List[str]) -> List[int]:
        """Count tokens for a list of texts."""
        print('Counting tokens...')
        return [len(self.tokenizer.encode(text)) for text in texts]
