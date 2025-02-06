import numpy as np
from typing import List, Callable
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from dotenv import load_dotenv

import os

class EmbeddingService:
    def __init__(self, model="text-embedding-3-small", batch_size=100, max_workers=8):
        load_dotenv()  # Load environment variables

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.batch_size = batch_size
        self.max_workers = max_workers

    async def generate_embeddings(
        self, 
        texts: List[str], 
        on_progress: Callable[[int], None]
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts with progress tracking"""
        
        def extract_wait_time(error_msg: str) -> float:
            import re
            match = re.search(r'Please try again in (\d+\.?\d*)s', str(error_msg))
            return float(match.group(1)) if match else 15
        print('testsss')
        # Split texts into batches
        batches = [
            texts[i:i + self.batch_size] 
            for i in range(0, len(texts), self.batch_size)
        ]
        
        embeddings = []
        total_processed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for batch in batches:
                futures.append(executor.submit(self._process_batch, batch))
                
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    batch_embeddings = future.result()
                    embeddings.extend(batch_embeddings)
                    
                    # Update progress
                    total_processed += len(batch_embeddings)
                    progress = int((total_processed / len(texts)) * 100)
                    on_progress(progress)
                    
                except Exception as e:
                    error_msg = str(e)
                    if "Rate limit" in error_msg:
                        wait_time = extract_wait_time(error_msg)
                        print(f"\nRate limit hit. Waiting {wait_time}s...")
                        time.sleep(wait_time + 0.05)
                        
                        # Retry the failed batch
                        try:
                            retry_result = self._process_batch(batch)
                            embeddings.extend(retry_result)
                            
                            # Update progress for retried batch
                            total_processed += len(retry_result)
                            progress = int((total_processed / len(texts)) * 100)
                            on_progress(progress)
                            
                        except Exception as retry_e:
                            print(f"Retry failed: {str(retry_e)}")
                            embeddings.extend([None] * len(batch))
                    else:
                        print(f"Batch failed: {error_msg}")
                        embeddings.extend([None] * len(batch))
                        
        # Filter out None values and ensure we have embeddings for all texts
        valid_embeddings = [e for e in embeddings if e is not None]
        if len(valid_embeddings) != len(texts):
            raise ValueError(
                f"Failed to generate embeddings for all texts. "
                f"Got {len(valid_embeddings)} valid embeddings for {len(texts)} texts."
            )
        print(f"Generated {len(valid_embeddings)} embeddings for {len(texts)} texts")
        return valid_embeddings
    
    def _process_batch(self, batch: List[str]) -> List[List[float]]:
        """Process a single batch through the OpenAI API"""
        response = self.client.embeddings.create(
            input=batch,
            model=self.model
        )
        return [item.embedding for item in response.data]

    def store_embeddings(self, embeddings: List[List[float]], file_path: str) -> None:
        """Store embeddings to a file"""
        np.save(file_path, np.array(embeddings))

    def load_embeddings(self, file_path: str) -> List[List[float]]:
        """Load embeddings from a file"""
        return np.load(file_path).tolist()
