from threading import Lock
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from openai import OpenAI
import tiktoken
import chromadb
from typing import Tuple
import os
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pydantic import BaseModel
from datetime import datetime
import time
f = open("api_key.txt", "r")


# ------ Phase 1: Data Selection ------ 
class Phase1Selector:
    def __init__(self, max_embed_tokens=5_000_000, target_sample_size=10_000, random_state=42):
        self.random_state = random_state
        self.tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")
        self.max_tokens = max_embed_tokens  # ~5M token capacity
        self.target_sample = target_sample_size  # Target 10k samples
        
    def calculate_phase1_size(self, df: pd.DataFrame) -> int:
        """Get number of rows that fit within token budget"""
        avg_tokens = df['review'].apply(lambda x: len(self.tokenizer.encode(x))).mean()
        max_rows = int(self.max_tokens / avg_tokens)
        return min(max_rows, len(df), self.target_sample)


# ------ Embedding Storage ------
class VectorDB:
    MAX_BATCH_SIZE = 5461  # ChromaDB limit
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        try:
            self.collection = self.client.get_collection("phase1_embeddings")
        except Exception:
            self.collection = self.client.create_collection("phase1_embeddings")

    def store_embeddings(self, ids: list, texts: list, embeddings: list):
        """Store embeddings in batches with retry logic"""
        total_items = len(texts)
        processed = 0
        
        # Process in batches
        with tqdm(total=total_items, desc="Storing embeddings") as pbar:
            while processed < total_items:
                # Get next batch
                batch_end = min(processed + self.MAX_BATCH_SIZE, total_items)
                batch_ids = ids[processed:batch_end]
                batch_texts = texts[processed:batch_end]
                batch_embeddings = embeddings[processed:batch_end]
                
                # Try to store batch with retries
                max_retries = 3
                retry_delay = 5
                
                for attempt in range(max_retries):
                    try:
                        self.collection.add(
                            ids=batch_ids,
                            documents=batch_texts,
                            embeddings=batch_embeddings
                        )
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise Exception(f"Failed to store batch after {max_retries} attempts: {str(e)}")
                        print(f"\nRetrying batch in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                
                # Update progress
                processed = batch_end
                pbar.update(len(batch_ids))

# ------ Diverse Sampling ------
class DiversitySampler:
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        
    def cluster_sample(self, embeddings: np.ndarray, original_indices, sample_fraction=0.8) -> np.ndarray:
        """Return indices of diverse samples using clustering"""
        clusterer = MiniBatchKMeans(
            n_clusters=self.n_clusters, 
            batch_size=1000,
            random_state=self.random_state
        )
        clusters = clusterer.fit_predict(embeddings)
        
        sampled_positions = []
        for c in np.unique(clusters):
            cluster_indices = np.where(clusters == c)[0]
            sample_size = max(1, int(len(cluster_indices) * sample_fraction))
            
            sampled_positions.extend(
                self.rng.choice(
                    cluster_indices, 
                    size=sample_size,
                    replace=False
                )
            )
            
        return np.array([original_indices[pos] for pos in sampled_positions])
    
class EmbeddingWorker:
    def __init__(self, client, model="text-embedding-3-small", batch_size=100, max_workers=8):
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.max_workers = max_workers
        
    def generate_embeddings(self, texts: list) -> list:
        """Process texts in parallel batches with rate limit handling"""
        
        def extract_wait_time(error_msg):
            import re
            match = re.search(r'Please try again in (\d+\.?\d*)s', str(error_msg))
            return float(match.group(1)) if match else 15
            
        batches = [texts[i:i+self.batch_size] 
                for i in range(0, len(texts), self.batch_size)]
        
        embeddings = []
        failed_batches = {}  # {batch_idx: (batch, attempts)}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i, batch in enumerate(batches):
                futures.append((i, executor.submit(self._process_batch, batch)))
                
            for batch_idx, future in tqdm(enumerate(futures), 
                                        total=len(futures),
                                        desc="Generating embeddings"):
                try:
                    embeddings.extend(future[1].result())
                except Exception as e:
                    error_msg = str(e)
                    if "Rate limit" in error_msg:
                        wait_time = extract_wait_time(error_msg)
                        print(f"\nRate limit hit. Waiting {wait_time}s...")
                        time.sleep(wait_time + 0.05)
                        
                        # Retry the failed batch
                        try:
                            retry_result = self._process_batch(batches[batch_idx])
                            embeddings.extend(retry_result)
                        except Exception as retry_e:
                            print(f"Retry failed: {str(retry_e)}")
                            embeddings.extend([None]*len(batches[batch_idx]))
                    else:
                        print(f"Batch failed: {error_msg}")
                        embeddings.extend([None]*len(batches[batch_idx]))
                        
        return embeddings[:len(texts)]  # Trim to original length
    
    def _process_batch(self, batch: list) -> list:
        """Process single batch through OpenAI API"""
        response = self.client.embeddings.create(
            input=batch,
            model=self.model
        )
        return [item.embedding for item in response.data]

# ------ Training & Evaluation ------
# ------ Updated Training & Evaluation ------
class CompactModelTrainer:
    def __init__(self, model_type='distilbert', output_dir="results"):
        self.model_type = model_type
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def train(self, texts: list, labels: list) -> Tuple:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        if self.model_type == 'distilbert':
            return self._train_distilbert(texts, labels)
        else:
            return self._train_tfidf(texts, labels)
    
    def _train_distilbert(self, texts, labels):
        from transformers import TFDistilBertForSequenceClassification, Trainer, TrainingArguments
        from datasets import Dataset
        
        # Tokenization
        tokenizer = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        encodings = tokenizer(texts, truncation=True, padding=True)
        
        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'label': labels
        }).train_test_split(test_size=0.2)
        
        # Training setup
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        
        training_args = TrainingArguments(
            output_dir=self.run_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            evaluation_strategy="epoch",
            logging_dir=os.path.join(self.run_dir, "logs"),
            logging_steps=10,
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=1)
            return {
                'accuracy': accuracy_score(p.label_ids, preds),
                'auc': roc_auc_score(p.label_ids, p.predictions[:,1])
            }
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_metrics,
        )
        
        # Training
        trainer.train()
        
        # Save artifacts
        model.save_pretrained(self.run_dir)
        tokenizer.save_pretrained(self.run_dir)
        
        # Save metrics
        with open(os.path.join(self.run_dir, "metrics.json"), "w") as f:
            json.dump(trainer.evaluate(), f)
            
        return model, tokenizer

    def _train_tfidf(self, texts, labels):
        # Training
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(texts)
        model = LogisticRegression().fit(X, labels)
        
        # Evaluation
        preds = model.predict_proba(X)[:,1]
        metrics = {
            "train_auc": roc_auc_score(labels, preds),
            "train_accuracy": accuracy_score(labels, model.predict(X))
        }
        
        # Save artifacts
        joblib.dump(model, os.path.join(self.run_dir, "logreg_model.pkl"))
        joblib.dump(vectorizer, os.path.join(self.run_dir, "tfidf_vectorizer.pkl"))
        
        # Save metrics
        with open(os.path.join(self.run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)
            
        # Save visualization
        self._save_confusion_matrix(model, X, labels)
            
        return model, vectorizer

    def _save_confusion_matrix(self, model, X, labels):
        cm = confusion_matrix(labels, model.predict(X))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(os.path.join(self.run_dir, "confusion_matrix.png"))
        plt.close()

class LLMLabeler:
    def __init__(self, client, batch_size=5, max_workers=5, cache_file="llm_labels_cache.csv", 
                 rate_limit=True, wait_time=3, sequential=False, max_retries=1000):
        self.client = client
        self.batch_size = batch_size
        self.max_workers = max_workers if not sequential else 1  # Force 1 worker if sequential
        self.cache_file = cache_file
        self.labels_cache = self._load_cache()
        self.partial_results = []
        self.rate_limit = rate_limit
        self.wait_time = wait_time
        self.sequential = sequential  # New sequential mode flag

        # Initialize token tracking variables and a lock for thread safety
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.token_lock = Lock()
        self.processed_texts = set()  # Track processed texts
        self.total_texts = 0  # Track total texts to process
        self.max_retries = max_retries
        self.failed_batches = {}  # {batch_idx: (batch, attempts)}


    def _process_batch_with_retry(self, batch, batch_idx):
        def extract_wait_time(error_msg):
            import re
            match = re.search(r'Please try again in (\d+\.?\d*)s', str(error_msg))
            return float(match.group(1)) if match else 15

        attempts = 0
        while attempts < self.max_retries:
            try:
                return self._process_batch(batch)
            except Exception as e:
                attempts += 1
                error_msg = str(e)
                
                if "Rate limit" in error_msg:
                    wait_time = extract_wait_time(error_msg)
                    print(f"\nRate limit hit. Waiting {wait_time}s... (Attempt {attempts}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Batch {batch_idx} failed: {error_msg}")
                    if attempts == self.max_retries:
                        return [None] * len(batch), [None] * len(batch), 0, 
        return [None] * len(batch), [None] * len(batch), 0, 0

    def _load_cache(self):
        try:
            cache_df = pd.read_csv(self.cache_file)
            # Ensure required columns exist
            for col in ['text', 'label', 'confidence']:
                if col not in cache_df.columns:
                    cache_df[col] = None
            # Create a dictionary mapping text to a dict of label and confidence
            return cache_df.set_index('text')[['label', 'confidence']].to_dict('index')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return {}

    def _save_cache(self, texts, labels, confidences):
            # Add processed texts to tracking set
            self.processed_texts.update(texts)
            
            # Validate lengths and data
            min_length = min(len(texts), len(labels), len(confidences))
            if min_length < len(texts):
                print(f"Warning: Data length mismatch. Truncating to {min_length} entries")
            
            texts = texts[:min_length]
            labels = labels[:min_length]
            confidences = confidences[:min_length]
            
            # Remove None values
            valid_data = [(t, l, c) for t, l, c in zip(texts, labels, confidences) if l is not None]
            if not valid_data:
                print("No valid data to save")
                return
                
            texts, labels, confidences = zip(*valid_data)
            
            cache_df = pd.DataFrame({
                'text': texts,
                'label': labels,
                'confidence': [c if c is not None else 0.0 for c in confidences]
            })
            
            print(f"Saving {len(cache_df)} entries to cache")

    def label_texts(self, texts: list) -> list:
        self.total_texts = len(texts)
        
        # Initialize results tracking
        new_labels = []
        new_confidences = []
        completed_batches = 0
        
        # Remove duplicates while preserving order
        seen = set()
        unique_texts = []
        for text in texts:
            if text not in seen and text not in self.processed_texts:
                seen.add(text)
                unique_texts.append(text)
        
        print(f"Total texts: {self.total_texts}")
        print(f"Unique unprocessed texts: {len(unique_texts)}")
        
        # Check cache first
        labels = [None] * len(texts)  # Initialize with None
        confidences = [None] * len(texts)
        texts_to_label = []
        cache_positions = {}  # Map text to original positions
        
        for i, text in enumerate(texts):
            if text in self.labels_cache:
                cached = self.labels_cache[text]
                labels[i] = cached['label']
                confidences[i] = cached['confidence']
            elif text not in self.processed_texts:
                texts_to_label.append(text)
                cache_positions[text] = i
        
        print(f"Texts in cache: {len(texts) - len(texts_to_label)}")
        print(f"Texts to label: {len(texts_to_label)}")
        
        if not texts_to_label:
            self._print_token_usage()
            return labels
            
        # Process uncached texts in batches
        batches = []
        current_batch = []
        for text in texts_to_label:
            if len(current_batch) < self.batch_size:
                current_batch.append(text)
            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
                current_batch = []
        if current_batch:  # Add remaining texts
            batches.append(current_batch)
            
        print(f"Number of batches: {len(batches)}")
        print(f"Batch size: {self.batch_size}")
        
        try:
            if self.sequential:
                # Sequential processing
                for batch_idx, batch in enumerate(tqdm(batches, desc="Processing sequentially")):
                    labels_batch, confs_batch, prompt_tokens, comp_tokens = self._process_batch_with_retry(batch, batch_idx)
                    new_labels.extend(labels_batch)
                    new_confidences.extend(confs_batch)
                    
                    self.token_lock.acquire()
                    try:
                        self.prompt_tokens += prompt_tokens
                        self.completion_tokens += comp_tokens
                    finally:
                        self.token_lock.release()
                    
                    if self.rate_limit:
                        time.sleep(self.wait_time)
                        
                    # Save intermediate progress
                    self._save_cache(batch, labels_batch, confs_batch)
                    
            else:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {executor.submit(self._process_batch_with_retry, batch, i): i 
                             for i, batch in enumerate(batches)}
                    
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                        batch_idx = futures[future]
                        batch = batches[batch_idx]
                        
                        try:
                            labels_batch, confs_batch, prompt_tokens, comp_tokens = future.result()
                            
                            # Update results
                            for text, label, conf in zip(batch, labels_batch, confs_batch):
                                if text in cache_positions and label is not None:
                                    pos = cache_positions[text]
                                    labels[pos] = label
                                    confidences[pos] = conf
                            
                            with self.token_lock:
                                self.prompt_tokens += prompt_tokens
                                self.completion_tokens += comp_tokens
                                
                            completed_batches += 1
                            
                            if completed_batches % 5 == 0:
                                valid_labels = [l for l in labels if l is not None]
                                valid_confidences = [c for c in confidences if c is not None]
                                if valid_labels:
                                    self._save_cache(batch, valid_labels, valid_confidences)
                        
                        except Exception as e:
                            print(f"Future failed for batch {batch_idx}: {str(e)}")
                            self.failed_batches[batch_idx] = (batch, 1)
                                    
        except KeyboardInterrupt:
            print("\nInterrupted! Saving progress...")
            self._save_cache(texts_to_label[:completed_batches*self.batch_size],
                            [l for l in labels if l is not None],
                            [c for c in confidences if c is not None])
            self._print_token_usage()
            raise
            
        # Final save
        self._save_cache(texts_to_label, 
                        [l for l in labels if l is not None],
                        [c for c in confidences if c is not None])
        
        self._print_token_usage()
        return labels

    def _print_token_usage(self):
        print(f"Total Prompt Tokens Used: {self.prompt_tokens}")
        print(f"Total Completion Tokens Used: {self.completion_tokens}")
        print(f"Total Tokens Used: {self.prompt_tokens + self.completion_tokens}")

    def _process_batch(self, texts: list) -> tuple:
        """Process single batch through OpenAI API"""
        labels = []
        confidences = []
        batch_prompt_tokens = 0
        batch_completion_tokens = 0
        texts = [t for t in texts if t not in self.processed_texts]
        if not texts:
            return [], [], 0, 0
        FEW_SHOT_EXAMPLES = """
        **Clear Examples**:
        1. Text: "This movie changed my life - perfect in every way!"
        → Sentiment: positive (1.0 confidence)
        Why: Unambiguous superlative praise

        2. Text: "A tedious, poorly-acted mess with no redeeming qualities"
        → Sentiment: negative (0.98 confidence)
        Why: Explicit negative descriptors

        **Complex Positive Examples**:
        3. Text: "This film deals with the Irish rebellion [...] Highly recommended."
        Analysis: Contains critical analysis but concludes with strong recommendation
        → Sentiment: positive (0.85 confidence)
        Key Markers: 
        - "Magnificent performance" (acting praise)
        - "Highly recommended" (explicit endorsement)
        - "Very good film well worth watching" (overall positive)

        4. Text: "So bad it's almost good"
        Analysis: Ironic praise common in cult films
        → Sentiment: positive (0.75 confidence)
        Why: Cultural context awareness

        **Ambiguous Examples**:
        5. Text: "Visually stunning but emotionally hollow"
        Analysis: Mixed technical/artistic merit  
        → Sentiment: negative (0.65 confidence)
        Why: Dominant negative aspect

        **Confidence Guidelines**:
        0.95+ : Unambiguous praise/criticism ("Masterpiece"/"Worst ever")
        0.8-0.94: Clear sentiment with qualifiers ("Flawed but essential")
        0.6-0.79: Requires cultural/contextual interpretation
        <0.6 : Flag for human review

        **New Rules**:
        1. Acknowledge critical analysis within positive reviews
        2. "Highly recommended" = strong positive regardless of qualifiers
        3. Actor praise + "worth watching" = minimum 0.7 confidence
        4. Historical context discussion ≠ negative sentiment
        """

        system_prompt = f"""Act as a professional film critic. {FEW_SHOT_EXAMPLES}
            Analyze the text and respond with JSON containing:
            - sentiment: positive/negative
            - confidence: 0-1"""

        for text in texts:
            try:
                response = self.client.beta.chat.completions.parse(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    response_format=SentimentLabel,
                    temperature=0.0,
                )
                
                response_data = json.loads(response.choices[0].message.content)
                sentiment = response_data.get('sentiment', 'negative')
                confidence = response_data.get('confidence', 0.0)
                
                labels.append(1 if sentiment == "positive" else 0)
                confidences.append(float(confidence))

                # Track token usage
                if hasattr(response, 'usage'):
                    usage = response.usage
                    batch_prompt_tokens += usage.prompt_tokens
                    batch_completion_tokens += usage.completion_tokens

            except Exception as e:
                print(f"Error labeling text: {str(e)}")
                labels.append(None)
                confidences.append(None)

        # Ensure equal lengths
        assert len(labels) == len(confidences) == len(texts), "Mismatched batch processing results"
        
        return labels, confidences, batch_prompt_tokens, batch_completion_tokens
    

# ------ Full Pipeline ------
class ActiveLearningPipeline:
    def __init__(self, cache_dir="cache"):
        self.phase1_selector = Phase1Selector()
        self.vectordb = VectorDB()
        self.sampler = DiversitySampler()
        self.labeler = OpenAI(api_key=f.read())
        self.llm_labeler = LLMLabeler(self.labeler)


        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "llm_labels_cache.csv")
        self.llm_labeler = LLMLabeler(self.labeler, cache_file=cache_file)
    def run(self, input_csv: str):
        print("\n=== Starting Active Learning Pipeline ===")
        start_time = time.time()
        
        # Load data
        print("\n[1/8] Loading dataset...")
        load_start = time.time()
        df = pd.read_csv(input_csv)
        print(f"Original dataset size: {len(df):,} rows")

        print(f"Loaded {len(df)} rows. Time: {time.time()-load_start:.1f}s")

        # Phase 1 Selection
        print("\n[2/8] Selecting initial sample...")
        phase_start = time.time()
        phase1_size = self.phase1_selector.calculate_phase1_size(df)
        phase1_data = df.sample(n=phase1_size, random_state=42)
        print(f"Selected {len(phase1_data)} samples for embedding. Time: {time.time()-phase_start:.1f}s")

        # Embedding Generation
        print("\n[3/8] Generating embeddings...")
        embed_start = time.time()
        worker = EmbeddingWorker(self.labeler)
        texts_to_embed = phase1_data['review'].tolist()
        embeddings = worker.generate_embeddings(texts_to_embed)
        print(f"Generated {len(embeddings)} embeddings. Time: {time.time()-embed_start:.1f}s")

        # Filter and Store
        print("\n[4/8] Storing embeddings...")
        store_start = time.time()
        valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
        phase1_data = phase1_data.iloc[valid_indices]
        embeddings = [embeddings[i] for i in valid_indices]
        self.vectordb.store_embeddings(
            ids=phase1_data.index.astype(str).tolist(),
            texts=phase1_data['review'].tolist(),
            embeddings=embeddings
        )
        print(f"Stored {len(embeddings)} valid embeddings. Time: {time.time()-store_start:.1f}s")
        time.sleep(60)
        # Label all Phase 1 data
        print("\n[5/8] Labeling all Phase 1 data...")
        label_start = time.time()
        labels = self.llm_labeler.label_texts(phase1_data['review'].tolist())
        phase1_data = phase1_data.assign(label=labels).dropna(subset=['label'])
        valid_labels = phase1_data['label'].notnull().sum()
        print(f"Labeled {valid_labels}/{len(phase1_data)} samples. Time: {time.time()-label_start:.1f}s")
        print(f"Processed {len(self.llm_labeler.processed_texts)} unique texts")
        phase1_data = phase1_data.dropna(subset=['label'])

        # Split into train/eval using diverse sampling
        print("\n[6/8] Splitting data into train/eval...")
        split_start = time.time()
        train_indices = self.sampler.cluster_sample(
            np.array(embeddings),
            phase1_data.index.values,
            sample_fraction=0.8
        )
        train_data = phase1_data.loc[train_indices]
        eval_data = phase1_data[~phase1_data.index.isin(train_indices)]
        print(f"Split complete. Train: {len(train_data)}, Eval: {len(eval_data)}. Time: {time.time()-split_start:.1f}s")

        # Training
        print("\n[7/8] Training model...")
        train_start = time.time()
        trainer = CompactModelTrainer(model_type='distilbert')
        model, vectorizer = trainer.train(
            train_data['review'].tolist(),
            train_data['label'].tolist()
        )

        # Evaluate on held-out set
        X_eval = vectorizer.transform(eval_data['review'])
        eval_preds = model.predict_proba(X_eval)[:,1]
        eval_metrics = {
            'eval_auc': roc_auc_score(eval_data['label'], eval_preds),
            'eval_accuracy': accuracy_score(eval_data['label'], eval_preds > 0.5)
        }

        # Save evaluation metrics
        with open(os.path.join(trainer.run_dir, "eval_metrics.json"), "w") as f:
            json.dump(eval_metrics, f)
            
        print(f"Training & evaluation completed. AUC: {eval_metrics['eval_auc']:.3f}")
        print(f"Time: {time.time()-train_start:.1f}s")

        # Inference on complete dataset
        print("\n[8/8] Running inference on complete dataset...")
        final_start = time.time()
        
        # Transform and predict on all data
        X_all = vectorizer.transform(df['review'])
        df['pred'] = model.predict_proba(X_all)[:,1]
        df['flag'] = df['pred'].between(0, 0.7)
        
        # Save complete results
        results_path = os.path.join(trainer.run_dir, "complete_results.csv")
        df.to_csv(results_path, index=True)
        
        # Save model and vectorizer
        model_path = os.path.join(trainer.run_dir, "final_model.pkl")
        vectorizer_path = os.path.join(trainer.run_dir, "final_vectorizer.pkl")
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        print(f"\n=== Pipeline completed in {time.time() - start_time:.1f} seconds ===")
        print(f"Results saved to: {trainer.run_dir}")
        print(f"Flagged {df['flag'].sum()} low-confidence samples")
        print(f"Model performance (AUC): {eval_metrics['eval_auc']:.3f}")
        flagged_data = df[df['flag']].copy()
        flagged_path = os.path.join(trainer.run_dir, "flagged_results.csv")
        flagged_data.to_csv(flagged_path, index=True)
        return {
            'model': model,
            'vectorizer': vectorizer,
            'results_path': results_path,
            'model_path': model_path,
            'vectorizer_path': vectorizer_path,
            'eval_metrics': eval_metrics,
            'run_dir': trainer.run_dir,
            'flagged_data': flagged_data,
            'flagged_path': flagged_path
        }
    
    # def _llm_label(self, text: str) -> int:
    #     response = self.labeler.beta.chat.completions.parse(
    #         model="gpt-4o-mini-2024-07-18",
    #           messages=[
    #             {"role": "system", "content": "You're a sentiment classifier. Analyze this text and respond ONLY with either 'positive' or 'negative' based on its emotional sentimen"},
    #             {"role": "user", "content": text}
    #         ],
    #         response_format=SentimentLabel,
    #         temperature=0.0,  # More deterministic output
    #         max_tokens=1  # Force single-token response
    #     )
    #     response = json.loads(response.choices[0].message.content)
    #     label = response['sentiment']
        
    #     if label == "positive":
    #         return 1
    #     else:
    #         return 0
        
class SentimentLabel(BaseModel):
    sentiment: str
    confidence: float
     
# ------ Main Script ------
if __name__ == "__main__":
    f = open("api_key.txt", "r")
    pipeline = ActiveLearningPipeline()
    result = pipeline.run("IMDB Dataset.csv")
    print(f"Flagged data saved at: {result['run_dir']}")
    print(f"Flagged data shape: {result['flagged_data'].shape}")
