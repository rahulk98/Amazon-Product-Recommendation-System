import os
import math
import json
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from prepare_data import load_and_prepare_data
from models import MLPModel


class RecommenderTrainer:
    def __init__(self):
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.n_users = None
        self.n_items = None 
        self.n_main_cats = None
        self.user2idx = None
        self.item2idx = None
        self.idx2item = None
        self.cat2idx = None
        self.title_embeddings = None
        
    def load_data(self):
        """Load and prepare training data"""
        data_loader = load_and_prepare_data()
        merged_df = data_loader.prepare_data()
        
        # Build mappings from the merged dataframe to ensure consistency
        user_ids = merged_df['user_id'].unique()
        item_ids = merged_df['parent_asin'].unique()
        
        self.user2idx = {u: i for i, u in enumerate(user_ids)}
        self.item2idx = {a: i for i, a in enumerate(item_ids)}
        self.idx2item = {i: a for a, i in self.item2idx.items()}
        
        # Re-map the indices in the dataframes using our mappings
        merged_df = merged_df.copy()
        merged_df['user_idx'] = merged_df['user_id'].map(self.user2idx)
        merged_df['item_idx'] = merged_df['parent_asin'].map(self.item2idx)
        
        # Split into train/val/test after re-mapping
        self.train_df, self.val_df, self.test_df = data_loader.train_test_split(merged_df)
        
        # Category mapping
        cats = self.train_df['main_cat_idx'].unique()
        self.cat2idx = {int(c): int(c) for c in cats}
        
        # Dimensions
        self.n_users = len(user_ids)
        self.n_items = len(item_ids)
        self.n_main_cats = int(self.train_df['main_cat_idx'].max()) + 1
        
        print(f"Loaded data: {len(self.train_df)} train, {len(self.val_df)} val, {len(self.test_df)} test")
        print(f"Users: {self.n_users}, Items: {self.n_items}, Categories: {self.n_main_cats}")
        
    def prepare_title_embeddings(self):
        """Load or compute title embeddings"""
        repo_root = Path(__file__).resolve().parents[2]
        emb_path = repo_root / "data" / "embeddings" / "title_embeddings.npy"
        
        if emb_path.exists():
            self.title_embeddings = np.load(emb_path)
            print("Loaded existing title embeddings")
        else:
            print("Computing title embeddings...")
            # Load metadata to get titles
            _, metadata = load_and_prepare_data().load_data()
            titles_df = metadata[['parent_asin', 'title']].copy()
            titles_df['title'] = titles_df['title'].fillna('Unknown Product').astype(str)
            
            # Create title lookup
            title_lookup = dict(zip(titles_df['parent_asin'], titles_df['title']))
            item_titles = []
            for i in range(self.n_items):
                asin = self.idx2item.get(i, 'unknown')
                title = title_lookup.get(asin, 'Unknown Product')
                item_titles.append(title)
            
            # Encode with SBERT
            model = SentenceTransformer('all-MiniLM-L6-v2')
            self.title_embeddings = model.encode(item_titles, convert_to_numpy=True)
            
            # Save for future use
            os.makedirs(emb_path.parent, exist_ok=True)
            np.save(emb_path, self.title_embeddings)
            print("Saved title embeddings")
        
        return torch.from_numpy(self.title_embeddings).float()
        
    def prepare_training_data(self):
        """Prepare tensors for training"""
        # User-positive item arrays
        u_arr = torch.tensor(self.train_df['user_idx'].values, dtype=torch.long)
        pos_item_arr = torch.tensor(self.train_df['item_idx'].values, dtype=torch.long)
        pos_cat_arr = torch.tensor(self.train_df['main_cat_idx'].values, dtype=torch.long)
        pos_price_arr = torch.tensor(self.train_df['price_log_scaled'].fillna(0).values, dtype=torch.float32)
        pos_avg_arr = torch.tensor(self.train_df['average_rating_scaled'].fillna(0).values, dtype=torch.float32)
        pos_count_arr = torch.tensor(self.train_df['rating_number_log_scaled'].fillna(0).values, dtype=torch.float32)
        
        # Item-level arrays for inference
        item_cat = np.zeros(self.n_items, dtype=np.int64)
        item_price = np.zeros(self.n_items, dtype=np.float32)
        item_avg = np.zeros(self.n_items, dtype=np.float32)
        item_count = np.zeros(self.n_items, dtype=np.float32)
        
        # Fill item arrays - only for items that exist in our mapping
        meta_source = self.train_df[['item_idx', 'main_cat_idx', 'price_log_scaled', 
                                   'average_rating_scaled', 'rating_number_log_scaled']].drop_duplicates('item_idx')
        for _, row in meta_source.iterrows():
            i = int(row.item_idx)
            # Skip items that are out of bounds 
            if i < self.n_items:
                item_cat[i] = int(row.main_cat_idx)
                item_price[i] = float(row.price_log_scaled) if pd.notna(row.price_log_scaled) else 0.0
                item_avg[i] = float(row.average_rating_scaled) if pd.notna(row.average_rating_scaled) else 0.0
                item_count[i] = float(row.rating_number_log_scaled) if pd.notna(row.rating_number_log_scaled) else 0.0
        
        # User -> positives mapping for negative sampling
        user_pos = {}
        for u, i in zip(u_arr.numpy(), pos_item_arr.numpy()):
            user_pos.setdefault(int(u), set()).add(int(i))
        
        return {
            'u_arr': u_arr,
            'pos_item_arr': pos_item_arr, 
            'pos_cat_arr': pos_cat_arr,
            'pos_price_arr': pos_price_arr,
            'pos_avg_arr': pos_avg_arr,
            'pos_count_arr': pos_count_arr,
            'item_cat': item_cat,
            'item_price': item_price,
            'item_avg': item_avg,
            'item_count': item_count,
            'user_pos': user_pos
        }
    
    def sample_negatives(self, user_batch, n_negatives, user_pos_sets):
        """Sample negative items for each user in batch"""
        B = user_batch.size(0)
        neg_items = torch.randint(0, self.n_items, (B, n_negatives))
        
        for idx, u in enumerate(user_batch.tolist()):
            positives = user_pos_sets.get(u, set())
            for k in range(n_negatives):
                tries = 0
                while neg_items[idx, k].item() in positives and tries < 10:
                    neg_items[idx, k] = torch.randint(0, self.n_items, (1,))
                    tries += 1
        return neg_items
    
    def bpr_loss(self, pos_scores, neg_scores):
        """BPR loss with softplus"""
        return nn.functional.softplus(neg_scores - pos_scores).mean()
    
    def evaluate_validation_loss(self, model, title_embeddings_tensor, train_data, n_samples=1000):
        """Evaluate BPR loss on validation set"""
        model.eval()
        
        if len(self.val_df) == 0:
            return float('inf')
        
        # Sample validation interactions if too many
        val_sample = self.val_df.sample(n=min(n_samples, len(self.val_df)), random_state=42)
        
        val_losses = []
        with torch.no_grad():
            for _, row in val_sample.iterrows():
                u = int(row['user_idx'])
                pos_item = int(row['item_idx'])
                
                if u >= self.n_users or pos_item >= self.n_items:
                    continue
                
                # Sample negative items
                neg_items = []
                attempts = 0
                while len(neg_items) < 5 and attempts < 50:  # 5 negatives
                    neg_item = np.random.randint(0, self.n_items)
                    if neg_item != pos_item:
                        neg_items.append(neg_item)
                    attempts += 1
                
                if len(neg_items) == 0:
                    continue
                
                # Prepare tensors
                user_tensor = torch.tensor([u] * (1 + len(neg_items)), dtype=torch.long)
                item_tensor = torch.tensor([pos_item] + neg_items, dtype=torch.long)
                
                # Get metadata for items
                cat_tensor = torch.tensor([train_data['item_cat'][i] for i in [pos_item] + neg_items], dtype=torch.long)
                price_tensor = torch.tensor([train_data['item_price'][i] for i in [pos_item] + neg_items], dtype=torch.float32)
                avg_tensor = torch.tensor([train_data['item_avg'][i] for i in [pos_item] + neg_items], dtype=torch.float32)
                count_tensor = torch.tensor([train_data['item_count'][i] for i in [pos_item] + neg_items], dtype=torch.float32)
                
                # Score all items
                scores = model.score(
                    user_tensor, item_tensor, cat_tensor, price_tensor, avg_tensor, count_tensor,
                    title_embeddings_tensor[item_tensor]
                )
                
                # Calculate BPR loss between positive and each negative
                pos_score = scores[0]
                for neg_score in scores[1:]:
                    loss = self.bpr_loss(pos_score.unsqueeze(0), neg_score.unsqueeze(0))
                    val_losses.append(loss.item())
        
        return np.mean(val_losses) if val_losses else float('inf')
    
    def train_model(self, max_epochs=50, batch_size=1024, lr=5e-4, l2=1e-6, 
                   n_negatives=5, patience=3):
        """Train MLP model with BPR loss and early stopping"""
        
        # Prepare data
        train_data = self.prepare_training_data()
        title_embeddings_tensor = self.prepare_title_embeddings()
        
        # Model parameters (matching notebook hyperparameters)
        model_params = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_main_cats': self.n_main_cats,
            'text_emb_dim': 384,  # SBERT dimension
            'emb_dim': 64,
            'text_proj_dim': 64,
            'mlp_dim': 64,
            'hidden_dim': 128,
            'use_avg_rating': True,
            'use_rating_count': True,
            'use_price': True,
            'use_text': True,
            'dropout': 0.1
        }
        
        model = MLPModel(**model_params)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)
        
        n_train = len(train_data['u_arr'])
        indices = np.arange(n_train)
        
        # Training loop with validation-based early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        print("Starting training...")
        epoch_pbar = tqdm(range(1, max_epochs + 1), desc="Training Epochs", unit="epoch")
        for epoch in epoch_pbar:
            model.train()
            np.random.shuffle(indices)
            
            epoch_losses = []
            batch_pbar = tqdm(range(0, n_train, batch_size), desc=f"Epoch {epoch}", unit="batch", leave=False)
            for start in batch_pbar:
                batch_idx = indices[start:start+batch_size]
                if len(batch_idx) == 0:
                    continue
                    
                ub = train_data['u_arr'][batch_idx]
                pb = train_data['pos_item_arr'][batch_idx]
                cb = train_data['pos_cat_arr'][batch_idx]
                price_b = train_data['pos_price_arr'][batch_idx]
                avg_b = train_data['pos_avg_arr'][batch_idx]
                count_b = train_data['pos_count_arr'][batch_idx]
                
                # Sample negatives
                neg_b = self.sample_negatives(ub, n_negatives, train_data['user_pos'])
                
                optimizer.zero_grad()
                losses = []
                
                for k in range(n_negatives):
                    nb = neg_b[:, k]
                    
                    # Positive scores
                    pos_scores = model.score(
                        ub, pb, cb, price_b, avg_b, count_b,
                        title_embeddings_tensor[pb]
                    )
                    
                    # Negative scores
                    neg_scores = model.score(
                        ub, nb, 
                        torch.tensor(train_data['item_cat'][nb], dtype=torch.long),
                        torch.tensor(train_data['item_price'][nb], dtype=torch.float32),
                        torch.tensor(train_data['item_avg'][nb], dtype=torch.float32), 
                        torch.tensor(train_data['item_count'][nb], dtype=torch.float32),
                        title_embeddings_tensor[nb]
                    )
                    
                    losses.append(self.bpr_loss(pos_scores, neg_scores))
                
                loss = torch.stack(losses).mean()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                # Update batch progress bar with current loss
                batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            batch_pbar.close()
            avg_train_loss = np.mean(epoch_losses)
            
            # Evaluate on validation set
            val_loss = self.evaluate_validation_loss(model, title_embeddings_tensor, train_data)
            
            # Early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
            
            # Update epoch progress bar with losses
            epoch_pbar.set_postfix({
                "train_loss": f"{avg_train_loss:.4f}", 
                "val_loss": f"{val_loss:.4f}",
                "best_val": f"{best_val_loss:.4f}", 
                "patience": f"{patience_counter}/{patience}"
            })
                
            if patience_counter >= patience:
                epoch_pbar.write(f"Early stopping at epoch {epoch} (val_loss: {val_loss:.4f})")
                break
        
        epoch_pbar.close()
        
        # Load best validation state
        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            print("Warning: No improvement found, using final model state")
        
        # Save model with assets
        self.save_model(model, model_params, train_data, title_embeddings_tensor)
        
        return model, train_data, title_embeddings_tensor
        
    def save_model(self, model, model_params, train_data, title_embeddings_tensor):
        """Save model with complete inference assets"""
        saved_dir = Path("saved_files")
        saved_dir.mkdir(exist_ok=True)
        
        repo_root = Path(__file__).resolve().parents[2]
        
        # Prepare complete item metadata for inference (no need to load full dataset)
        item_metadata = {}
        
        # Get preprocessing stats from training data for standardization
        price_mean = self.train_df['price_log_scaled'].mean()
        price_std = self.train_df['price_log_scaled'].std()
        avg_rating_mean = self.train_df['average_rating_scaled'].mean()
        avg_rating_std = self.train_df['average_rating_scaled'].std()
        rating_count_mean = self.train_df['rating_number_log_scaled'].mean()
        rating_count_std = self.train_df['rating_number_log_scaled'].std()
        
        # Build complete item metadata from training data  
        # Get original metadata since merged_df only has scaled features
        from prepare_data import load_and_prepare_data
        data_loader = load_and_prepare_data()
        _, original_metadata = data_loader.load_data()
        
        # Create lookup for original metadata values
        metadata_lookup = {}
        for _, row in original_metadata.iterrows():
            metadata_lookup[row['parent_asin']] = {
                'price': float(row['price']) if pd.notna(row['price']) else 0.0,
                'average_rating': float(row['average_rating']) if pd.notna(row['average_rating']) else 0.0,
                'rating_number': int(row['rating_number']) if pd.notna(row['rating_number']) else 0,
                'title': str(row['title']) if pd.notna(row['title']) else 'Unknown'
            }
        
        # Get scaled features from training data
        meta_source = self.train_df[['parent_asin', 'item_idx', 'main_cat_idx', 'price_log_scaled', 
                                   'average_rating_scaled', 'rating_number_log_scaled']].drop_duplicates('parent_asin')
        
        for _, row in meta_source.iterrows():
            item_id = row['parent_asin']
            original_meta = metadata_lookup.get(item_id, {})
            
            item_metadata[item_id] = {
                'item_idx': int(row['item_idx']),
                'main_cat_idx': int(row['main_cat_idx']),
                'price_log_scaled': float(row['price_log_scaled']) if pd.notna(row['price_log_scaled']) else 0.0,
                'average_rating_scaled': float(row['average_rating_scaled']) if pd.notna(row['average_rating_scaled']) else 0.0,
                'rating_number_log_scaled': float(row['rating_number_log_scaled']) if pd.notna(row['rating_number_log_scaled']) else 0.0,
                'price': original_meta.get('price', 0.0),
                'average_rating': original_meta.get('average_rating', 0.0),
                'rating_number': original_meta.get('rating_number', 0),
                'title': original_meta.get('title', 'Unknown')
            }
        
        # Create inverse mappings
        idx2user = {v: k for k, v in self.user2idx.items()}
        
        # Preprocessing parameters for consistent feature scaling
        preprocessing = {
            'price_log_stats': {'mean': price_mean, 'std': price_std},
            'average_rating_stats': {'mean': avg_rating_mean, 'std': avg_rating_std}, 
            'rating_count_log_stats': {'mean': rating_count_mean, 'std': rating_count_std},
            'cat2idx': self.cat2idx
        }
        
        assets = {
            'user2idx': self.user2idx,
            'item2idx': self.item2idx,
            'idx2user': idx2user,
            'idx2item': self.idx2item,
            'item_metadata': item_metadata,
            'preprocessing': preprocessing,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_main_cats': self.n_main_cats,
            'title_embeddings_path': str(repo_root / 'data' / 'embeddings' / 'title_embeddings.npy')
        }
        
        payload = {
            'model_state_dict': model.state_dict(),
            'model_config': {'params': model_params},  # Use new format
            'assets': assets
        }
        
        model_path = saved_dir / "mlp_model.pth"
        torch.save(payload, model_path)
        print(f"Model saved to {model_path} with complete inference assets")
        print(f"Saved {len(item_metadata)} item metadata entries")
    
    def evaluate_model(self, model, train_data, title_embeddings_tensor, k=10, n_test_items=200):
        """Evaluate model on test set with sampled catalog"""
        model.eval()
        
        # Get test users who also appear in train
        train_users = set(self.train_df['user_idx'].unique())
        test_subset = self.test_df[self.test_df['user_idx'].isin(train_users)].copy()
        
        # Build user -> train items mapping
        user_train_items = {}
        for _, row in self.train_df.iterrows():
            user_train_items.setdefault(int(row['user_idx']), set()).add(int(row['item_idx']))
        
        hits, recalls, ndcgs = [], [], []
        
        with torch.no_grad():
            for _, row in test_subset.iterrows():
                u = int(row['user_idx'])
                true_item = int(row['item_idx'])
                
                if u >= self.n_users or true_item >= self.n_items:
                    continue
                
                # Sample candidate items (including true item)
                train_items = user_train_items.get(u, set())
                
                # Sample negative candidates
                candidates = [true_item]
                while len(candidates) < min(n_test_items, self.n_items):
                    candidate = np.random.randint(0, self.n_items)
                    if candidate != true_item and candidate not in train_items:
                        candidates.append(candidate)
                
                # Score all candidates
                candidate_tensor = torch.tensor(candidates, dtype=torch.long)
                user_tensor = torch.full((len(candidates),), u, dtype=torch.long)
                
                scores = model.score(
                    user_tensor, candidate_tensor,
                    torch.tensor(train_data['item_cat'][candidate_tensor], dtype=torch.long),
                    torch.tensor(train_data['item_price'][candidate_tensor], dtype=torch.float32),
                    torch.tensor(train_data['item_avg'][candidate_tensor], dtype=torch.float32),
                    torch.tensor(train_data['item_count'][candidate_tensor], dtype=torch.float32),
                    title_embeddings_tensor[candidate_tensor]
                )
                
                # Get top-k
                _, top_indices = torch.topk(scores, k=min(k, len(candidates)), largest=True)
                top_items = [candidates[idx.item()] for idx in top_indices]
                
                # Metrics
                hit = 1.0 if true_item in top_items else 0.0
                recall = hit  # Single item case
                ndcg = 1.0 / math.log2(top_items.index(true_item) + 2) if true_item in top_items else 0.0
                
                hits.append(hit)
                recalls.append(recall)
                ndcgs.append(ndcg)
        
        # Results
        hit_rate = np.mean(hits) if hits else 0.0
        recall_rate = np.mean(recalls) if recalls else 0.0
        ndcg_score = np.mean(ndcgs) if ndcgs else 0.0
        
        print(f"Evaluation Results (K={k}, test_items={n_test_items}):")
        print(f"HR@{k}: {hit_rate:.4f}")
        print(f"Recall@{k}: {recall_rate:.4f}")
        print(f"NDCG@{k}: {ndcg_score:.4f}")
        print(f"Evaluated on {len(hits)} test cases")
        
        return {
            'hr': hit_rate,
            'recall': recall_rate,
            'ndcg': ndcg_score,
            'n_test': len(hits)
        }


def main():
    """Main training script"""
    trainer = RecommenderTrainer()
    
    # Load data
    trainer.load_data()
    
    # Train model
    model, train_data, title_embeddings = trainer.train_model()
    
    # Evaluate
    results = trainer.evaluate_model(model, train_data, title_embeddings)
    
    print("Training completed successfully!")
    return results


if __name__ == "__main__":
    main()