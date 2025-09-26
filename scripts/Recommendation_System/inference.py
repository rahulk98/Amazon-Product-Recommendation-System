import os
import json
from typing import Dict, List, Iterable, Optional, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import faiss
import pandas as pd
import warnings

# Ensure project root is on sys.path so 'scripts.Models.models' can be imported
import sys
from pathlib import Path

# Add current directory and project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from scripts.Recommendation_System.models import MLPModel
except ImportError:
    # Fallback for direct execution
    from models import MLPModel


class RecommendationInference:

    
    def __init__(self, model_path=None, faiss_index_path=None, faiss_meta_path=None):

        # Set default paths relative to this file
        script_dir = Path(__file__).parent
        self.model_path = model_path or script_dir / 'saved_files' / 'mlp_model.pth'
        self.faiss_index_path = faiss_index_path or script_dir / 'saved_files' / 'item_faiss_ip.index'
        self.faiss_meta_path = faiss_meta_path or script_dir / 'saved_files' / 'item_faiss_ip.meta.json'
        
        self.model = None
        self.faiss_index = None
        self.item_idx_to_id = None
        self.metadata_df = None
        self.user2idx = None
        self.item2idx = None
        self._load_components()
    
    def _load_components(self):
        """Load model, FAISS index, and metadata"""
        # Add current directory to sys.path for model loading
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Load model checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats (same as build_FAISS_index.py)
        if "model_params" in checkpoint and "model_state_dict" in checkpoint:
            # Old format
            model_params = checkpoint["model_params"]
            model_state_dict = checkpoint["model_state_dict"]
        elif "model_config" in checkpoint and "model_state_dict" in checkpoint:
            # New format
            model_config = checkpoint["model_config"]
            if not isinstance(model_config, dict) or "params" not in model_config:
                raise ValueError("model_config must contain 'params' key.")
            model_params = model_config["params"]
            model_state_dict = checkpoint["model_state_dict"]
        else:
            raise ValueError("Checkpoint must contain either ('model_params' and 'model_state_dict') or ('model_config' and 'model_state_dict').")
        
        # Create model with loaded config
        self.model = MLPModel(**model_params)
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(str(self.faiss_index_path))
        n_faiss_items = self.faiss_index.ntotal
        
        # Extract all data from checkpoint assets
        assets = checkpoint.get("assets", {})
        if not isinstance(assets, dict):
            raise ValueError("Checkpoint must contain 'assets' with required mappings and preprocessing data")
        
        # Extract mappings
        self.user2idx = assets.get("user2idx")
        self.item2idx = assets.get("item2idx") 
        self.idx2user = assets.get("idx2user")
        idx2item_raw = assets.get("idx2item")
        
        if not all([self.user2idx, self.item2idx, self.idx2user, idx2item_raw]):
            raise ValueError("Missing required mappings in checkpoint assets")
        
        # Fix idx2item handling - convert list to dict if needed and validate alignment
        if isinstance(idx2item_raw, list):
            self.idx2item = {i: item_id for i, item_id in enumerate(idx2item_raw)}
        elif isinstance(idx2item_raw, dict):
            self.idx2item = idx2item_raw
        else:
            raise ValueError("idx2item must be either list or dict")
        
        # Validate FAISS index alignment
        n_items = len(self.item2idx)
        if len(self.idx2item) != n_items:
            raise ValueError(f"idx2item length ({len(self.idx2item)}) != n_items ({n_items})")
        
        if n_faiss_items != n_items:
            raise ValueError(f"FAISS index size ({n_faiss_items}) != n_items ({n_items})")
        
        # Verify FAISS ids match row indices (sample check)
        for idx in [0, min(10, n_items-1), n_items-1]:
            if idx not in self.idx2item:
                raise ValueError(f"Missing idx2item mapping for index {idx}")
        
        # Extract complete item metadata (preprocessed and ready to use)
        self.item_metadata = assets.get("item_metadata")
        if not self.item_metadata:
            raise ValueError("Missing item metadata in checkpoint assets")
        
        # Extract preprocessing parameters
        self.preprocessing = assets.get("preprocessing", {})
        if not self.preprocessing:
            raise ValueError("Missing preprocessing parameters in checkpoint assets")
        
        # Load text embeddings (only external dependency)
        text_emb_path = assets.get("title_embeddings_path")
        if text_emb_path and Path(text_emb_path).exists():
            self.text_embeddings = np.load(text_emb_path)
            if self.text_embeddings.shape[0] != n_items:
                raise ValueError(f"Text embeddings size ({self.text_embeddings.shape[0]}) != n_items ({n_items})")
            print(f"Loaded text embeddings: {self.text_embeddings.shape}")
        else:
            # Try alternative paths for Docker environment
            alternative_paths = [
                Path(__file__).parent.parent.parent / "data" / "embeddings" / "title_embeddings.npy",
                Path("/app/data/embeddings/title_embeddings.npy"),
                Path("data/embeddings/title_embeddings.npy")
            ]
            
            self.text_embeddings = None
            for alt_path in alternative_paths:
                if alt_path.exists():
                    try:
                        self.text_embeddings = np.load(alt_path)
                        if self.text_embeddings.shape[0] != n_items:
                            print(f"Warning: Text embeddings size ({self.text_embeddings.shape[0]}) != n_items ({n_items})")
                            self.text_embeddings = None
                            continue
                        print(f"Loaded text embeddings from {alt_path}: {self.text_embeddings.shape}")
                        break
                    except Exception as e:
                        print(f"Failed to load text embeddings from {alt_path}: {e}")
                        continue
            
            if self.text_embeddings is None:
                print("Warning: No text embeddings found - model will run without text features")
        
        print(f"Loaded model with {len(self.user2idx)} users, {len(self.item2idx)} items, {len(self.item_metadata)} metadata entries")
    
    def _get_item_features(self, item_id):
        """Get preprocessed features for an item using stored metadata"""
        if item_id not in self.item_metadata:
            return None
        
        metadata = self.item_metadata[item_id]
        return {
            'item_idx': metadata['item_idx'],
            'main_cat_idx': metadata['main_cat_idx'],
            'price_log_scaled': metadata['price_log_scaled'],
            'average_rating_scaled': metadata['average_rating_scaled'], 
            'rating_number_log_scaled': metadata['rating_number_log_scaled'],
            'title': metadata.get('title', 'Unknown'),
            'price': metadata.get('price', 0.0),
            'average_rating': metadata.get('average_rating', 0.0),
            'rating_number': metadata.get('rating_number', 0)
        }
    
    def top_k_for_user(self, user_id, k=10, faiss_N=200, rerank=True, filter_args=None):

        with torch.no_grad():
            # Convert user_id to index
            if user_id not in self.user2idx:
                raise ValueError(f"User {user_id} not found in training data")
            
            user_idx = self.user2idx[user_id]
            
            # Step 1: Compute user embedding using MLP user tower
            user_tensor = torch.tensor([user_idx], dtype=torch.long)
            user_vec = self.model.compute_user_vectors(user_tensor)  # (1, mlp_dim)
            user_vec_numpy = F.normalize(user_vec, p=2, dim=1).cpu().numpy()  # L2 normalize for cosine similarity
            
            # Step 2: FAISS candidate retrieval
            faiss_scores, faiss_indices = self.faiss_index.search(user_vec_numpy, faiss_N)
            faiss_scores = faiss_scores[0]  # Remove batch dimension
            faiss_indices = faiss_indices[0]
            
            # Convert FAISS indices to item IDs
            candidate_items = []
            for i, (faiss_idx, faiss_score) in enumerate(zip(faiss_indices, faiss_scores)):
                if faiss_idx >= 0:  # Valid index
                    item_id = self.idx2item.get(faiss_idx)
                    if item_id and item_id in self.item_metadata:
                        candidate_items.append((item_id, faiss_score, faiss_idx))
            
            if not rerank:
                # Return top-k from FAISS results only
                return [(item_id, float(faiss_score), float(faiss_score)) 
                        for item_id, faiss_score, _ in candidate_items[:k]]
            
            # Step 3: Rerank with full MLP scoring
            if not candidate_items:
                return []
                
            # Prepare batch data for MLP scoring
            batch_user_idx = []
            batch_item_idx = []
            batch_item_ids = []
            batch_main_cat_idx = []
            batch_price = []
            batch_avg_rating = []
            batch_rating_count = []
            batch_text_embs = []
            batch_faiss_scores = []
            
            for item_id, faiss_score, _ in candidate_items:
                if item_id in self.item_metadata:
                    features = self._get_item_features(item_id)
                    if features:
                        batch_user_idx.append(user_idx)
                        batch_item_idx.append(features['item_idx'])
                        batch_item_ids.append(item_id)
                        batch_main_cat_idx.append(features['main_cat_idx'])
                        batch_price.append(features['price_log_scaled'])
                        batch_avg_rating.append(features['average_rating_scaled'])
                        batch_rating_count.append(features['rating_number_log_scaled'])
                        batch_faiss_scores.append(faiss_score)
                        
                        # Get text embedding for this item
                        if self.text_embeddings is not None:
                            batch_text_embs.append(self.text_embeddings[features['item_idx']])
            
            if not batch_user_idx:
                return []
            
            # Convert to tensors
            user_batch = torch.tensor(batch_user_idx, dtype=torch.long)
            item_batch = torch.tensor(batch_item_idx, dtype=torch.long)
            cat_batch = torch.tensor(batch_main_cat_idx, dtype=torch.long)
            price_batch = torch.tensor(batch_price, dtype=torch.float32)
            avg_rating_batch = torch.tensor(batch_avg_rating, dtype=torch.float32)
            rating_count_batch = torch.tensor(batch_rating_count, dtype=torch.float32)
            
            # Text embeddings
            text_emb_batch = None
            if batch_text_embs:
                text_emb_batch = torch.tensor(np.array(batch_text_embs), dtype=torch.float32)
            
            # Get MLP scores
            mlp_scores = self.model.score(
                user_batch, item_batch, cat_batch,
                price_val=price_batch,
                avg_rating_val=avg_rating_batch,
                rating_count_val=rating_count_batch,
                text_emb=text_emb_batch
            )
            
            # Combine results and sort by MLP score
            results = []
            for i, (item_id, faiss_score) in enumerate(zip(batch_item_ids, batch_faiss_scores)):
                mlp_score = mlp_scores[i].item()
                results.append((item_id, mlp_score, faiss_score))
            
            # Sort by MLP score (descending) and return top-k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
    
    def get_item_details(self, item_id):
        """Get details for a specific item"""
        if item_id not in self.item_metadata:
            return None
            
        metadata = self.item_metadata[item_id]
        return {
            'item_id': item_id,
            'title': metadata.get('title', 'Unknown'),
            'main_category': 'Unknown',  # Would need category reverse mapping
            'price': metadata.get('price', 0.0),
            'average_rating': metadata.get('average_rating', 0.0),
            'rating_number': metadata.get('rating_number', 0)
        }
    
    def get_recommendations_with_details(self, user_id, k=10, faiss_N=200, rerank=True):
        recommendations = self.top_k_for_user(user_id, k, faiss_N, rerank)
        
        detailed_recommendations = []
        for item_id, mlp_score, faiss_score in recommendations:
            details = self.get_item_details(item_id)
            if details:
                details.update({
                    'mlp_score': float(mlp_score),
                    'faiss_score': float(faiss_score)
                })
                detailed_recommendations.append(details)
        
        return detailed_recommendations


def demo():
    recommender = RecommendationInference()
    
    # Get test users directly from model's user mappings (no dataset dependency!)
    available_users = list(recommender.user2idx.keys())
    import random
    random.seed(42)  # For reproducible demo
    test_users = random.sample(available_users, min(3, len(available_users)))
    
    print(f"\nTesting with {len(test_users)} users (randomly sampled from {len(available_users)} available users)...")
    
    for user_id in test_users:
        print(f"\n{'='*60}")
        print(f"Recommendations for user: {user_id}")
        print(f"{'='*60}")
        
        # Get recommendations with details
        recommendations = recommender.get_recommendations_with_details(
            user_id, k=3, faiss_N=50, rerank=True
        )
        
        for i, item in enumerate(recommendations, 1):
            print(f"\n{i}. {item['title'][:60]}...")
            print(f"   Category: {item['main_category']}")
            print(f"   Price: ${item['price']:.2f}")
            print(f"   Rating: {item['average_rating']:.1f} ({item['rating_number']} reviews)")
            print(f"   MLP Score: {item['mlp_score']:.4f}, FAISS Score: {item['faiss_score']:.4f}")


if __name__ == "__main__":
    demo()

