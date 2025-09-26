import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class load_and_prepare_data:
    def __init__(self):
        pass
    def load_data(self):
        repo_root = Path(__file__).resolve().parents[2]
        review_path = repo_root / 'data' /  'Electronics.jsonl'
        meta_path = repo_root / 'data'  / 'meta_Electronics.jsonl'
        reviews = pd.read_json(review_path, lines=True)
        metadata = pd.read_json(meta_path, lines=True)
        # Adjust NaN and missing values, and remove unnecessary columns
        metadata['main_category'] = metadata['main_category'].fillna("Unknown")
        metadata.loc[(metadata['main_category'] == "nan") | (metadata['main_category'] == 'NaN'), 'main_category'] = "Unknown"
        metadata['price'] = pd.to_numeric(metadata['price'], errors='coerce')
        metadata['price'] = metadata['price'].fillna(metadata['price'].median())
        metadata['bought_together'] = metadata['bought_together'].apply(lambda x: x if isinstance(x, list) else [])
        metadata = metadata.drop(columns=['subtitle', 'author', 'store'])
        # Remove duplicates
        duplicates = reviews.duplicated(subset=['user_id', 'parent_asin'])
        # Remove duplicates - keep latest timestamp and average rating for duplicates
        if duplicates.sum() > 0:
            # Group by user and item, average the rating, keep latest timestamp
            review_data = review_data.groupby(['user_id', 'parent_asin']).agg({
            'rating': 'mean',
            'timestamp': 'max',
            'verified_purchase': 'last',
            'helpful_vote': 'last',
            'text': 'last',        # Keep text from most recent review
            'title': 'last'        # Keep title from most recent review
        }).reset_index()
        #Convert to datetime
        reviews['reviewTime'] = pd.to_datetime(reviews['timestamp'], unit='s')
        #Sort by time (useful for train/test splitting)
        review_data = reviews.sort_values(by='reviewTime')
        return reviews, metadata
    
    def prepare_data(self):
        #reviews, metadata = self.load_data()
        # Build absolute paths relative to repository root to avoid CWD issues
        repo_root = Path(__file__).resolve().parents[2]
        review_path = repo_root / 'data' / 'processed' / 'review_data.jsonl'
        meta_path = repo_root / 'data' / 'processed' / 'metadata.jsonl'

        # Read JSONL files
        reviews = pd.read_json(review_path, lines=True)
        metadata = pd.read_json(meta_path, lines=True)
        ratings_df = reviews[['user_id', 'parent_asin', 'rating', 'reviewTime']].copy()
        metadata_df = metadata[['parent_asin', 'main_category', 'average_rating', 'rating_number', 'price', 'title']].copy()
        
        # Step 1 - Data Preparation

        # Map users/items to integer IDs.
        user2idx = {u: i for i, u in enumerate(ratings_df['user_id'].unique())}
        item2idx = {i: j for j, i in enumerate(ratings_df['parent_asin'].unique())}

        # Add the encoded columns
        ratings_df['user_idx'] = ratings_df['user_id'].map(user2idx)
        ratings_df['item_idx'] = ratings_df['parent_asin'].map(item2idx)
        
        ## Metadata Preparation
        metadata_df['price'] = pd.to_numeric(metadata_df['price'], errors='coerce')
        metadata_df['average_rating'] = pd.to_numeric(metadata_df['average_rating'], errors='coerce').fillna(0)
        metadata_df['rating_number'] = pd.to_numeric(metadata_df['rating_number'], errors='coerce').fillna(0).astype(int)

        # Transform numeric fields (log + scale where it makes sense)
        metadata_df['price_log'] = np.log1p(metadata_df['price'])
        metadata_df['rating_number_log'] = np.log1p(metadata_df['rating_number'])

        scalers = {}
        for col in ['price_log', 'average_rating', 'rating_number_log']:
            scaler = StandardScaler()
            metadata_df[col + '_scaled'] = scaler.fit_transform(metadata_df[[col]])
            scalers[col] = scaler

        # Map main_category to integer IDs
        main_categories = metadata_df['main_category'].fillna('Unknown').astype(str)
        cat2idx = {cat: idx+1 for idx, cat in enumerate(main_categories.unique())}
        cat2idx['<unk>'] = 0
        metadata_df['main_cat_idx'] = main_categories.map(lambda c: cat2idx.get(c, 0))

        # Merge on parent_asin
        merged_df = ratings_df.merge(
            metadata_df[['parent_asin', 'main_cat_idx', 
                        'price_log_scaled', 'average_rating_scaled', 'rating_number_log_scaled']],
            on='parent_asin',
            how='left'
        )
        return merged_df
    
    def train_test_split(self, merged_df):
        # Create train/validation/test split with last review as test, second-to-last as validation
        merged_df = merged_df.sort_values(by=['user_id', 'reviewTime'])
        
        # Get test set (last interaction per user)
        test_df = merged_df.groupby('user_id').tail(1)
        
        # Get validation set (second-to-last interaction per user, only for users with 2+ interactions)
        remaining_df = merged_df.drop(test_df.index)
        val_df = remaining_df.groupby('user_id').tail(1)
        
        # Filter validation set to only include users who have at least 2 interactions after removing test
        user_counts = remaining_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= 1].index
        val_df = val_df[val_df['user_id'].isin(valid_users)]
        
        # Get train set (everything else)
        train_df = remaining_df.drop(val_df.index)
        
        return train_df, val_df, test_df
    
    
if __name__ == "__main__":
    obj = load_and_prepare_data()
    merged_df = obj.prepare_data()
    print(f"Merged data shape: {merged_df.shape}")
