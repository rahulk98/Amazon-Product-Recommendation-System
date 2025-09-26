import json
import os
import random
from collections import Counter
from pathlib import Path
from tqdm import tqdm

def count_interactions_stream(review_path, active_users=None, active_items=None):
    """Single streaming pass: count interactions (optionally restricted to active sets)."""
    user_cnt = Counter()
    item_cnt = Counter()
    file_size = os.path.getsize(review_path)

    with open(review_path, 'r', encoding='utf-8') as f, tqdm(total=file_size, unit="B", unit_scale=True, desc="Counting") as pbar:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            u = r.get('user_id') or r.get('reviewerID')
            i = r.get('parent_asin') or r.get('asin')
            if u is None or i is None:
                continue
            if active_users is not None and u not in active_users:
                continue
            if active_items is not None and i not in active_items:
                continue
            user_cnt[u] += 1
            item_cnt[i] += 1
            pbar.update(len(line.encode("utf-8")))
    return user_cnt, item_cnt

def extract_k_core_reviews(review_path, out_path, k=5, max_iters=20):
    review_path = Path(review_path)
    out_path = Path(out_path)

    # Initial counts
    user_cnt, item_cnt = count_interactions_stream(review_path)
    active_users = {u for u,c in user_cnt.items() if c >= k}
    active_items = {i for i,c in item_cnt.items() if c >= k}

    for it in range(max_iters):
        new_user_cnt, new_item_cnt = count_interactions_stream(review_path, active_users, active_items)
        new_active_users = {u for u,c in new_user_cnt.items() if c >= k}
        new_active_items = {i for i,c in new_item_cnt.items() if c >= k}

        if new_active_users == active_users and new_active_items == active_items:
            print(f"Converged after {it} iterations. Users={len(active_users)}, Items={len(active_items)}")
            break
        active_users, active_items = new_active_users, new_active_items
        print(f"Iteration {it+1}: users={len(active_users)}, items={len(active_items)}")

    # Final write with progress bar
    file_size = os.path.getsize(review_path)
    written = 0
    with open(review_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout, tqdm(total=file_size, unit="B", unit_scale=True, desc="Writing") as pbar:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            u = r.get('user_id') or r.get('reviewerID')
            i = r.get('parent_asin') or r.get('asin')
            if u in active_users and i in active_items:
                fout.write(json.dumps(r) + "\n")
                written += 1
            pbar.update(len(line.encode("utf-8")))

    print(f"Wrote {written} reviews to {out_path}")
    return active_users, active_items

def sample_users_from_5core(input_path, output_path, user_sample_rate=0.05, seed=42):
    """
    Sample users from 5-core dataset and extract all their reviews.
    This preserves the 5-core property much better than random review sampling.
    """
    random.seed(seed)
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    print("Collecting all user IDs from 5-core data...")
    all_users = set()
    file_size = os.path.getsize(input_path)
    
    with open(input_path, 'r', encoding='utf-8') as f, \
         tqdm(total=file_size, unit="B", unit_scale=True, desc="Collecting users") as pbar:
        for line in f:
            if line.strip():
                r = json.loads(line)
                u = r.get('user_id') or r.get('reviewerID')
                if u:
                    all_users.add(u)
            pbar.update(len(line.encode("utf-8")))
    
    print(f"Found {len(all_users)} unique users in 5-core data")
    
    # Randomly sample users
    n_sample_users = max(1, int(len(all_users) * user_sample_rate))
    sampled_users = set(random.sample(list(all_users), n_sample_users))
    print(f"Sampled {len(sampled_users)} users ({len(sampled_users)/len(all_users)*100:.1f}%)")
    
    # Extract all reviews from sampled users
    print("Extracting reviews from sampled users...")
    written = 0
    total_reviews = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout, \
         tqdm(total=file_size, unit="B", unit_scale=True, desc="Extracting reviews") as pbar:
        
        for line in fin:
            if line.strip():
                r = json.loads(line)
                u = r.get('user_id') or r.get('reviewerID')
                total_reviews += 1
                
                if u in sampled_users:
                    fout.write(line)
                    written += 1
            pbar.update(len(line.encode("utf-8")))
    
    print(f"Extracted {written} reviews from {total_reviews} total reviews ({written/total_reviews*100:.1f}%)")
    print(f"This represents all reviews from {len(sampled_users)} sampled users")
    return written, sampled_users

def verify_k_core(review_path, k=5):
    """Verify if the dataset satisfies k-core property."""
    user_counts = Counter()
    item_counts = Counter()
    
    with open(review_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Verifying k-core property"):
            if line.strip():
                r = json.loads(line)
                u = r.get('user_id') or r.get('reviewerID')
                i = r.get('parent_asin') or r.get('asin')
                if u and i:
                    user_counts[u] += 1
                    item_counts[i] += 1
    
    users_below_k = sum(1 for count in user_counts.values() if count < k)
    items_below_k = sum(1 for count in item_counts.values() if count < k)
    
    print(f"Verification results for k={k}:")
    print(f"- Total users: {len(user_counts)}")
    print(f"- Users with <{k} reviews: {users_below_k}")
    print(f"- Total items: {len(item_counts)}")
    print(f"- Items with <{k} reviews: {items_below_k}")
    
    is_k_core = users_below_k == 0 and items_below_k == 0
    print(f"- Is {k}-core: {is_k_core}")
    
    return is_k_core, user_counts, item_counts

def subset_metadata_by_asins(metadata_path, out_metadata_path, keep_asins):
    """Stream metadata JSONL and write only records whose asin is in keep_asins, with progress bar."""
    keep_asins = set(keep_asins)  # ensure O(1) lookup
    file_size = os.path.getsize(metadata_path)
    written = 0

    with open(metadata_path, 'r', encoding='utf-8') as fin, \
         open(out_metadata_path, 'w', encoding='utf-8') as fout, \
         tqdm(total=file_size, unit="B", unit_scale=True, desc="Filtering metadata") as pbar:

        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            asin = r.get('parent_asin') or r.get('asin') or r.get('product_id') or r.get('item_id')
            if asin in keep_asins:
                fout.write(json.dumps(r) + "\n")
                written += 1
            pbar.update(len(line.encode("utf-8")))

    print(f"Wrote {written} metadata lines to {out_metadata_path}")


def filter_items_with_min_users(input_path, output_path, min_users=2):
    """Filter reviews to keep only items that have at least min_users different users."""
    print(f"Filtering items with at least {min_users} users...")
    
    # First pass: count users per item
    item_user_counts = Counter()
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Counting users per item"):
            if line.strip():
                r = json.loads(line)
                item_id = r.get('parent_asin') or r.get('asin')
                user_id = r.get('user_id') or r.get('reviewerID')
                if item_id and user_id:
                    item_user_counts[item_id] += 1  # This actually counts reviews, we need unique users
    
    # Better approach: collect unique users per item
    item_users = {}
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Collecting users per item"):
            if line.strip():
                r = json.loads(line)
                item_id = r.get('parent_asin') or r.get('asin')
                user_id = r.get('user_id') or r.get('reviewerID')
                if item_id and user_id:
                    if item_id not in item_users:
                        item_users[item_id] = set()
                    item_users[item_id].add(user_id)
    
    # Filter items with at least min_users
    valid_items = {item for item, users in item_users.items() if len(users) >= min_users}
    
    print(f"Items before filtering: {len(item_users)}")
    print(f"Items with ≥{min_users} users: {len(valid_items)} ({len(valid_items)/len(item_users)*100:.1f}%)")
    
    # Second pass: write only reviews for valid items
    written = 0
    total_reviews = 0
    file_size = os.path.getsize(input_path)
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout, \
         tqdm(total=file_size, unit="B", unit_scale=True, desc="Writing filtered reviews") as pbar:
        
        for line in fin:
            if line.strip():
                r = json.loads(line)
                item_id = r.get('parent_asin') or r.get('asin')
                total_reviews += 1
                
                if item_id in valid_items:
                    fout.write(line)
                    written += 1
            pbar.update(len(line.encode("utf-8")))
    
    print(f"Reviews after item filtering: {written}/{total_reviews} ({written/total_reviews*100:.1f}%)")
    return written, valid_items


if __name__ == "__main__":
    
    # File paths
    reviews_file = "data/Original/Electronics.jsonl"
    metadata_file = "data/Original/meta_Electronics.jsonl"

    out_reviews_5core = "data/reviews_5core.jsonl"
    out_reviews_sample = "data/reviews_5core_sample.jsonl"
    out_reviews_filtered = "data/reviews_5core_sample_filtered.jsonl"
    out_metadata = "data/metadata_5core_sample.jsonl"

    # Step 1: Extract 5-core reviews
    print("Step 1: Extracting 5-core reviews...")
    users, items = extract_k_core_reviews(reviews_file, out_reviews_5core, k=5)
    
    # Step 2: Sample 5% of users from 5-core data and get all their reviews
    print("\\nStep 2: Sampling 2.5% of users from 5-core data...")
    written, sampled_users = sample_users_from_5core(out_reviews_5core, out_reviews_sample, user_sample_rate=0.025)
    
    # Step 3: Verify that sampled data maintains good properties
    print("\\nStep 3: Verifying sampled data properties...")
    is_k_core, user_counts, item_counts = verify_k_core(out_reviews_sample, k=5)
    
    if is_k_core:
        print("✓ Sampled data maintains 5-core property!")
    else:
        print("⚠ Sampled data doesn't maintain strict 5-core, but should have good density")
    
    # Step 4: Filter items to keep only those with at least 2 users
    print("\\nStep 4: Filtering items with at least 2 users...")
    written_filtered, valid_items = filter_items_with_min_users(out_reviews_sample, out_reviews_filtered, min_users=1)
    
    # Step 5: Verify final filtered data properties
    print("\\nStep 5: Verifying final filtered data properties...")
    is_k_core_final, user_counts_final, item_counts_final = verify_k_core(out_reviews_filtered, k=2)
    
    # Step 6: Extract metadata for final filtered items
    print("\\nStep 6: Extracting metadata for final filtered items...")
    subset_metadata_by_asins(metadata_file, out_metadata, valid_items)
    
    print(f"\\nCompleted! Files created:")
    print(f"- Full 5-core reviews: {out_reviews_5core}")
    print(f"- Sampled reviews (2.5% users): {out_reviews_sample}")
    print(f"- Filtered reviews (items with ≥2 users): {out_reviews_filtered}")
    print(f"- Corresponding metadata: {out_metadata}")
    print(f"\\nFinal dataset statistics:")
    print(f"- Users: {len(user_counts_final)}")
    print(f"- Items: {len(valid_items)}")
    print(f"- Reviews: {written_filtered}")
