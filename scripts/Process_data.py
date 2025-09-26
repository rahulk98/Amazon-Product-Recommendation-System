import pandas as pd


def print_missing_summary(df, name):
    missing_counts = df.isna().sum()
    if len(df) == 0:
        print(f"No rows in {name} to compute missing summary.")
        return
    missing_percent = (missing_counts / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"missing": missing_counts, "percent": missing_percent})
    missing_df = missing_df[missing_df["missing"] > 0].sort_values(
        "missing", ascending=False
    )
    if not missing_df.empty:
        print(f"Missing values in {name} (count, %):")
        print(missing_df.to_string())
    else:
        print(f"No missing values in {name}.")


def preview_rows(df, columns, name, n=3):
    existing_cols = [c for c in columns if c in df.columns]
    if existing_cols:
        print(f"Sample {name} rows (first {n}):")
        print(df[existing_cols].head(n).to_string(index=False))
    else:
        print(f"No expected columns found to preview for {name}.")


# Read data in chunks to avoid MemoryError
chunk_size = 10000  # Adjust based on available memory
review_data_chunks = []
for chunk in pd.read_json("data/Original/Electronics.jsonl", lines=True, chunksize=chunk_size):
    review_data_chunks.append(chunk)
review_data = pd.concat(review_data_chunks, ignore_index=True)

metadata_chunks = []
for chunk in pd.read_json("data/Original/meta_Electronics.jsonl", lines=True, chunksize=chunk_size):
    metadata_chunks.append(chunk)
metadata = pd.concat(metadata_chunks, ignore_index=True)

print(f"Loaded review_data with shape {review_data.shape}")
print(f"Loaded metadata with shape {metadata.shape}")
print("Review data dtypes:\n" + review_data.dtypes.to_string())
print("Metadata dtypes:\n" + metadata.dtypes.to_string())
print_missing_summary(review_data, "review_data")
print_missing_summary(metadata, "metadata")
preview_rows(
    review_data,
    [
        "user_id",
        "parent_asin",
        "rating",
        "timestamp",
        "verified_purchase",
        "helpful_vote",
        "title",
    ],
    "reviews",
    n=5,
)
preview_rows(
    metadata,
    ["asin", "title", "brand", "main_category", "price", "bought_together"],
    "metadata",
    n=5,
)

metadata["main_category"] = metadata["main_category"].fillna("Unknown")
metadata.loc[
    (metadata["main_category"] == "nan") | (metadata["main_category"] == "NaN"),
    "main_category",
] = "Unknown"

metadata["price"] = pd.to_numeric(metadata["price"], errors="coerce")
metadata["price"] = metadata["price"].fillna(metadata["price"].median())
metadata["bought_together"] = metadata["bought_together"].apply(
    lambda x: x if isinstance(x, list) else []
)

metadata = metadata.drop(columns=["subtitle", "author", "store"])

print_missing_summary(review_data, "review_data (after basic cleaning)")
print_missing_summary(metadata, "metadata (after basic cleaning)")
if "price" in metadata.columns:
    print("Price summary (metadata):")
    print(metadata["price"].describe(percentiles=[0.5, 0.9, 0.95]).to_string())
if "main_category" in metadata.columns:
    top_categories = metadata["main_category"].value_counts().head(10)
    print("Top main_category values (top 10):")
    print(top_categories.to_string())
if "bought_together" in metadata.columns:
    empty_bt = metadata["bought_together"].apply(lambda x: len(x) == 0).sum()
    print(f"Rows with empty bought_together: {empty_bt} / {len(metadata)}")

# Check for duplicates
print("Before deduplication:")
print(f"Total reviews: {len(review_data)}")
duplicates = review_data.duplicated(subset=["user_id", "parent_asin"])
print(f"Duplicate (user, item) pairs: {duplicates.sum()}")

# Remove duplicates - keep latest timestamp and average rating for duplicates
if duplicates.sum() > 0:
    # Group by user and item, average the rating, keep latest timestamp
    review_data = (
        review_data.groupby(["user_id", "parent_asin"])
        .agg(
            {
                "rating": "mean",
                "timestamp": "max",
                "verified_purchase": "last",
                "helpful_vote": "last",
                "text": "last",  # Keep text from most recent review
                "title": "last",  # Keep title from most recent review
            }
        )
        .reset_index()
    )
    print(
        f"After deduplication: Total reviews: {len(review_data)} (reduced by {duplicates.sum()})"
    )
else:
    print("No duplicate (user, item) pairs found. Skipping deduplication.")

    # Step 1: Convert to datetime
review_data["reviewTime"] = pd.to_datetime(review_data["timestamp"], unit="s")

# Step 2: Sort by time (useful for train/test splitting)
review_data = review_data.sort_values(by="reviewTime")

# Step 3: Normalize timestamps
min_time = review_data["reviewTime"].min()
review_data["days_since_start"] = (review_data["reviewTime"] - min_time).dt.days
print(
    "Review time range: "
    f"{review_data['reviewTime'].min()} to {review_data['reviewTime'].max()}"
)
print(
    f"Unique users: {review_data['user_id'].nunique()} | "
    f"Unique items: {review_data['parent_asin'].nunique()}"
)

# Save the processed data
review_data.to_json(
    "../data/processed/review_data_full.jsonl", orient="records", lines=True
)
metadata.to_json("../data/processed/metadata_full.jsonl", orient="records", lines=True)

print(
    "Saved processed data to ../data/processed "
    f"(reviews={len(review_data)}, metadata={len(metadata)})"
)
