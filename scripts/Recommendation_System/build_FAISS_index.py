import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add current directory to path so we can import models
sys.path.insert(0, str(Path(__file__).parent))
from models import MLPModel


import faiss 


def main():
	here = Path(__file__).resolve().parent
	saved_dir = here / "saved_files"
	saved_dir.mkdir(parents=True, exist_ok=True)

	model_path = saved_dir / "mlp_model.pth"
	if not model_path.exists():
		raise FileNotFoundError(f"Missing checkpoint: {model_path}")

	# Load full checkpoint payload; handle different checkpoint formats
	ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
	if not isinstance(ckpt, dict):
		raise ValueError("Checkpoint must be a dictionary.")
	
	# Handle both old format (model_params) and new format (model_config)
	if "model_params" in ckpt and "model_state_dict" in ckpt:
		# Old format
		model_params = ckpt["model_params"]
		model_state_dict = ckpt["model_state_dict"]
	elif "model_config" in ckpt and "model_state_dict" in ckpt:
		# New format
		model_config = ckpt["model_config"]
		if not isinstance(model_config, dict) or "params" not in model_config:
			raise ValueError("model_config must contain 'params' key.")
		model_params = model_config["params"]
		model_state_dict = ckpt["model_state_dict"]
	else:
		raise ValueError("Checkpoint must contain either ('model_params' and 'model_state_dict') or ('model_config' and 'model_state_dict').")

	model = MLPModel(**model_params)  # type: ignore[arg-type]
	model.load_state_dict(model_state_dict)  # type: ignore[arg-type]
	model.eval()

	# Extract item embedding matrix
	if not hasattr(model, "item_emb"):
		raise AttributeError("MLPModel does not expose 'item_emb' embedding.")
	item_vecs = model.item_emb.weight.detach().cpu().numpy().astype("float32")

	# Normalize for cosine via inner product
	faiss.normalize_L2(item_vecs)

	dim = item_vecs.shape[1]
	n_items = item_vecs.shape[0]

	# Build exact IP index with ID mapping (IDs = item indices)
	index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
	ids = np.arange(n_items, dtype=np.int64)
	index.add_with_ids(item_vecs, ids)

	# Save index and minimal metadata
	index_path = saved_dir / "item_faiss_ip.index"
	faiss.write_index(index, str(index_path))

	meta = {
		"metric": "IP",
		"normalized": True,
		"n_items": int(n_items),
		"emb_dim": int(dim),
		"checkpoint": model_path.name,
	}

	# If training saved idx2item mapping in assets, persist for lookup
	assets = ckpt.get("assets", {}) if isinstance(ckpt, dict) else {}
	idx2item = assets.get("idx2item") if isinstance(assets, dict) else None
	if isinstance(idx2item, dict):
		idx2item_path = saved_dir / "idx2item.json"
		with open(idx2item_path, "w", encoding="utf-8") as f:
			json.dump(idx2item, f)
		meta["idx2item_path"] = idx2item_path.name

	with open(saved_dir / "item_faiss_ip.meta.json", "w", encoding="utf-8") as f:
		json.dump(meta, f, indent=2)

	print(f"Saved FAISS index: {index_path}")


if __name__ == "__main__":
	main()

