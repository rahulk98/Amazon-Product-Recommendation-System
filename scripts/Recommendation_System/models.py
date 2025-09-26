import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCollaborativeFiltering(nn.Module):
    """Basic CF model with only user-item embeddings for baseline comparison"""
    
    def __init__(self, n_users, n_items, emb_dim=64, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        
        # Only user and item embeddings
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        
        # Biases
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        std = 0.01
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=std)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=std)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def score(self, user_idx, item_idx):
        """Returns batch of scores for user-item pairs"""
        u = self.user_emb(user_idx)  # (B, D)
        i = self.item_emb(item_idx)  # (B, D)
        
        if self.dropout is not None:
            u = self.dropout(u)
            i = self.dropout(i)
        
        dot = (u * i).sum(dim=-1)  # (B,)
        b_u = self.user_bias(user_idx).squeeze(-1)  # (B,)
        b_i = self.item_bias(item_idx).squeeze(-1)  # (B,)
        
        return dot + b_u + b_i
    
    def forward(self, user_idx, item_idx):
        return self.score(user_idx, item_idx)


class AdditiveHybridMFWithText(nn.Module):


    def __init__(self,
                 n_users,
                 n_items,
                 n_main_cats,
                 text_emb_dim,
                 emb_dim=64,
                 text_proj_dim=None,  # New parameter for text projection dimension
                 use_avg_rating=True,
                 use_rating_count=True,
                 use_price=True,
                 use_text=True,
                 dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_text = use_text
        
        # If text_proj_dim not specified, use emb_dim (backward compatibility)
        self.text_proj_dim = text_proj_dim if text_proj_dim is not None else emb_dim

        # main embeddings
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.cat_emb  = nn.Embedding(n_main_cats, emb_dim)   # 0 reserved for <unk>

        # numeric feature projections (scalar -> embedding)
        self.use_price = use_price
        self.use_avg_rating = use_avg_rating
        self.use_rating_count = use_rating_count

        if self.use_price:
            self.price_proj = nn.Linear(1, emb_dim, bias=True)
        else:
            self.price_proj = None

        if self.use_avg_rating:
            self.avg_proj = nn.Linear(1, emb_dim, bias=True)
        else:
            self.avg_proj = None

        if self.use_rating_count:
            self.count_proj = nn.Linear(1, emb_dim, bias=True)
        else:
            self.count_proj = None

        # text projection (SBERT dim -> text_proj_dim -> emb_dim for additive combination)
        if self.use_text:
            if self.text_proj_dim == emb_dim:
                # Direct projection for efficiency when dimensions match
                self.text_proj = nn.Linear(text_emb_dim, emb_dim, bias=True)
            else:
                # Two-stage projection for regularization when dimensions differ
                self.text_proj = nn.Sequential(
                    nn.Linear(text_emb_dim, self.text_proj_dim, bias=True),
                    nn.ReLU(),
                    nn.Linear(self.text_proj_dim, emb_dim, bias=True)
                )
        else:
            self.text_proj = None

        # biases
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        # optional dropout on item vector
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # initialization (small normal)
        self._init_weights()

    def _init_weights(self):
        std = 0.01
        for emb in (self.user_emb, self.item_emb, self.cat_emb):
            nn.init.normal_(emb.weight, mean=0.0, std=std)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        # Linear proj init
        for proj in [self.price_proj, self.avg_proj, self.count_proj]:
            if proj is not None:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
        
        # Initialize text projection (handle both single layer and sequential)
        if self.text_proj is not None:
            if isinstance(self.text_proj, nn.Sequential):
                for layer in self.text_proj:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
            else:
                nn.init.xavier_uniform_(self.text_proj.weight)
                nn.init.zeros_(self.text_proj.bias)

    def forward_item_vector(self, item_idx, main_cat_idx, price_val=None,
                            avg_rating_val=None, rating_count_val=None, text_emb=None):

        v_item = self.item_emb(item_idx)           # (B, D)
        v_cat  = self.cat_emb(main_cat_idx)        # (B, D)
        parts = [v_item, v_cat]

        if self.price_proj is not None and price_val is not None:
            # ensure shape (B,1)
            pv = price_val.view(-1, 1).float()
            v_price = self.price_proj(pv)         # (B, D)
            parts.append(v_price)

        if self.avg_proj is not None and avg_rating_val is not None:
            av = avg_rating_val.view(-1, 1).float()
            v_avg = self.avg_proj(av)
            parts.append(v_avg)

        if self.count_proj is not None and rating_count_val is not None:
            cv = rating_count_val.view(-1, 1).float()
            v_count = self.count_proj(cv)
            parts.append(v_count)

        if self.text_proj is not None and text_emb is not None:
            v_text = self.text_proj(text_emb)     # (B, D) - now supports text_proj_dim
            parts.append(v_text)

        item_vec = sum(parts)  # additive combination

        if self.dropout is not None:
            item_vec = self.dropout(item_vec)

        return item_vec

    def score(self, user_idx, item_idx, main_cat_idx, price_val=None,
              avg_rating_val=None, rating_count_val=None, text_emb=None):
        """
        Returns (batch,) scores for the provided inputs.
        """
        u = self.user_emb(user_idx)                 # (B, D)
        item_vec = self.forward_item_vector(item_idx, main_cat_idx,
                                            price_val, avg_rating_val, rating_count_val, text_emb)  # (B, D)
        dot = (u * item_vec).sum(dim=-1)            # (B,)
        b_u = self.user_bias(user_idx).squeeze(-1)  # (B,)
        b_i = self.item_bias(item_idx).squeeze(-1)  # (B,)
        return dot + b_u + b_i

    def forward(self, user_idx, item_idx, main_cat_idx, price_val=None,
                avg_rating_val=None, rating_count_val=None, text_emb=None):
        """
        Standard forward that returns scores. Kept for compatibility.
        """
        return self.score(user_idx, item_idx, main_cat_idx, price_val, avg_rating_val, rating_count_val, text_emb)


class MLPModel(nn.Module):

    
    def __init__(self, 
                 n_users, 
                 n_items, 
                 n_main_cats,
                 text_emb_dim,
                 emb_dim=64,
                 text_proj_dim=None,  # New parameter for text projection dimension
                 mlp_dim=128,  # Output dimension for both towers
                 hidden_dim=256,
                 use_avg_rating=True,
                 use_rating_count=True,
                 use_price=True,
                 use_text=True,
                 dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.mlp_dim = mlp_dim
        self.use_text = use_text
        self.use_price = use_price
        self.use_avg_rating = use_avg_rating
        self.use_rating_count = use_rating_count
        
        # If text_proj_dim not specified, use emb_dim (backward compatibility)
        self.text_proj_dim = text_proj_dim if text_proj_dim is not None else emb_dim
        
        # Base embeddings
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.cat_emb = nn.Embedding(n_main_cats, emb_dim)
        
        # Feature projections
        self.price_proj = nn.Linear(1, emb_dim, bias=True) if use_price else None
        self.avg_proj = nn.Linear(1, emb_dim, bias=True) if use_avg_rating else None
        self.count_proj = nn.Linear(1, emb_dim, bias=True) if use_rating_count else None
        
        # Text projection with configurable intermediate dimension
        if self.use_text:
            if self.text_proj_dim == emb_dim:
                # Direct projection for efficiency when dimensions match
                self.text_proj = nn.Linear(text_emb_dim, emb_dim, bias=True)
            else:
                # Two-stage projection for regularization when dimensions differ
                self.text_proj = nn.Sequential(
                    nn.Linear(text_emb_dim, self.text_proj_dim, bias=True),
                    nn.ReLU(),
                    nn.Linear(self.text_proj_dim, emb_dim, bias=True)
                )
        else:
            self.text_proj = None
        
        # Calculate item input dimension
        item_input_dim = emb_dim * 2  # item_emb + cat_emb
        if use_price:
            item_input_dim += emb_dim
        if use_avg_rating:
            item_input_dim += emb_dim
        if use_rating_count:
            item_input_dim += emb_dim
        if use_text:
            item_input_dim += emb_dim
            
        # User tower MLP
        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mlp_dim)
        )
        
        # Item tower MLP  
        self.item_mlp = nn.Sequential(
            nn.Linear(item_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mlp_dim)
        )
        
        # Biases
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        
        self._init_weights()
        
    def _init_weights(self):
        # Initialize embeddings
        std = 0.01
        for emb in [self.user_emb, self.item_emb, self.cat_emb]:
            nn.init.normal_(emb.weight, mean=0.0, std=std)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
        # Initialize feature projection layers
        for proj in [self.price_proj, self.avg_proj, self.count_proj]:
            if proj is not None:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
        
        # Initialize text projection (handle both single layer and sequential)
        if self.text_proj is not None:
            if isinstance(self.text_proj, nn.Sequential):
                for layer in self.text_proj:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
            else:
                nn.init.xavier_uniform_(self.text_proj.weight)
                nn.init.zeros_(self.text_proj.bias)
                
        # Initialize MLP layers
        for module in [self.user_mlp, self.item_mlp]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def compute_item_vectors(self, item_idx, main_cat_idx, price_val=None,
                           avg_rating_val=None, rating_count_val=None, text_emb=None):
        """
        Compute item tower vectors - can be precomputed offline for efficiency
        """
        # Base item features
        item_emb = self.item_emb(item_idx)  # (B, D)
        cat_emb = self.cat_emb(main_cat_idx)  # (B, D)
        features = [item_emb, cat_emb]
        
        # Add metadata features
        if self.price_proj is not None and price_val is not None:
            price_feat = self.price_proj(price_val.view(-1, 1).float())
            features.append(price_feat)
            
        if self.avg_proj is not None and avg_rating_val is not None:
            avg_feat = self.avg_proj(avg_rating_val.view(-1, 1).float())
            features.append(avg_feat)
            
        if self.count_proj is not None and rating_count_val is not None:
            count_feat = self.count_proj(rating_count_val.view(-1, 1).float())
            features.append(count_feat)
            
        if self.text_proj is not None and text_emb is not None:
            text_feat = self.text_proj(text_emb)  # Now supports text_proj_dim
            features.append(text_feat)
        
        # Concatenate all features and pass through item MLP
        item_input = torch.cat(features, dim=-1)  # (B, total_dim)
        v_item = self.item_mlp(item_input)  # (B, mlp_dim)
        
        return v_item
    
    def compute_user_vectors(self, user_idx):
        """
        Compute user tower vectors
        """
        user_emb = self.user_emb(user_idx)  # (B, D)
        v_user = self.user_mlp(user_emb)    # (B, mlp_dim)
        return v_user
    
    def score(self, user_idx, item_idx, main_cat_idx, price_val=None,
              avg_rating_val=None, rating_count_val=None, text_emb=None):
        """
        Compute scores using two-tower architecture
        """
        v_user = self.compute_user_vectors(user_idx)  # (B, mlp_dim)
        v_item = self.compute_item_vectors(item_idx, main_cat_idx, price_val, 
                                         avg_rating_val, rating_count_val, text_emb)  # (B, mlp_dim)
        
        # Dot product + biases
        dot = (v_user * v_item).sum(dim=-1)  # (B,)
        b_u = self.user_bias(user_idx).squeeze(-1)  # (B,)
        b_i = self.item_bias(item_idx).squeeze(-1)  # (B,)
        
        return dot + b_u + b_i
    
    def forward(self, user_idx, item_idx, main_cat_idx, price_val=None,
                avg_rating_val=None, rating_count_val=None, text_emb=None):
        """
        Standard forward pass
        """
        return self.score(user_idx, item_idx, main_cat_idx, price_val, 
                         avg_rating_val, rating_count_val, text_emb)


# Model factory function for easy model creation
def create_model(model_type, n_users, n_items, n_main_cats, text_emb_dim, **kwargs):

    if model_type == "CF":
        return BasicCollaborativeFiltering(
            n_users=n_users,
            n_items=n_items,
            **kwargs
        )
    elif model_type == "Text":
        return AdditiveHybridMFWithText(
            n_users=n_users,
            n_items=n_items,
            n_main_cats=n_main_cats,
            text_emb_dim=text_emb_dim,
            use_text=True,
            **kwargs
        )
    elif model_type == "Baseline":
        return AdditiveHybridMFWithText(
            n_users=n_users,
            n_items=n_items,
            n_main_cats=n_main_cats,
            text_emb_dim=text_emb_dim,
            use_text=False,  # No text for baseline
            **kwargs
        )
    elif model_type == "MLP":
        return MLPModel(
            n_users=n_users,
            n_items=n_items,
            n_main_cats=n_main_cats,
            text_emb_dim=text_emb_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
