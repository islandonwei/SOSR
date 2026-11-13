import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict
import random
from datetime import datetime

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Paths
    queries_path = "./queries.json"
    categories_path = "./categories.csv"
    model_path = "./paraphrase-multilingual-mpnet-base-v2-local"
    output_dir = "./trained_model"
    checkpoint_dir = "./checkpoints"
    log_dir = "./logs"
    
    # Model
    embedding_dim = 768
    num_levels = 6  # Level 1-6
    
    # Training - Stage 1
    stage1_epochs = 50
    stage1_lr = 1e-5
    stage1_batch_size = 64
    
    # Training - Stage 2
    stage2_epochs = 50
    stage2_lr_encoder = 1e-5
    stage2_lr_predictor = 1e-3
    stage2_batch_size = 64
    
    # Loss
    temperature = 0.1
    lambda_weight = 2.0
    
    # Negative sampling
    num_same_level_neg = 10
    num_parent_level_neg = 5
    num_child_level_neg = 5
    
    # Training
    warmup_ratio = 0.1
    weight_decay = 1e-2
    max_grad_norm = 1.0
    
    # Data split
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# Create directories
os.makedirs(config.output_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)
os.makedirs(config.log_dir, exist_ok=True)

# ============================================================================
# Category Tree Construction
# ============================================================================
class CategoryNode:
    def __init__(self, category_id, category_name, level):
        self.category_id = category_id
        self.category_name = category_name
        self.level = level
        self.parent = None
        self.children = []
    
    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self

class CategoryTree:
    def __init__(self, categories_df):
        self.root = CategoryNode("ROOT", "ROOT", 0)
        self.nodes = {"ROOT": self.root}  # category_id -> node
        self.build_tree(categories_df)
    
    def build_tree(self, df):
        print("Building category tree...")
        
        # Sort by level to ensure parents are created before children
        df = df.sort_values('category_level')
        
        for _, row in df.iterrows():
            category_id = row['category_id']
            category_name = row['category_name']
            level = int(row['category_level'])
            category_label = row['category_label']
            
            # Parse parent from category_label
            # e.g., "Sports and Recreation > Hockey > Hockey Club"
            path_parts = [p.strip() for p in category_label.split('>')]
            
            # Create node
            node = CategoryNode(category_id, category_name, level)
            self.nodes[category_id] = node
            
            # Find parent
            if level == 1:
                parent = self.root
            else:
                # Parent is the second-to-last in the path
                parent_name = path_parts[-2] if len(path_parts) >= 2 else None
                parent = None
                
                # Find parent node by name
                for node_id, n in self.nodes.items():
                    if n.category_name == parent_name and n.level == level - 1:
                        parent = n
                        break
                
                if parent is None:
                    # Fallback: attach to root
                    parent = self.root
            
            parent.add_child(node)
        
        print(f"Tree built with {len(self.nodes)} nodes (including ROOT)")
    
    def get_siblings(self, category_id):
        """Get sibling nodes (same parent, same level)"""
        if category_id not in self.nodes:
            return []
        
        node = self.nodes[category_id]
        if node.parent is None:
            return []
        
        siblings = [child for child in node.parent.children 
                   if child.category_id != category_id]
        return siblings
    
    def get_parent_siblings(self, category_id):
        """Get parent's siblings (uncle nodes)"""
        if category_id not in self.nodes:
            return []
        
        node = self.nodes[category_id]
        if node.parent is None or node.parent.parent is None:
            return []
        
        return self.get_siblings(node.parent.category_id)
    
    def get_sibling_children(self, category_id):
        """Get children of sibling nodes (cousin nodes)"""
        if category_id not in self.nodes:
            return []
        
        siblings = self.get_siblings(category_id)
        cousins = []
        for sibling in siblings:
            cousins.extend(sibling.children)
        return cousins

# ============================================================================
# Dataset
# ============================================================================
class QueryDataset(Dataset):
    def __init__(self, data, tokenizer, category_tree, id_to_idx, is_train=True):
        self.data = data  # List of (query, category_id, level)
        self.tokenizer = tokenizer
        self.category_tree = category_tree
        self.id_to_idx = id_to_idx  # category_id -> index (0-1244)
        self.idx_to_id = {v: k for k, v in id_to_idx.items()}
        self.is_train = is_train
        
        # Get all category IDs for each level
        self.level_to_categories = defaultdict(list)
        for cat_id, node in category_tree.nodes.items():
            if cat_id != "ROOT":
                self.level_to_categories[node.level].append(cat_id)
    
    def __len__(self):
        return len(self.data)
    
    def get_negative_samples(self, category_id, level):
        """Hierarchy-aware negative sampling"""
        negatives = []
        
        # 1. Same-level negatives (siblings)
        siblings = self.category_tree.get_siblings(category_id)
        same_level_ids = [s.category_id for s in siblings]
        
        # If not enough siblings, sample from same level
        if len(same_level_ids) < config.num_same_level_neg:
            other_same_level = [c for c in self.level_to_categories[level] 
                               if c != category_id and c not in same_level_ids]
            same_level_ids.extend(random.sample(
                other_same_level, 
                min(config.num_same_level_neg - len(same_level_ids), len(other_same_level))
            ))
        else:
            same_level_ids = random.sample(same_level_ids, config.num_same_level_neg)
        
        negatives.extend([self.id_to_idx[cid] for cid in same_level_ids if cid in self.id_to_idx])
        
        # 2. Parent-level negatives (uncles)
        if level > 1:
            uncles = self.category_tree.get_parent_siblings(category_id)
            uncle_ids = [u.category_id for u in uncles]
            
            if len(uncle_ids) < config.num_parent_level_neg:
                other_parent_level = [c for c in self.level_to_categories[level - 1]
                                     if c not in uncle_ids]
                uncle_ids.extend(random.sample(
                    other_parent_level,
                    min(config.num_parent_level_neg - len(uncle_ids), len(other_parent_level))
                ))
            else:
                uncle_ids = random.sample(uncle_ids, config.num_parent_level_neg)
            
            negatives.extend([self.id_to_idx[cid] for cid in uncle_ids if cid in self.id_to_idx])
        
        # 3. Child-level negatives (cousins)
        if level < 6:
            cousins = self.category_tree.get_sibling_children(category_id)
            cousin_ids = [c.category_id for c in cousins]
            
            if len(cousin_ids) < config.num_child_level_neg:
                other_child_level = [c for c in self.level_to_categories[level + 1]]
                other_child_level = [c for c in other_child_level if c not in cousin_ids]
                cousin_ids.extend(random.sample(
                    other_child_level,
                    min(config.num_child_level_neg - len(cousin_ids), len(other_child_level))
                ))
            else:
                cousin_ids = random.sample(cousin_ids, config.num_child_level_neg)
            
            negatives.extend([self.id_to_idx[cid] for cid in cousin_ids if cid in self.id_to_idx])
        
        return negatives
    
    def __getitem__(self, idx):
        query, category_id, level = self.data[idx]
        
        # Tokenize query
        query_encoded = self.tokenizer(
            query,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        
        # Get positive category index
        positive_idx = self.id_to_idx[category_id]
        
        # Get negative samples (only for training)
        if self.is_train:
            negative_indices = self.get_negative_samples(category_id, level)
        else:
            negative_indices = []
        
        return {
            'input_ids': query_encoded['input_ids'].squeeze(0),
            'attention_mask': query_encoded['attention_mask'].squeeze(0),
            'positive_idx': positive_idx,
            'negative_indices': negative_indices,
            'level': level - 1  # Convert to 0-indexed (0-5)
        }

def collate_fn(batch):
    """Custom collate function to handle variable-length negative samples"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    positive_idx = torch.tensor([item['positive_idx'] for item in batch])
    levels = torch.tensor([item['level'] for item in batch])
    
    # Pad negative indices to same length
    max_neg_len = max(len(item['negative_indices']) for item in batch)
    negative_indices = []
    for item in batch:
        negs = item['negative_indices']
        # Pad with -1 (will be ignored in loss)
        padded = negs + [-1] * (max_neg_len - len(negs))
        negative_indices.append(padded)
    negative_indices = torch.tensor(negative_indices)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'positive_idx': positive_idx,
        'negative_indices': negative_indices,
        'level': levels
    }

# ============================================================================
# Model
# ============================================================================
class SpecificityAwareModel(nn.Module):
    def __init__(self, model_path, num_levels, num_categories):
        super().__init__()
        
        # Load pre-trained sentence transformer
        self.encoder = AutoModel.from_pretrained(model_path)
        self.embedding_dim = self.encoder.config.hidden_size
        
        # Level predictor
        self.level_predictor = nn.Linear(self.embedding_dim, num_levels)
        
        # Category embeddings (will be computed on-the-fly during training)
        self.num_categories = num_categories
    
    def encode_text(self, input_ids, attention_mask):
        """Encode text using mean pooling"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def predict_level(self, embeddings):
        """Predict level distribution"""
        logits = self.level_predictor(embeddings)
        return logits
    
    def forward(self, input_ids, attention_mask):
        embeddings = self.encode_text(input_ids, attention_mask)
        level_logits = self.predict_level(embeddings)
        return embeddings, level_logits

# ============================================================================
# Loss Functions
# ============================================================================
def compute_level_loss(level_logits, level_targets):
    """Cross-entropy loss for level prediction"""
    return F.cross_entropy(level_logits, level_targets)

def compute_contrastive_loss(query_embeddings, category_embeddings, 
                             positive_indices, negative_indices, temperature):
    """InfoNCE contrastive loss"""
    batch_size = query_embeddings.size(0)
    
    # Compute similarities
    # query_embeddings: [B, D]
    # category_embeddings: [num_categories, D]
    
    losses = []
    for i in range(batch_size):
        query_emb = query_embeddings[i]  # [D]
        pos_idx = positive_indices[i]
        neg_indices = negative_indices[i]
        
        # Get positive category embedding
        pos_emb = category_embeddings[pos_idx]  # [D]
        
        # Compute positive similarity
        pos_sim = torch.dot(query_emb, pos_emb) / temperature
        
        # Get negative category embeddings (filter out padding -1)
        valid_neg_mask = neg_indices >= 0
        if valid_neg_mask.sum() == 0:
            continue
        
        neg_indices_valid = neg_indices[valid_neg_mask]
        neg_embs = category_embeddings[neg_indices_valid]  # [num_neg, D]
        
        # Compute negative similarities
        neg_sims = torch.matmul(neg_embs, query_emb) / temperature  # [num_neg]
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
        loss = -F.log_softmax(logits, dim=0)[0]
        losses.append(loss)
    
    if len(losses) == 0:
        return torch.tensor(0.0, device=query_embeddings.device)
    
    return torch.stack(losses).mean()

# ============================================================================
# Data Loading
# ============================================================================
def load_data():
    print("Loading data...")
    
    # Load categories
    categories_df = pd.read_csv(config.categories_path)
    print(f"Loaded {len(categories_df)} categories")
    
    # Build category tree
    category_tree = CategoryTree(categories_df)
    
    # Create category_id to index mapping
    all_category_ids = categories_df['category_id'].tolist()
    id_to_idx = {cid: idx for idx, cid in enumerate(all_category_ids)}
    
    # Load queries
    with open(config.queries_path, 'r') as f:
        queries_data = json.load(f)
    
    print(f"Loaded {len(queries_data)} category entries")
    
    # Flatten to (query, category_id, level) tuples
    all_data = []
    for entry in queries_data:
        category_id = entry['id']
        level = entry['level']
        queries = entry['queries']
        
        for query in queries:
            all_data.append((query, category_id, level))
    
    print(f"Total queries: {len(all_data)}")
    
    # Split by category to avoid leakage
    category_ids = list(set([d[1] for d in all_data]))
    random.shuffle(category_ids)
    
    n_train = int(len(category_ids) * config.train_ratio)
    n_val = int(len(category_ids) * config.val_ratio)
    
    train_cat_ids = set(category_ids[:n_train])
    val_cat_ids = set(category_ids[n_train:n_train + n_val])
    test_cat_ids = set(category_ids[n_train + n_val:])
    
    train_data = [d for d in all_data if d[1] in train_cat_ids]
    val_data = [d for d in all_data if d[1] in val_cat_ids]
    test_data = [d for d in all_data if d[1] in test_cat_ids]
    
    print(f"Train: {len(train_data)} queries ({len(train_cat_ids)} categories)")
    print(f"Val: {len(val_data)} queries ({len(val_cat_ids)} categories)")
    print(f"Test: {len(test_data)} queries ({len(test_cat_ids)} categories)")
    
    # Also store category names for evaluation
    id_to_name = {row['category_id']: row['category_name'] 
                  for _, row in categories_df.iterrows()}
    
    return (train_data, val_data, test_data, 
            category_tree, id_to_idx, id_to_name, all_category_ids, categories_df)

# ============================================================================
# Evaluation
# ============================================================================
@torch.no_grad()
def evaluate(model, dataloader, category_embeddings, split_name="Val"):
    model.eval()
    
    total_level_correct = 0
    total_top1_correct = 0
    total_top5_correct = 0
    total_level_aware_correct = 0
    total_samples = 0
    
    all_level_preds = []
    all_level_targets = []
    
    for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        positive_idx = batch['positive_idx'].to(config.device)
        level_targets = batch['level'].to(config.device)
        
        # Forward pass
        query_embeddings, level_logits = model(input_ids, attention_mask)
        
        # Level prediction
        level_probs = F.softmax(level_logits, dim=1)
        level_preds = torch.argmax(level_probs, dim=1)
        
        # Category matching (soft level-aware)
        # Compute similarities with all categories
        similarities = torch.matmul(query_embeddings, category_embeddings.T)  # [B, num_cat]
        
        # Apply soft level weighting
        batch_size = similarities.size(0)
        for i in range(batch_size):
            # Get level probabilities for this query
            level_prob = level_probs[i]  # [6]
            
            # Weight similarities by level probabilities
            # We need to know which level each category belongs to
            # For efficiency, we can approximate by using the predicted level
            # But for accurate evaluation, we should use ground truth in val/test
        
        # For simplicity in evaluation, use argmax level
        # In inference, we'll use soft weighting
        top5_indices = torch.topk(similarities, k=5, dim=1).indices  # [B, 5]
        
        # Metrics
        total_level_correct += (level_preds == level_targets).sum().item()
        total_top1_correct += (top5_indices[:, 0] == positive_idx).sum().item()
        
        for i in range(batch_size):
            if positive_idx[i] in top5_indices[i]:
                total_top5_correct += 1
            
            # Level-aware: correct level AND correct category
            if level_preds[i] == level_targets[i] and top5_indices[i, 0] == positive_idx[i]:
                total_level_aware_correct += 1
        
        total_samples += batch_size
        
        all_level_preds.extend(level_preds.cpu().numpy())
        all_level_targets.extend(level_targets.cpu().numpy())
    
    metrics = {
        'level_accuracy': total_level_correct / total_samples,
        'top1_accuracy': total_top1_correct / total_samples,
        'top5_accuracy': total_top5_correct / total_samples,
        'level_aware_accuracy': total_level_aware_correct / total_samples
    }
    
    model.train()
    return metrics

# ============================================================================
# Training
# ============================================================================
def train_stage1(model, train_loader, val_loader, category_embeddings):
    """Stage 1: Train level predictor only (encoder frozen)"""
    print("\n" + "="*80)
    print("STAGE 1: Training Level Predictor (Encoder Frozen)")
    print("="*80)
    
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Only train level predictor
    optimizer = torch.optim.AdamW(
        model.level_predictor.parameters(),
        lr=config.stage1_lr,
        weight_decay=config.weight_decay
    )
    
    best_val_acc = 0.0
    metrics_history = []
    
    for epoch in range(config.stage1_epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.stage1_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            level_targets = batch['level'].to(config.device)
            
            # Forward pass
            _, level_logits = model(input_ids, attention_mask)
            
            # Compute loss
            loss = compute_level_loss(level_logits, level_targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.level_predictor.parameters(), config.max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, category_embeddings, "Val")
        
        print(f"\nEpoch {epoch+1}/{config.stage1_epochs}")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val Level Acc: {val_metrics['level_accuracy']:.4f}")
        print(f"  Val Top-1 Acc: {val_metrics['top1_accuracy']:.4f}")
        print(f"  Val Top-5 Acc: {val_metrics['top5_accuracy']:.4f}")
        
        # Save metrics
        metrics_history.append({
            'epoch': epoch + 1,
            'stage': 1,
            'train_loss': avg_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        })
        
        # Save best model
        if val_metrics['level_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['level_accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
            }, os.path.join(config.checkpoint_dir, 'stage1_best.pt'))
            print(f"  → Saved best model (val_acc: {best_val_acc:.4f})")
    
    # Unfreeze encoder for stage 2
    for param in model.encoder.parameters():
        param.requires_grad = True
    
    return metrics_history

def train_stage2(model, train_loader, val_loader, category_embeddings, 
                 category_names_list, tokenizer):
    """Stage 2: Joint fine-tuning with contrastive loss"""
    print("\n" + "="*80)
    print("STAGE 2: Joint Fine-tuning")
    print("="*80)
    
    # Discriminative learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config.stage2_lr_encoder},
        {'params': model.level_predictor.parameters(), 'lr': config.stage2_lr_predictor}
    ], weight_decay=config.weight_decay)
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * config.stage2_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_val_acc = 0.0
    metrics_history = []
    
    for epoch in range(config.stage2_epochs):
        model.train()
        total_loss = 0
        total_level_loss = 0
        total_contrastive_loss = 0
        
        # Re-compute category embeddings at start of each epoch
        print("Computing category embeddings...")
        category_embeddings = compute_category_embeddings(model, category_names_list, tokenizer)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.stage2_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            positive_idx = batch['positive_idx'].to(config.device)
            negative_indices = batch['negative_indices'].to(config.device)
            level_targets = batch['level'].to(config.device)
            
            # Forward pass
            query_embeddings, level_logits = model(input_ids, attention_mask)
            
            # Compute losses
            level_loss = compute_level_loss(level_logits, level_targets)
            contrastive_loss = compute_contrastive_loss(
                query_embeddings, category_embeddings,
                positive_idx, negative_indices, config.temperature
            )
            
            # Total loss
            loss = level_loss + config.lambda_weight * contrastive_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            total_level_loss += level_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'level_loss': level_loss.item(),
                'cont_loss': contrastive_loss.item(),
                'lr': scheduler.get_last_lr()[0]
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_level_loss = total_level_loss / len(train_loader)
        avg_cont_loss = total_contrastive_loss / len(train_loader)
        
        # Evaluate
        category_embeddings = compute_category_embeddings(model, category_names_list, tokenizer)
        val_metrics = evaluate(model, val_loader, category_embeddings, "Val")
        
        print(f"\nEpoch {epoch+1}/{config.stage2_epochs}")
        print(f"  Train Loss: {avg_loss:.4f} (Level: {avg_level_loss:.4f}, Contrastive: {avg_cont_loss:.4f})")
        print(f"  Val Level Acc: {val_metrics['level_accuracy']:.4f}")
        print(f"  Val Top-1 Acc: {val_metrics['top1_accuracy']:.4f}")
        print(f"  Val Top-5 Acc: {val_metrics['top5_accuracy']:.4f}")
        print(f"  Val Level-Aware Acc: {val_metrics['level_aware_accuracy']:.4f}")
        
        # Save metrics
        metrics_history.append({
            'epoch': epoch + 1,
            'stage': 2,
            'train_loss': avg_loss,
            'train_level_loss': avg_level_loss,
            'train_contrastive_loss': avg_cont_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        })
        
        # Save best model
        combined_score = val_metrics['level_accuracy'] + val_metrics['top1_accuracy']
        if combined_score > best_val_acc:
            best_val_acc = combined_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, os.path.join(config.checkpoint_dir, 'stage2_best.pt'))
            print(f"  → Saved best model (combined_score: {combined_score:.4f})")
    
    return metrics_history

def compute_category_embeddings(model, category_names, tokenizer):
    """Compute embeddings for all categories"""
    model.eval()
    embeddings_list = []
    
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(category_names), batch_size):
            batch_names = category_names[i:i+batch_size]
            
            encoded = tokenizer(
                batch_names,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(config.device)
            attention_mask = encoded['attention_mask'].to(config.device)
            
            embeddings = model.encode_text(input_ids, attention_mask)
            embeddings_list.append(embeddings)
    
    model.train()
    return torch.cat(embeddings_list, dim=0)

# ============================================================================
# Main
# ============================================================================
def main():
    print("="*80)
    print("Specificity-Aware Semantic Matching - Training")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Model: {config.model_path}")
    print()
    
    # Load data
    (train_data, val_data, test_data, 
     category_tree, id_to_idx, id_to_name, 
     all_category_ids, categories_df) = load_data()
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {config.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    
    # Create datasets
    train_dataset = QueryDataset(train_data, tokenizer, category_tree, id_to_idx, is_train=True)
    val_dataset = QueryDataset(val_data, tokenizer, category_tree, id_to_idx, is_train=False)
    test_dataset = QueryDataset(test_data, tokenizer, category_tree, id_to_idx, is_train=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.stage1_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.stage1_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.stage1_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Initialize model
    print(f"\nInitializing model...")
    num_categories = len(all_category_ids)
    model = SpecificityAwareModel(
        config.model_path,
        config.num_levels,
        num_categories
    ).to(config.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Get category names in order
    category_names_list = [id_to_name[cid] for cid in all_category_ids]
    
    # Compute initial category embeddings
    print("\nComputing initial category embeddings...")
    category_embeddings = compute_category_embeddings(model, category_names_list, tokenizer)
    print(f"Category embeddings shape: {category_embeddings.shape}")
    
    # Stage 1: Train level predictor
    stage1_metrics = train_stage1(model, train_loader, val_loader, category_embeddings)
    
    # Stage 2: Joint fine-tuning
    stage2_metrics = train_stage2(model, train_loader, val_loader, 
                                   category_embeddings, category_names_list, tokenizer)
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("Legacy diagnostic on the validation alias")
    print("="*80)
    
    # Load best stage2 model
    checkpoint = torch.load(os.path.join(config.checkpoint_dir, 'stage2_best.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Compute final category embeddings
    category_embeddings = compute_category_embeddings(model, category_names_list, tokenizer)
    
    test_metrics = evaluate(model, test_loader, category_embeddings, "Validation alias")
    
    print("\nValidation-alias diagnostic (use validation_summary.json):")
    print(f"  Level Accuracy: {test_metrics['level_accuracy']:.4f}")
    print(f"  Top-1 Accuracy: {test_metrics['top1_accuracy']:.4f}")
    print(f"  Top-5 Accuracy: {test_metrics['top5_accuracy']:.4f}")
    print(f"  Level-Aware Accuracy: {test_metrics['level_aware_accuracy']:.4f}")
    
    # Save final model
    print(f"\nSaving final model to {config.output_dir}...")
    model.encoder.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # Save level predictor separately
    torch.save({
        'level_predictor_state_dict': model.level_predictor.state_dict(),
        'num_levels': config.num_levels,
        'embedding_dim': config.embedding_dim,
    }, os.path.join(config.output_dir, 'level_predictor.pt'))
    
    # Save all metrics
    all_metrics = stage1_metrics + stage2_metrics + [{'stage': 'validation_alias_diagnostic', **test_metrics}]
    
    with open(os.path.join(config.log_dir, 'training_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Save config
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    # Convert device to string if it exists
    if 'device' in config_dict:
        config_dict['device'] = str(config_dict['device'])
    # Also handle the case where device is directly accessed
    try:
        config_dict['device'] = str(config.device)
    except:
        pass
    with open(os.path.join(config.log_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("\nTraining completed!")
    print(f"Metrics saved to {config.log_dir}/training_metrics.json")
    print(f"Model saved to {config.output_dir}")

if __name__ == "__main__":
    main()
