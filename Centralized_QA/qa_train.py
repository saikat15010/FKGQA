import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from tqdm import tqdm
from dataset import MetaQADataset, qa_collate_fn
from models import QuestionEncoder, DistMult

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load pretrained KG model
print("Loading DistMult model...")
ckpt = torch.load("distmult.pt", map_location=device)
kg = DistMult(
    len(ckpt["ent2id"]),
    len(ckpt["rel2id"]),
    200
).to(device)
kg.load_state_dict(ckpt["model"])
kg.eval()

# Freeze KG parameters
for p in kg.parameters():
    p.requires_grad = False

# Initialize question encoder
print("Initializing Question Encoder...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
qenc = QuestionEncoder(200).to(device)
opt = torch.optim.Adam(qenc.parameters(), lr=2e-5)

# Load datasets
print("Loading datasets...")
train_ds = MetaQADataset("data/qa_train.txt", ckpt["ent2id"])
dev_ds = MetaQADataset("data/qa_dev.txt", ckpt["ent2id"])

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=qa_collate_fn)
dev_dl = DataLoader(dev_ds, batch_size=16, shuffle=False, collate_fn=qa_collate_fn)

print(f"Train samples: {len(train_ds)}, Dev samples: {len(dev_ds)}")


def compute_loss(scores, answer_ids_batch):
    """
    Compute loss for multi-answer questions using BCE with logits
    scores: (batch_size, num_entities)
    answer_ids_batch: list of lists of answer entity IDs
    """
    batch_size, num_entities = scores.shape
    
    # Create target tensor
    targets = torch.zeros(batch_size, num_entities, device=device)
    for i, answer_ids in enumerate(answer_ids_batch):
        targets[i, answer_ids] = 1.0
    
    # Binary cross entropy loss
    loss = F.binary_cross_entropy_with_logits(scores, targets)
    return loss


def evaluate(model, kg, dataloader, tokenizer):
    """Evaluate on dev/test set"""
    model.eval()
    hits1 = hits10 = mrr = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            toks = tokenizer(
                batch["question"],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            q = model(**toks)
            h = kg.ent(batch["head_id"].to(device))
            scores = kg.score(h, q)
            
            # Evaluate each sample
            for i, answer_ids in enumerate(batch["answer_ids"]):
                sample_scores = scores[i]
                
                # Get best rank among all correct answers
                ranks = []
                for aid in answer_ids:
                    rank = (sample_scores > sample_scores[aid]).sum().item() + 1
                    ranks.append(rank)
                
                best_rank = min(ranks)
                
                mrr += 1.0 / best_rank
                hits1 += int(best_rank <= 1)
                hits10 += int(best_rank <= 10)
                total += 1
    
    model.train()
    return hits1 / total, hits10 / total, mrr / total


# Training loop
print("\nStarting training...")
num_epochs = 5
best_dev_hits1 = 0

for epoch in range(num_epochs):
    qenc.train()
    total_loss = 0
    
    pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in pbar:
        toks = tokenizer(
            batch["question"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        q = qenc(**toks)
        h = kg.ent(batch["head_id"].to(device))
        scores = kg.score(h, q)
        
        # Compute loss with multi-answer support
        loss = compute_loss(scores, batch["answer_ids"])
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_train_loss = total_loss / len(train_dl)
    
    # Evaluate on dev set
    print(f"\nEvaluating on dev set...")
    dev_h1, dev_h10, dev_mrr = evaluate(qenc, kg, dev_dl, tokenizer)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Dev Hits@1: {dev_h1:.4f}, Hits@10: {dev_h10:.4f}, MRR: {dev_mrr:.4f}")
    
    # Save best model
    if dev_h1 > best_dev_hits1:
        best_dev_hits1 = dev_h1
        torch.save(qenc.state_dict(), "qencoder_best.pt")
        print(f"  ✓ New best model saved! (Hits@1: {dev_h1:.4f})")

# Save final model
torch.save(qenc.state_dict(), "qencoder.pt")
print("\nTraining complete! Final model saved to qencoder.pt")
print(f"Best dev Hits@1: {best_dev_hits1:.4f}")