import torch
import numpy as np
from tqdm import tqdm
from models import DistMult

def load_kb(path):
    """Load knowledge base triples"""
    triples = []
    ent2id, rel2id = {}, {}

    def get(d, x):
        if x not in d:
            d[x] = len(d)
        return d[x]

    with open(path) as f:
        for line in f:
            h, r, t = line.strip().split("|")
            triples.append((
                get(ent2id, h),
                get(rel2id, r),
                get(ent2id, t)
            ))
    return triples, ent2id, rel2id


def get_batch(triples, batch_size, num_entities):
    """Generate batch with negative sampling"""
    indices = np.random.randint(0, len(triples), batch_size)
    
    h_batch, r_batch, t_batch = [], [], []
    neg_t_batch = []
    
    for idx in indices:
        h, r, t = triples[idx]
        h_batch.append(h)
        r_batch.append(r)
        t_batch.append(t)
        
        # Negative sampling
        neg_t = np.random.randint(0, num_entities)
        while neg_t == t:  # Ensure negative is different
            neg_t = np.random.randint(0, num_entities)
        neg_t_batch.append(neg_t)
    
    return (
        torch.tensor(h_batch),
        torch.tensor(r_batch),
        torch.tensor(t_batch),
        torch.tensor(neg_t_batch)
    )


def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    batch_size = 128
    dim = 200
    lr = 1e-3
    epochs = 20
    
    # Load data
    print("Loading knowledge base...")
    triples, ent2id, rel2id = load_kb("data/kb.txt")
    print(f"Entities: {len(ent2id)}, Relations: {len(rel2id)}, Triples: {len(triples)}")
    
    # Initialize model
    model = DistMult(len(ent2id), len(rel2id), dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr)
    
    # Training
    steps_per_epoch = len(triples) // batch_size
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
        for _ in pbar:
            h, r, t, neg_t = get_batch(triples, batch_size, len(ent2id))
            h, r, t, neg_t = h.to(device), r.to(device), t.to(device), neg_t.to(device)
            
            # Positive and negative scores
            pos_score = model(h, r, t)
            neg_score = model(h, r, neg_t)
            
            # Margin ranking loss
            loss = torch.clamp(1.0 - pos_score + neg_score, min=0).mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / steps_per_epoch
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                "model": model.state_dict(),
                "ent2id": ent2id,
                "rel2id": rel2id,
                "epoch": epoch
            }, f"distmult_epoch_{epoch+1}.pt")
    
    # Save final model
    torch.save({
        "model": model.state_dict(),
        "ent2id": ent2id,
        "rel2id": rel2id
    }, "distmult.pt")
    print("Training complete! Model saved to distmult.pt")


if __name__ == "__main__":
    main()