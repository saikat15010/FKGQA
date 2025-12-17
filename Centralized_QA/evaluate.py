import torch
from transformers import RobertaTokenizer
from tqdm import tqdm
from dataset import MetaQADataset
from models import QuestionEncoder, DistMult

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load models
print("Loading models...")
kg_ckpt = torch.load("distmult.pt", map_location=device)
kg = DistMult(len(kg_ckpt["ent2id"]), len(kg_ckpt["rel2id"]), 200).to(device)
kg.load_state_dict(kg_ckpt["model"])
kg.eval()

qenc = QuestionEncoder(200).to(device)
qenc.load_state_dict(torch.load("qencoder_best.pt", map_location=device))
qenc.eval()

# Load tokenizer and dataset
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
ds = MetaQADataset("data/qa_test.txt", kg_ckpt["ent2id"])

print(f"Test samples: {len(ds)}")

# Evaluation
hits1 = hits10 = mrr = 0
total = len(ds)

print("\nEvaluating on test set...")
with torch.no_grad():
    for s in tqdm(ds):
        # Encode question
        toks = tokenizer(s["question"], return_tensors="pt").to(device)
        q = qenc(**toks)
        
        # Get head entity embedding
        h = kg.ent(torch.tensor([s["head_id"]], device=device))
        
        # Score all entities
        scores = kg.score(h, q).squeeze()
        
        # Get best rank among all correct answers
        ranks = []
        for answer_id in s["answer_ids"]:
            # Count how many entities scored higher than this answer
            rank = (scores > scores[answer_id]).sum().item() + 1
            ranks.append(rank)
        
        # Use the best (minimum) rank
        best_rank = min(ranks)
        
        # Update metrics
        mrr += 1.0 / best_rank
        hits1 += int(best_rank <= 1)
        hits10 += int(best_rank <= 10)

# Print results
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"Hits@1:  {hits1/total:.4f} ({hits1}/{total})")
print(f"Hits@10: {hits10/total:.4f} ({hits10}/{total})")
print(f"MRR:     {mrr/total:.4f}")
print("="*50)

# Optional: Show some example predictions
print("\nExample predictions:")
with torch.no_grad():
    for i in range(min(5, len(ds))):
        s = ds[i]
        toks = tokenizer(s["question"], return_tensors="pt").to(device)
        q = qenc(**toks)
        h = kg.ent(torch.tensor([s["head_id"]], device=device))
        scores = kg.score(h, q).squeeze()
        
        # Get top-5 predictions
        top5_indices = scores.topk(5).indices.tolist()
        
        # Get entity names (reverse lookup)
        id2ent = {v: k for k, v in kg_ckpt["ent2id"].items()}
        top5_entities = [id2ent[idx] for idx in top5_indices]
        true_answers = [id2ent[aid] for aid in s["answer_ids"]]
        
        print(f"\nQ: {s['question']}")
        print(f"True: {', '.join(true_answers)}")
        print(f"Pred: {', '.join(top5_entities)}")
