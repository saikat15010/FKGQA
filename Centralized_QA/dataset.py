import re
import torch
from torch.utils.data import Dataset

ENTITY_RE = re.compile(r"\[(.*?)\]")

class MetaQADataset(Dataset):
    def __init__(self, qa_path, entity2id):
        self.samples = []
        self.entity2id = entity2id

        with open(qa_path, encoding="utf-8") as f:
            for line in f:
                q, ans = line.strip().split("\t")
                head = ENTITY_RE.search(q).group(1)
                answers = ans.split("|")

                self.samples.append({
                    "question": q,
                    "head_id": entity2id[head],
                    "answer_ids": [entity2id[a] for a in answers]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def qa_collate_fn(batch):
    """Custom collate function to handle variable-length answer lists"""
    return {
        "question": [item["question"] for item in batch],
        "head_id": torch.tensor([item["head_id"] for item in batch]),
        "answer_ids": [item["answer_ids"] for item in batch]  # Keep as list
    }