# Centralized Question Answering over Knowledge Graphs

A neural QA system that combines **DistMult** (knowledge graph embeddings) with **RoBERTa** (language model) to answer questions about structured knowledge.

## How It Works

1. **DistMult** learns embeddings for entities and relations from a knowledge graph
2. **RoBERTa** encodes questions as "virtual relations"  
3. System scores all entities to find the answer

**Example:**
```
Question: "Who directed [Fight Club]?"
Answer: David Fincher (highest scoring entity)
```

## Installation
```bash
pip install torch transformers numpy tqdm
```

## Data Format

Create a `data/` folder with these files:

**kb.txt** (Knowledge graph triples):
```
Brad Pitt|acted_in|Fight Club
Fight Club|directed_by|David Fincher
Fight Club|release_year|1999
```

**qa_train.txt, qa_dev.txt, qa_test.txt** (Questions with tab-separated answers):
```
Who directed [Fight Club]?	David Fincher
What movies did [Brad Pitt] act in?	Fight Club|Troy
When was [Fight Club] released?	1999
```

**Dataset Recommendation:** Use [MetaQA](https://github.com/yuyuz/MetaQA) (movie domain QA dataset)
```
I have used the 1-hop QA dataset
```

## Usage
```bash
# 1. Train knowledge graph embeddings
python kg_train.py

# 2. Train question encoder
python qa_train.py

# 3. Evaluate
python evaluate.py
```

## Results

Expected performance on MetaQA:
- **Hits@1**: 70-85%
- **Hits@10**: 90-95%
- **MRR**: 75-88%

## Configuration

Adjust hyperparameters in the scripts:

**kg_train.py:**
```python
epochs = 100        # Training epochs
dim = 200          # Embedding dimension
batch_size = 128   # Batch size
```

**qa_train.py:**
```python
num_epochs = 10    # Training epochs
lr = 2e-5          # Learning rate
batch_size = 16    # Batch size
```

## Project Structure
```
├── data/
│   ├── kb.txt              # Knowledge graph
│   ├── qa_train.txt        # Training questions
│   ├── qa_dev.txt          # Validation questions
│   └── qa_test.txt         # Test questions
├── dataset.py              # Data loader
├── models.py               # DistMult + RoBERTa models
├── kg_train.py            # Train KG embeddings
├── qa_train.py            # Train question encoder
└── evaluate.py            # Evaluation script
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- Transformers >= 4.30
- NumPy, tqdm
