# SentimentScope: Sentiment Analysis using Transformers

A transformer-based sentiment analysis model built from scratch to classify IMDB movie reviews as positive or negative. Developed as part of the CineScope recommendation system to better understand user sentiment and deliver more personalized experiences.

## Project Overview

SentimentScope uses a custom GPT-style transformer architecture adapted for binary classification. The model processes tokenized movie reviews and outputs sentiment predictions (positive/negative).

### Key Components

- **Data Preparation** — Loading, exploring, and splitting the IMDB dataset (25,000 train / 25,000 test reviews)
- **Custom Dataset & DataLoader** — PyTorch `Dataset` class with BERT subword tokenization (`bert-base-uncased`)
- **Transformer Architecture** — Custom `DemoGPT` model with multi-head attention, feed-forward layers, and a classification head
- **Training Pipeline** — Epoch-based training with AdamW optimizer and cross-entropy loss
- **Evaluation** — Accuracy calculation on validation and test sets

### Results

- **Test Accuracy: 77.07%** (exceeds the 75% target)
- Validation Accuracy: 78.04% after 3 epochs
- Model checkpoint saved as `sentiment_model_checkpoint.pth`

## Getting Started

### Prerequisites

- Python 3.12
- pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd movie_reviews

# Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset

The IMDB dataset is provided as `aclImdb_v1.tar.gz`. Extract it before running the notebook:

```bash
tar -xzf aclImdb_v1.tar.gz
```

### Running the Notebook

Open `SentimentScope_starter.ipynb` in Jupyter or VS Code and run all cells sequentially.

## Project Structure

```
movie_reviews/
├── SentimentScope_starter.ipynb   # Main notebook with all code
├── requirements.txt               # Python dependencies
├── sentiment_model_checkpoint.pth # Trained model weights
├── aclImdb_v1.tar.gz              # IMDB dataset archive
├── aclImdb/                       # Extracted dataset
│   ├── train/
│   │   ├── pos/                   # Positive training reviews
│   │   └── neg/                   # Negative training reviews
│   └── test/
│       ├── pos/                   # Positive test reviews
│       └── neg/                   # Negative test reviews
└── README.md
```

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Vocabulary Size | 30,522 (BERT) |
| Embedding Dimension | 128 |
| Context Size | 128 tokens |
| Transformer Layers | 4 |
| Attention Heads | 4 |
| Dropout Rate | 0.1 |
| Learning Rate | 3e-4 |
| Batch Size | 32 |
| Epochs | 3 |

## Dependencies

See [requirements.txt](requirements.txt) for the full list. Key packages:

- `torch==2.2.2`
- `transformers==4.57.6`
- `pandas==3.0.2`
- `matplotlib==3.10.8`
- `numpy==1.26.4`