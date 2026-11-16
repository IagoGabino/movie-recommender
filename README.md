# Movie Recommender System

Predicts user ratings for movies by combining collaborative and content-based methods.

Machine Learning coursework (2024/1, UFMG). Individual project.

## Approach

- **Items with ratings:** Funk SVD (matrix factorization trained by gradient descent), implemented by hand and in PyTorch.
- **Items without ratings (cold start):** content-based model using TF-IDF over movie tags and cosine similarity.
- Pipeline split into a data loader, a data processor (tags and similarity matrix) and a recommender class.

## Running

The data is not included. Place `ratings.jsonl`, `content.jsonl` and `targets.csv` next to `main.ipynb` and run it. Requirements: pandas, numpy, scikit-learn, torch.
