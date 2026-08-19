# Netflix Titles Regression Analysis

Linear and logistic regression, both implemented from scratch with NumPy (gradient descent, no scikit-learn), applied to the Netflix Titles dataset from Kaggle.

## Files

- `Netflix_Regression.ipynb` — main notebook, two parts
- `netflix_titles.csv` — dataset (8,807 titles)

## Part 1 — Linear Regression

Predicts movie duration (minutes) from release year, using only `Movie` entries (`TV Show` rows are excluded since their `duration` field is measured in seasons, not minutes).

- Feature: `release_year` (normalized)
- Target: `duration_min` (parsed out of the `duration` column)
- 80/20 train/test split, seed = 42
- Gradient descent: 1000 epochs, lr = 0.01
- Result: Test R² ≈ 0.03, Test MSE ≈ 776

The low R² is expected — release year alone barely explains runtime. Most of the variation in movie length comes from genre and content type, which aren't in the model.

## Part 2 — Logistic Regression

Predicts whether a title is a `Movie` or a `TV Show`.

Features:
- `release_year`
- `country_us` — 1 if United States is listed as a production country, else 0
- `genre_count` — number of genres listed under `listed_in`

- 80/20 train/test split, seed = 42
- Gradient descent: 1000 epochs, lr = 0.01
- Result: accuracy ≈ 0.69, AUC ≈ 0.66
- Includes an ROC curve and a best-threshold search (maximizing TPR − FPR)

## Running it

```bash
pip install numpy pandas matplotlib
jupyter notebook Netflix_Regression.ipynb
```

Run the cells top to bottom. The random seed is fixed, so re-running reproduces the same numbers.
