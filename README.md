# Zomato Bangalore Restaurant Rating Prediction

Predicting a restaurant's Zomato rating (`rate`, out of 5) from its attributes, using a full machine learning workflow: data cleaning, visualization, EDA, leakage-safe preprocessing, and tuned regression models.

**Best result:** LightGBM / XGBoost / Random Forest reach a test **R² ≈ 0.53** and an average error of about **0.21 rating points** (MAE), compared with R² = 0 and an error of 0.35 for a baseline that always predicts the mean.

## Dataset

[Zomato Bangalore Restaurants](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants) (Kaggle, by Himanshu Poddar): 51,717 rows × 17 columns scraped from Zomato's listing pages.

The data is **not included** in this repository. Data on kaggle: https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants .

## Workflow

1. **Data cleaning**
   - Converted `rate` (`"4.1/5"`, `"NEW"`, `"-"`) and `approx_cost` (`"1,200"`) to numbers; dropped unrated restaurants (the target is never imputed).
   - **Deduplicated** the data: each row is a *listing*, so the same restaurant appears once per category list. Rows were collapsed to one per restaurant (`name` + `address`), and the `listed_in(type)` categories were merged into 0/1 flags instead of being thrown away.
   - Engineered target-free features: chain size, number of cuisines and restaurant types, whether dish/menu information exists.
   - Final data: **9,438 rated restaurants**.
2. **Visualization**: distribution of `rate`, `votes` and `cost`, binary features vs. rating, `votes`/`cost` vs. rating, and location analysis.
3. **EDA and preprocessing** (every learned step is fitted on the training set only, to prevent data leakage)
   - 80/20 train/test split.
   - **Multi-valued columns** (`rest_type`, `cuisines`): rare values grouped into `others`, then multi-hot encoded so a restaurant can belong to several categories.
   - `location`: rare areas grouped into `others`, then one-hot encoded.
   - Log transform of skewed columns (`votes`, `cost`, `n_outlets`) and standardization (scaled data for linear/distance-based models, unscaled for tree-based models).
4. **Modeling**: baseline, default models, then randomized search with 5-fold cross-validation on the training set for the three best models.

## Results

Test set (1,888 restaurants):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (mean) | 0.347 | 0.430 | 0.000 |
| Linear Regression | 0.230 | 0.317 | 0.456 |
| KNN (k=5) | 0.234 | 0.317 | 0.455 |
| Decision Tree (depth 7) | 0.234 | 0.324 | 0.431 |
| SVR (default) | 0.216 | 0.306 | 0.492 |
| Random Forest (default) | 0.219 | 0.305 | 0.496 |
| XGBoost (default) | 0.224 | 0.309 | 0.485 |
| LightGBM (default) | 0.214 | 0.296 | 0.527 |
| **Random Forest (tuned)** | 0.213 | 0.295 | 0.529 |
| **XGBoost (tuned)** | 0.213 | 0.295 | 0.531 |
| **LightGBM (tuned)** | 0.214 | 0.295 | 0.531 |

The three tuned ensembles are effectively tied. An unrestricted decision tree memorizes the training set (train R² = 0.9999 vs. test R² = 0.05), so its depth had to be limited.

## Key learnings

- Check what a "row" means before modeling: 51,717 rows describe only about 12,500 distinct restaurants.
- Multi-valued columns are best handled by splitting and multi-hot encoding, not by treating each combination as a category.
- Anything that learns from the data (category counts, scaling) must be fitted on the training set only.
- Tree-based models don't need scaling; linear, KNN and SVR models do.

## How to run

```bash
git clone https://github.com/<your-username>/zomato-bangalore-rating-prediction.git
cd zomato-bangalore-rating-prediction
pip install -r requirements.txt
jupyter notebook zomato-restaurants-project.ipynb
```

## Tech stack

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · XGBoost · LightGBM

## Limitations and future work

- Only a few restaurant attributes are available (no review text, food quality or service), so about half of the variation in ratings remains unexplained.
- `votes` and `has_dish_liked` partly reflect popularity.
- Linear Regression, KNN and SVR were left at default settings.
- Next steps: a classification version (good / not good, `rate >= 3.8`), review-text features, and weighting restaurants by `votes`.
