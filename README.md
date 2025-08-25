# IBM Article Recommendations — Project README

A clean, **rubric‑aligned** implementation of an article recommendation system using the IBM/Udacity interactions dataset.
The project is organized to keep **all reusable code in `src/`** and a **notebook for reporting**. It implements:

1) **Exploratory Data Analysis (EDA)**
2) **Rank‑based Recommendations (popularity)**
3) **User–User Collaborative Filtering**
4) **Content‑based Recommendations** (TF‑IDF on titles)
5) **Matrix Factorization** (SVD) + discussion & validation guidance

> This repo intentionally uses **only** `user-item-interactions.csv` for the main pipeline to match the course notebook.
> You can later plug a fuller content dataset (e.g., `articles_community.csv`) for better content recommendations.

---

## ✅ Rubric Alignment (Checklist)

- [x] EDA variables computed with **exact names**: `median_val`, `user_article_interactions`, `max_views_by_user`, `max_views`, `most_viewed_article_id`, `unique_articles`, `unique_users`, `total_articles`  
- [x] Rank‑based functions: `get_ranked_article_unique_counts`, `get_top_article_ids`, `get_top_articles`  
- [x] Collaborative Filtering functions: `create_user_item_matrix`, `get_top_sorted_users`, `user_user_recs`, `user_user_recs_part2` (+ helpers `find_similar_users`, `get_user_articles`)  
- [x] Content‑based function: `make_content_recs` (TF‑IDF over titles) + `select_optimal_k` (KMeans silhouette)  
- [x] Matrix Factorization: `fit_svd`, `get_svd_similar_article_ids`  
- [x] Notebook runs end‑to‑end and includes a **short discussion + validation plan**

---

## Repository Structure

```
recsys_ibm/
├─ data/                       # Put your CSVs here
│  └─ user-item-interactions.csv
├─ notebooks/
│  └─ clean_guide_compatible.ipynb   # Clean, complete notebook (imports from src)
├─ src/
│  ├─ data.py                         # email_mapper
│  ├─ rank.py                         # popularity / top articles
│  ├─ collaborative.py                # user-user CF
│  ├─ content.py                      # TF-IDF + content recs
│  ├─ matrix_factorization.py         # SVD utilities
│  └─ utils.py                        # small helpers (get_article_names)
├─ tests/
│  └─ test_smoke.py                   # minimal sanity tests
├─ requirements.txt
└─ README.md
```

---

## Environment Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> Python 3.10+ recommended.

---

## Data

Expected file in `data/`:
- `user-item-interactions.csv` with at least:
  - `email` (or `user_id`) — if `email` is present, it is mapped to `user_id` via `data.email_mapper`
  - `article_id` (int)
  - `title` (string) — used for content TF‑IDF
  - (optional additional columns are fine)

---

## Quickstart (Notebook)

Open and run:
```
notebooks/clean_guide_compatible.ipynb
```

What it does:
1. Loads `data/user-item-interactions.csv`
2. Creates `user_id` from `email` if needed
3. Casts `article_id` to `int`
4. **Part I – EDA:** computes rubric variables
5. **Part II – Rank-based:** top N article ids/titles
6. **Part III – Collaborative Filtering:** nearest neighbors + recommendations for a sample user
7. **Part IV – Content-based:** TF‑IDF on titles, `k` selection with silhouette, similar-article recs
8. **Part V – SVD:** latent‑space similar articles
9. Sanity checks and a short discussion

---

## How Each Part Works (Functions)

### 1) EDA
- Variables are computed directly in the notebook with exact names:
  - `median_val`, `user_article_interactions`, `max_views_by_user`, `max_views`,
    `most_viewed_article_id`, `unique_articles`, `unique_users`, `total_articles`.

### 2) Rank‑based (popularity) — `src/rank.py`
- `get_ranked_article_unique_counts(df)` → users‑per‑article series (desc)
- `get_top_article_ids(n, df)` → top N ids (ties broken by lower `article_id`)
- `get_top_articles(n, df)` → top N titles

### 3) User–User CF — `src/collaborative.py`
- `create_user_item_matrix(df)` → binary matrix `users × articles`
- `get_top_sorted_users(user_id, df, user_item)` → neighbors sorted by similarity, interactions, id
- `user_user_recs(user_id, df, user_item, m)` → top‑m rec ids (+ titles in notebook)
- `user_user_recs_part2(...)` → weighted by neighbor similarity × log(1+interactions)
- Helpers: `find_similar_users`, `get_user_articles`

### 4) Content‑based — `src/content.py`
- `build_tfidf_from_df(df, text_cols=['title'])` → TF‑IDF matrix over titles
- `select_optimal_k(X, ...)` → KMeans + silhouette to inspect cluster quality
- `make_content_recs(article_id, df, m, vect, X)` → similar article titles via cosine similarity

> With titles only, silhouette may be modest; direct cosine similarity is still useful.
> For stronger results, plug richer text (e.g., `doc_full`) and pass that column in `text_cols`.

### 5) Matrix Factorization (SVD) — `src/matrix_factorization.py`
- `fit_svd(user_item, k)` → returns `U, s, Vt` (truncated SVD on the binary matrix)
- `get_svd_similar_article_ids(article_id, user_item, Vt, include_similarity=False|True)` →
  top similar items in the latent space

> Choosing `k`: either inspect explained variance (from `s**2`) or use a simple validation split.

---

## Validation Guidance (What to Report)

- **Offline (recommended):** temporal holdout or per‑user split; report `Recall@K`, `MAP@K`.  
- **Online:** A/B test on the recommendations module measuring CTR and dwell time.  
- **Cold‑start:** use **Rank‑based** for new users; use **Content‑based** for new items.

A short discussion is included in the notebook template.

---

## Testing

Run minimal sanity tests:
```bash
python -m pytest -q
```

You can add your own tests under `tests/` to validate functions or shapes.

---

## Troubleshooting

- **IndexError in content‑based:** ensure you deduplicate by `article_id` before aligning TF‑IDF indices (the provided `make_content_recs` already handles this).  
- **Missing `user_id`:** if your CSV has `email`, the notebook maps it with `email_mapper` and drops `email`.  
- **SVD memory:** reduce `k` if your `user_item` matrix is large.  
- **Types:** cast `article_id` to `int` to avoid pivot/indexing issues.

---

## Extending the Project (Optional)

- Use a richer content dataset (`articles_community.csv`) and point `text_cols` to the full text column.  
- Add an **evaluation module** (MAP@K/Recall@K) and automated hyperparameter search (grid over `k`).  
- Package `src/` as a small pip module or expose a **Streamlit** demo for interactive recommendations.

---

## GitHub

```bash
git init
git add .
git commit -m "IBM Recsys: EDA, Rank, CF, Content (TF-IDF), SVD"
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

---

## License

This repository is for educational purposes associated with the Udacity project and IBM interactions dataset.  
Replace this section with your preferred license if you make the repo public.
