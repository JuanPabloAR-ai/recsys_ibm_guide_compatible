from __future__ import annotations
from typing import Tuple, List
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

def fit_svd(user_item: pd.DataFrame, k: int=50):
    R = csr_matrix(user_item.values, dtype=float)
    U, s, Vt = svds(R, k=min(k, min(R.shape)-1))
    idx = np.argsort(-s)
    return U[:, idx], s[idx], Vt[idx, :]

def get_svd_similar_article_ids(article_id: int, user_item: pd.DataFrame, Vt, include_similarity: bool=False, m: int=10):
    cols = user_item.columns.astype(int).tolist()
    if int(article_id) not in cols:
        return [] if not include_similarity else []
    item_idx = cols.index(int(article_id))
    item_vecs = Vt.T
    sims = cosine_similarity(item_vecs[item_idx].reshape(1,-1), item_vecs).ravel()
    order = np.argsort(-sims)
    if include_similarity:
        return [[cols[i], float(sims[i])] for i in order if cols[i] != int(article_id)][:m]
    return [cols[i] for i in order if cols[i] != int(article_id)][:m]
