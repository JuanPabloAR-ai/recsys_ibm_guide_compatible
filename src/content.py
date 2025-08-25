from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

def build_tfidf_from_df(df: pd.DataFrame, text_cols: Optional[list]=None, max_features: int=8000, ngram_range=(1,2)):
    if text_cols is None:
        text_cols = ['doc_full','doc_body','content','text','description','title']
    cols = [c for c in text_cols if c in df.columns]
    if not cols:
        raise ValueError("No hay columnas de texto disponibles para TF-IDF.")
    texts = df[cols[0]].fillna('').astype(str)
    vect = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words='english')
    X = vect.fit_transform(texts)
    return vect, X

def select_optimal_k(X, k_min: int=5, k_max: int=40, step: int=5) -> Tuple[int, float]:
    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max+1, step):
        km = KMeans(n_clusters=k, n_init='auto', random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) > 1:
            sc = silhouette_score(X, labels)
            if sc > best_score:
                best_k, best_score = k, sc
    return best_k, float(best_score)

def make_content_recs(article_id: int, df: pd.DataFrame, m: int=10, vect=None, X=None) -> List[str]:
    if vect is None or X is None:
        vect, X = build_tfidf_from_df(df, text_cols=['title'])
    try:
        idx = df[['article_id']].reset_index().set_index('article_id').loc[int(article_id),'index']
    except Exception:
        return []
    sims = cosine_similarity(X[idx], X).ravel()
    order = np.argsort(-sims)
    ids = df.iloc[order]['article_id'].astype(int).tolist()
    titles_map = (df[['article_id','title']].drop_duplicates('article_id')
                  .set_index('article_id')['title'].to_dict())
    recs = [titles_map.get(i, f"<title no encontrado {i}>") for i in ids if i != int(article_id)]
    return recs[:m]
