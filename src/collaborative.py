from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

def create_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    mat = (df.assign(val=1)
             .drop_duplicates(['user_id','article_id'])
             .pivot(index='user_id', columns='article_id', values='val')
             .fillna(0).astype(int))
    mat.columns = mat.columns.astype(int)
    return mat

def _similarity_matrix(user_item: pd.DataFrame) -> pd.DataFrame:
    vals = user_item.values.astype(float)
    sims = vals @ vals.T
    norms = np.linalg.norm(vals, axis=1, keepdims=True)
    denom = norms @ norms.T
    denom[denom == 0] = 1.0
    sims = sims / denom
    return pd.DataFrame(sims, index=user_item.index, columns=user_item.index)

def get_top_sorted_users(user_id: int, df: pd.DataFrame, user_item: pd.DataFrame) -> pd.DataFrame:
    sims = _similarity_matrix(user_item)
    if user_id not in sims.index:
        return pd.DataFrame(columns=['neighbor_id','similarity','num_interactions'])
    sim_series = sims.loc[user_id].drop(user_id, errors='ignore')
    n_inter = (df.groupby('user_id')['article_id'].nunique()
                 .reindex(sim_series.index).fillna(0).astype(int))
    out = (pd.DataFrame({'neighbor_id': sim_series.index,
                         'similarity': sim_series.values,
                         'num_interactions': n_inter.values})
             .sort_values(['similarity','num_interactions','neighbor_id'],
                          ascending=[False, False, True])
             .reset_index(drop=True))
    return out

def find_similar_users(user_id: int, user_item: pd.DataFrame, k: int=10) -> List[int]:
    sims = _similarity_matrix(user_item)
    if user_id not in sims.index:
        return []
    sim_series = sims.loc[user_id].drop(user_id, errors='ignore')
    return sim_series.sort_values(ascending=False).head(k).index.astype(int).tolist()

def get_user_articles(user_id: int, df: pd.DataFrame) -> Tuple[List[int], List[str]]:
    user_df = df.loc[df['user_id']==user_id, ['article_id','title']].drop_duplicates()
    ids = user_df['article_id'].astype(int).tolist()
    titles = user_df['title'].tolist()
    return ids, titles

def user_user_recs(user_id: int, df: pd.DataFrame, user_item: pd.DataFrame, m: int=10) -> Tuple[List[int], List[str]]:
    neighbors = get_top_sorted_users(user_id, df, user_item)
    seen = set(df.loc[df['user_id']==user_id, 'article_id'].astype(int).tolist())
    counts: Dict[int,int] = {}
    for _, row in neighbors.iterrows():
        n_id = int(row['neighbor_id'])
        n_seen = set(df.loc[df['user_id']==n_id, 'article_id'].astype(int).tolist())
        for art in n_seen - seen:
            counts[art] = counts.get(art, 0) + 1
        if len(counts) >= m*3:
            break
    ordered = [k for k,_ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:m]
    map_titles = (df[['article_id','title']].drop_duplicates('article_id')
                  .set_index('article_id')['title'].to_dict())
    names = [map_titles.get(a, f"<title no encontrado {a}>") for a in ordered]
    return ordered, names

def user_user_recs_part2(user_id: int, df: pd.DataFrame, user_item: pd.DataFrame, m: int=10) -> List[int]:
    neighbors = get_top_sorted_users(user_id, df, user_item)
    seen = set(df.loc[df['user_id']==user_id, 'article_id'].astype(int).tolist())
    scores: Dict[int,float] = {}
    for _, row in neighbors.iterrows():
        n_id = int(row['neighbor_id'])
        weight = float(row['similarity']) * np.log1p(int(row['num_interactions']))
        n_seen = set(df.loc[df['user_id']==n_id, 'article_id'].astype(int).tolist())
        for art in n_seen - seen:
            scores[art] = scores.get(art, 0.0) + weight
        if len(scores) >= m*5:
            break
    return [k for k,_ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))][:m]
