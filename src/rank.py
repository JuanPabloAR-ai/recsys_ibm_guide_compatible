from __future__ import annotations
from typing import List
import pandas as pd

def get_ranked_article_unique_counts(df: pd.DataFrame) -> pd.Series:
    s = (df.groupby('article_id')['user_id']
           .nunique()
           .sort_values(ascending=False))
    s.index = s.index.astype(int)
    return s

def get_top_article_ids(n: int, df: pd.DataFrame) -> List[int]:
    s = get_ranked_article_unique_counts(df).reset_index().sort_values(['user_id','article_id'], ascending=[False, True])
    return s.head(n)['article_id'].astype(int).tolist()

def get_top_articles(n: int, df: pd.DataFrame) -> List[str]:
    ids = get_top_article_ids(n, df)
    mapping = (df[['article_id','title']].drop_duplicates('article_id')
               .set_index('article_id')['title'].to_dict())
    return [mapping.get(int(a), f"<title no encontrado {a}>") for a in ids]
