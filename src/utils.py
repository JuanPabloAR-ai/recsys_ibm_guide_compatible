from __future__ import annotations
from typing import Iterable, List
import pandas as pd

def get_article_names(article_ids: Iterable[int], df: pd.DataFrame) -> List[str]:
    ids = [int(a) for a in list(article_ids)]
    mapping = (df[['article_id','title']]
               .drop_duplicates('article_id')
               .set_index('article_id')['title']
               .to_dict())
    return [mapping.get(int(a), f"<title no encontrado {a}>") for a in ids]
