from __future__ import annotations
import pandas as pd

def email_mapper(df: pd.DataFrame) -> list[int]:
    coded = {email: i for i, email in enumerate(df['email'].unique(), start=1)}
    return [coded[e] for e in df['email']]
