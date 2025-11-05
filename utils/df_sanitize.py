import json
import math
from typing import Any, List

import numpy as np
import pandas as pd


def _to_scalar_str(x: Any):
    """Convert arbitrary value to Arrow-friendly scalar/str representation."""
    # None/NaN → None (пусть Arrow видит пустоту)
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
    except Exception:
        pass

    # bytes → utf-8 (с подстановкой)
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", errors="replace")
        except Exception:
            return str(x)

    # коллекции → JSON
    if isinstance(x, (list, tuple, set, dict)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)

    # исключения/объекты → строка
    if not isinstance(x, (str, int, float, bool, np.number)):
        return str(x)

    return x


def sanitize_for_arrow(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of DataFrame safe to preview with Arrow-backed APIs."""
    df = df.copy()

    for col in df.columns:
        s = df[col]

        # Даты с tz → naive
        if pd.api.types.is_datetime64tz_dtype(s):
            df[col] = s.dt.tz_convert(None)
            continue

        # Object / смешанный тип → маппим в скаляры/строки
        if s.dtype == "object":
            df[col] = s.map(_to_scalar_str)

        # Категории → строки (иногда Arrow ломается на странных категориях)
        elif pd.api.types.is_categorical_dtype(s):
            df[col] = s.astype(str)

    return df


def find_bad_cols(df: pd.DataFrame, sample_size: int = 100) -> List[str]:
    """Return columns that pyarrow fails to materialise from the DataFrame."""
    try:
        import pyarrow as pa  # type: ignore
    except ImportError:
        return []

    bad: List[str] = []
    sample = df.head(sample_size)
    for column in sample.columns:
        try:
            pa.array(sample[column], from_pandas=True)
        except Exception:
            bad.append(column)
    return bad
