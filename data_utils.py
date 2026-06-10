import re
import logging
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gpu_utils import seed as gpu_seed
from config import RAW_CSV, TEST_SIZE, VAL_SIZE, RANDOM_SEED

logger = logging.getLogger(__name__)

_TEXT_ALIASES  = {"text", "document", "content", "body", "article",
                  "review", "sentence", "description", "abstract", "paragraph"}
_LABEL_ALIASES = {"label", "category", "class", "target", "tag",
                  "type", "genre", "topic", "subject"}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>",          " ", text)
    text = re.sub(r"http\S+|www\S+",   " ", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _sniff_separator(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        header = f.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def _has_header(path: str, sep: str) -> bool:
    with open(path, encoding="utf-8", errors="replace") as f:
        first = f.readline().strip().split(sep)
    return all(not c.strip().strip('"').lstrip("-").replace(".", "").isdigit()
               and len(c.strip()) < 40 for c in first[:3])


def _find_columns(df: pd.DataFrame):
    cols_lower = {c.strip().lower(): c for c in df.columns}
    text_col  = next((cols_lower[k] for k in _TEXT_ALIASES  if k in cols_lower), None)
    label_col = next((cols_lower[k] for k in _LABEL_ALIASES if k in cols_lower), None)
    if text_col is None or label_col is None:
        str_cols = [c for c in df.columns
                    if df[c].dtype == object or str(df[c].dtype).startswith("string")]
        if len(str_cols) >= 2:
            avg = {c: df[c].dropna().astype(str).str.len().mean() for c in str_cols}
            srt = sorted(avg, key=avg.get)
            if label_col is None: label_col = srt[0]
            if text_col  is None: text_col  = srt[-1]
        elif len(str_cols) == 1:
            raise ValueError(f"Only one string column found: {str_cols}")
        else:
            raise ValueError(f"No string columns found in {list(df.columns)}")
    if text_col == label_col:
        raise ValueError(f"text and label resolved to same column: '{text_col}'")
    logger.info("Columns — text: '%s'  label: '%s'", text_col, label_col)
    return text_col, label_col


def load_dataset(path: str = RAW_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download from https://www.kaggle.com/datasets/sunilthite/"
            "text-document-classification-dataset")
    sep     = _sniff_separator(path)
    has_hdr = _has_header(path, sep)
    logger.info("Loading '%s'  sep=%r  header=%s", path, sep, has_hdr)
    read_kw = dict(sep=sep, encoding="utf-8", encoding_errors="replace",
                   on_bad_lines="skip", quotechar='"', engine="python")
    if not has_hdr:
        read_kw["header"] = None
    df = pd.read_csv(path, **read_kw)
    if not has_hdr:
        df.columns = [f"col{i}" for i in range(len(df.columns))]
    df.columns = [str(c).strip() for c in df.columns]
    logger.info("Raw shape: %s  columns: %s", df.shape, list(df.columns))
    text_col, label_col = _find_columns(df)
    df = df[[label_col, text_col]].copy()
    df.columns = ["label", "text"]
    df = df.dropna(subset=["label", "text"])
    df = df[df["text"].astype(str).str.strip() != ""].reset_index(drop=True)
    df["text_clean"] = df["text"].apply(clean_text)
    logger.info("Loaded %d samples, %d classes: %s",
                len(df), df["label"].nunique(), sorted(df["label"].unique().tolist()))
    return df


def split_dataset(df: pd.DataFrame,
                  test_size: float = TEST_SIZE,
                  val_size:  float = VAL_SIZE,
                  seed:      int   = RANDOM_SEED):
    le = LabelEncoder()
    df = df.copy()
    df["label_enc"] = le.fit_transform(df["label"])
    X, y = df["text_clean"].values, df["label_enc"].values
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=rel_val, stratify=y_tv, random_state=seed)
    logger.info("Split — train:%d  val:%d  test:%d",
                len(X_train), len(X_val), len(X_test))
    return X_train, y_train, X_val, y_val, X_test, y_test, le


def prepare_data(path: str = RAW_CSV):
    return split_dataset(load_dataset(path))
