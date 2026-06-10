import logging
import os
import urllib.request
import numpy as np
from typing import List, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (
    LLM_SAMPLE_SIZE, LLM_TIMEOUT,
    OLLAMA_BASE_URL, OLLAMA_LLAMA_MODEL, OLLAMA_MISTRAL_MODEL,
    RANDOM_SEED,
)

_env_host = os.environ.get("OLLAMA_HOST", "")
if _env_host:
    OLLAMA_BASE_URL = _env_host if _env_host.startswith("http") else f"http://{_env_host}"

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "Classify the following document into exactly one of these categories: {classes}.\n\n"
    "Document:\n{text}\n\n"
    "Rules:\n"
    "- Reply with ONLY the category name.\n"
    "- Do not add any explanation, punctuation, or extra words.\n"
    "Category:"
)


def build_prompt(text: str, class_names: List[str]) -> str:
    return _PROMPT_TEMPLATE.format(
        classes=", ".join(class_names),
        text=" ".join(text.split()[:256]))


def parse_prediction(raw: str, class_names: List[str]) -> str:
    raw_l = raw.strip().lower()
    for c in class_names:
        if c.lower() == raw_l: return c
    for c in class_names:
        if c.lower() in raw_l: return c
    tok = raw_l.split()[0] if raw_l.split() else ""
    for c in class_names:
        if tok.startswith(c.lower()[:4]): return c
    return class_names[0]


def check_ollama_server(base_url: str = None) -> bool:
    try:
        urllib.request.urlopen(f"{base_url or OLLAMA_BASE_URL}/api/tags", timeout=5)
        return True
    except Exception:
        return False


class OllamaClassifier:
    def __init__(self, model_name: str,
                 sample_size: int = LLM_SAMPLE_SIZE,
                 timeout: int = LLM_TIMEOUT,
                 seed: int = RANDOM_SEED):
        self.model_name  = model_name
        self.sample_size = sample_size
        self.timeout     = timeout
        self.seed        = seed
        self.base_url    = OLLAMA_BASE_URL
        self.class_names: Optional[List[str]] = None
        self.name        = model_name.replace(":", "_")

    def fit(self, X, y, label_encoder=None):
        self.class_names = (list(label_encoder.classes_) if label_encoder
                            else [str(c) for c in sorted(set(y))])
        if not check_ollama_server(self.base_url):
            raise RuntimeError(
                f"Ollama server not reachable at {self.base_url}.\n"
                "Start with: ollama serve  or  bash apptainer/build_and_pull.sh")
        logger.info("Warming up '%s' at %s …", self.model_name, self.base_url)
        try:
            import ollama
            ollama.Client(host=self.base_url).chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                options={"num_predict": 3})
        except Exception as exc:
            raise RuntimeError(
                f"Could not reach '{self.model_name}' at {self.base_url}.\n"
                f"Run: bash apptainer/build_and_pull.sh\nError: {exc}") from exc
        logger.info("'%s' ready. %d classes.", self.model_name, len(self.class_names))
        return self

    def _classify_one(self, text: str) -> str:
        import ollama
        resp = ollama.Client(host=self.base_url).chat(
            model=self.model_name,
            messages=[{"role": "user", "content": build_prompt(text, self.class_names)}],
            options={"temperature": 0, "num_predict": 20, "seed": self.seed})
        return parse_prediction(resp["message"]["content"], self.class_names)

    def predict(self, X) -> np.ndarray:
        if self.class_names is None:
            raise RuntimeError("Call fit() before predict().")
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(X))
        if len(X) > self.sample_size:
            indices = rng.choice(indices, size=self.sample_size, replace=False)
            logger.warning("LLM eval sampled to %d / %d.", self.sample_size, len(X))
        pred_names = np.array([self.class_names[0]] * len(X), dtype=object)
        for k, idx in enumerate(indices):
            if k % 50 == 0:
                logger.info("  [%s] %d / %d …", self.model_name, k, len(indices))
            try:
                pred_names[idx] = self._classify_one(X[idx])
            except Exception as exc:
                logger.warning("  Ollama error on doc %d: %s", idx, exc)
        return np.array(
            [self.class_names.index(p) if p in self.class_names else 0
             for p in pred_names], dtype=int)

    def predict_proba(self, X):
        preds = self.predict(X)
        proba = np.zeros((len(X), len(self.class_names)), dtype=np.float32)
        proba[np.arange(len(X)), preds] = 1.0
        return proba


class LlamaClassifier(OllamaClassifier):
    def __init__(self, **kwargs):
        super().__init__(model_name=OLLAMA_LLAMA_MODEL, **kwargs)
        self.name = "llama3"


class MistralClassifier(OllamaClassifier):
    def __init__(self, **kwargs):
        super().__init__(model_name=OLLAMA_MISTRAL_MODEL, **kwargs)
        self.name = "mistral"


ALL_LLM_MODELS = {"llama": LlamaClassifier, "mistral": MistralClassifier}


def get_llm_model(name: str, **kwargs):
    if name not in ALL_LLM_MODELS:
        raise ValueError(f"Unknown LLM '{name}'. Choose from {list(ALL_LLM_MODELS)}")
    return ALL_LLM_MODELS[name](**kwargs)
