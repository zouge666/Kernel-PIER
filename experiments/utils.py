from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def make_stable_seed(
    text: str,
    theta: float,
    context_type: Optional[str] = None,
    ctx_label: Optional[str] = None,
    base_seed: int = 2026,
) -> int:
    parts = [str(base_seed)]
    if context_type is not None:
        parts.append(str(context_type))
    if ctx_label is not None:
        parts.append(str(ctx_label))
    parts.append(f"{theta:.6f}")
    parts.append(text)
    digest = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**32)


def load_sst2_sentences(max_samples: int, seed: int):
    ds = load_dataset("glue", "sst2", split="validation")
    dataset_size = len(ds)
    sample_size = min(int(max_samples), dataset_size)
    if sample_size < 4:
        raise ValueError("max_samples must select at least four SST-2 validation rows.")

    all_ids = np.arange(dataset_size, dtype=np.int64)
    labels = np.asarray(ds["label"], dtype=np.int64)
    if sample_size < dataset_size:
        selected_ids, _ = train_test_split(
            all_ids,
            train_size=sample_size,
            random_state=int(seed),
            stratify=labels,
        )
    else:
        selected_ids = all_ids

    n_fit = len(selected_ids) // 2
    fit_ids, eval_ids = train_test_split(
        selected_ids,
        train_size=n_fit,
        random_state=int(seed) + 1,
        stratify=labels[selected_ids],
    )
    fit_ids = np.asarray(fit_ids, dtype=np.int64)
    eval_ids = np.asarray(eval_ids, dtype=np.int64)
    fit_texts = [str(ds[int(i)]["sentence"]) for i in fit_ids]
    eval_texts = [str(ds[int(i)]["sentence"]) for i in eval_ids]
    metadata = {
        "data_backend": "huggingface_datasets",
        "dataset_id": "glue",
        "dataset_config": "sst2",
        "dataset_split": "validation",
        "dataset_fingerprint": str(getattr(ds, "_fingerprint", "unresolved")),
        "dataset_num_rows": int(dataset_size),
        "sampling_strategy": "stratified_without_replacement",
        "data_seed": int(seed),
        "fit_dataset_indices": "|".join(str(int(i)) for i in fit_ids),
        "eval_dataset_indices": "|".join(str(int(i)) for i in eval_ids),
        "fit_label_0_count": int(np.sum(labels[fit_ids] == 0)),
        "fit_label_1_count": int(np.sum(labels[fit_ids] == 1)),
        "eval_label_0_count": int(np.sum(labels[eval_ids] == 0)),
        "eval_label_1_count": int(np.sum(labels[eval_ids] == 1)),
    }
    return fit_texts, eval_texts, fit_ids, eval_ids, metadata
