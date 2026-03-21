from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[2]
KERNEL_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(KERNEL_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(KERNEL_PROJECT_DIR))

from isqed.core import ModelUnit
from isqed.ecosystem import Ecosystem
from isqed.geometry import DISCOSolver
from isqed.real_world import HuggingFaceWrapper, MaskingIntervention
from experiments.utils import make_stable_seed
from kernel_isqed import solve_kernel_residuals


class DeterministicTextModel(ModelUnit):
    def __init__(self, name: str, a: float, b: float, c: float):
        super().__init__(name=name)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def _forward(self, text_input):
        text = str(text_input)
        tokens = text.split()
        n = max(1, len(tokens))
        n_mask = sum(1 for t in tokens if t == "[MASK]")
        mask_ratio = n_mask / float(n)
        digest = hashlib.md5((self.name + "||" + text).encode("utf-8")).hexdigest()
        h = (int(digest[:8], 16) % 10000) / 10000.0
        z = self.a * (1.0 - mask_ratio) + self.b * np.sin(4.0 * h + self.c) + 0.2 * np.tanh(2.0 * h - 1.0)
        return float(1.0 / (1.0 + np.exp(-z)))


def assert_honesty_split(fit_ids: np.ndarray, eval_ids: np.ndarray) -> tuple[int, int]:
    fit_set = set(np.asarray(fit_ids).reshape(-1).tolist())
    eval_set = set(np.asarray(eval_ids).reshape(-1).tolist())
    if fit_set & eval_set:
        raise ValueError("Honesty split violated: fit/eval sample ids overlap.")
    return len(fit_set), len(eval_set)


def standardize_by_fit(Y_p_fit: np.ndarray, Y_p_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = Y_p_fit.mean(axis=0)
    sigma = Y_p_fit.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (Y_p_fit - mu) / sigma, (Y_p_eval - mu) / sigma


def median_heuristic_gamma(X: np.ndarray) -> float:
    if X.shape[0] < 2:
        return 1.0
    idx = np.arange(X.shape[0])
    pairs = np.stack(np.meshgrid(idx, idx), axis=-1).reshape(-1, 2)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    if pairs.shape[0] > 15000:
        rng = np.random.RandomState(0)
        sel = rng.choice(pairs.shape[0], size=15000, replace=False)
        pairs = pairs[sel]
    diffs = X[pairs[:, 0]] - X[pairs[:, 1]]
    d2 = np.sum(diffs * diffs, axis=1)
    med = float(np.median(d2))
    if med <= 1e-12:
        return 1.0
    return 1.0 / (2.0 * med)


def build_fallback_models():
    peers = [
        DeterministicTextModel("peer-bert", 3.2, 1.1, 0.1),
        DeterministicTextModel("peer-distilbert", 2.8, 0.9, 0.5),
        DeterministicTextModel("peer-albert", 2.4, 1.3, 1.0),
        DeterministicTextModel("peer-xlnet", 2.2, 1.0, 1.5),
    ]
    targets = [
        {"model": DeterministicTextModel("target-roberta", 1.9, 1.8, 0.9), "name": "Architectural Divergence (RoBERTa)", "type": "Low Redundancy"},
        {"model": peers[1], "name": "Perfect Redundancy (Clone)", "type": "High Redundancy", "clone_of_peer_idx": 1},
        {"model": DeterministicTextModel("target-distil-variant", 2.6, 1.2, 0.3), "name": "Parametric Divergence (Finetuned)", "type": "Uniqueness"},
    ]
    return peers, targets


def build_hf_models(device: str):
    peer_ids = [
        "textattack/bert-base-uncased-SST-2",
        "textattack/distilbert-base-uncased-SST-2",
        "textattack/albert-base-v2-SST-2",
        "textattack/xlnet-base-cased-SST-2",
    ]
    peers = []
    for pid in peer_ids:
        try:
            peers.append(HuggingFaceWrapper(pid, device))
        except Exception:
            continue

    targets = []
    try:
        targets.append(
            {
                "model": HuggingFaceWrapper("textattack/roberta-base-SST-2", device),
                "name": "Architectural Divergence (RoBERTa)",
                "type": "Low Redundancy",
            }
        )
    except Exception:
        pass

    distil_ref = next((p for p in peers if "distilbert" in p.name.lower()), None)
    if distil_ref is not None:
        targets.append(
            {
                "model": distil_ref,
                "name": "Perfect Redundancy (Clone)",
                "type": "High Redundancy",
                "clone_of_peer_idx": peers.index(distil_ref),
            }
        )
    try:
        targets.append(
            {
                "model": HuggingFaceWrapper("distilbert-base-uncased-finetuned-sst-2-english", device),
                "name": "Parametric Divergence (Finetuned)",
                "type": "Uniqueness",
            }
        )
    except Exception:
        pass
    return peers, targets


def load_sst2_sentences(max_samples: int, seed: int):
    try:
        ds = load_dataset("glue", "sst2", split="validation")
        sentences = ds["sentence"][:max_samples]
    except Exception:
        base = [
            "This movie is great.",
            "Terrible acting and boring scenes.",
            "I loved every second of this film.",
            "The storyline felt weak and predictable.",
            "Excellent performance and direction.",
            "I would not recommend this to anyone.",
        ]
        repeats = max(1, (max_samples + len(base) - 1) // len(base))
        sentences = (base * repeats)[:max_samples]

    arr = np.asarray(sentences, dtype=object)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(arr))
    arr = arr[perm]
    n_fit = len(arr) // 2
    fit_texts = arr[:n_fit].tolist()
    eval_texts = arr[n_fit:].tolist()
    fit_ids = perm[:n_fit]
    eval_ids = perm[n_fit:]
    return fit_texts, eval_texts, fit_ids, eval_ids


def run_bert_kernel_audit(
    theta: float,
    lambdas: Iterable[float],
    max_samples: int,
    data_seed: int,
    kernel_type: str,
    gamma: float | None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    peers, targets = build_hf_models(device=device)
    use_fallback = False
    if len(peers) < 2 or len(targets) == 0:
        peers, targets = build_fallback_models()
        use_fallback = True

    fit_texts, eval_texts, fit_ids, eval_ids = load_sst2_sentences(max_samples=max_samples, seed=data_seed)
    fit_size, eval_size = assert_honesty_split(fit_ids=fit_ids, eval_ids=eval_ids)

    intervention = MaskingIntervention()
    lambda_list = [float(v) for v in lambdas]

    rows = []
    for t_info in targets:
        eco = Ecosystem(target=t_info["model"], peers=peers)
        fit_X = fit_texts
        fit_Theta = [float(theta)] * len(fit_X)
        fit_seeds = [int(make_stable_seed(text=x, theta=float(theta))) for x in fit_X]

        y_fit, Y_p_fit = eco.batched_query(X=fit_X, Thetas=fit_Theta, intervention=intervention, seeds=fit_seeds)
        dist_fit, w_hat = DISCOSolver.solve_weights_and_distance(y_fit.reshape(-1, 1), Y_p_fit)
        w_hat = np.asarray(w_hat, dtype=float).reshape(-1)

        eval_X = eval_texts
        eval_Theta = [float(theta)] * len(eval_X)
        eval_seeds = [int(make_stable_seed(text=x, theta=float(theta))) for x in eval_X]
        y_eval, Y_p_eval = eco.batched_query(X=eval_X, Thetas=eval_Theta, intervention=intervention, seeds=eval_seeds)

        y_mix_eval = Y_p_eval @ w_hat
        convex_residuals = np.abs(y_eval - y_mix_eval)
        convex_pier = float(np.mean(convex_residuals))

        X_fit_k, X_eval_k = standardize_by_fit(Y_p_fit, Y_p_eval)
        gamma_used = median_heuristic_gamma(X_fit_k) if gamma is None else float(gamma)
        kernel_result = solve_kernel_residuals(
            y_fit=y_fit,
            Y_p_fit=X_fit_k,
            y_eval=y_eval,
            Y_p_eval=X_eval_k,
            lambdas=lambda_list,
            kernel_type={"name": kernel_type, "gamma": gamma_used},
        )

        for lam in lambda_list:
            ker_res = kernel_result.residuals_by_lambda[float(lam)]
            ker_pier = float(np.mean(np.abs(ker_res)))
            rows.append(
                {
                    "target_model": t_info["name"],
                    "target_group": t_info["type"],
                    "theta": float(theta),
                    "lambda": float(lam),
                    "budget": float(1.0 / float(lam)),
                    "convex_fit_distance": float(dist_fit),
                    "convex_pier": convex_pier,
                    "kernel_pier": ker_pier,
                    "pier_drop": float(convex_pier - ker_pier),
                    "pier_drop_ratio": float((convex_pier - ker_pier) / max(1e-12, convex_pier)),
                    "num_peers": int(len(peers)),
                    "n_fit": int(len(fit_texts)),
                    "n_eval": int(len(eval_texts)),
                    "honesty_fit_size": int(fit_size),
                    "honesty_eval_size": int(eval_size),
                    "kernel_type": kernel_type,
                    "gamma": float(gamma_used),
                    "model_backend": "fallback" if use_fallback else "huggingface",
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Exp3a: BERT kernel audit under masking intervention")
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--kernel-type", type=str, default="rbf")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
    parser.add_argument(
        "--out-csv",
        type=str,
        default="results/tables/exp3a_bert_kernel.csv",
    )
    args = parser.parse_args()

    df = run_bert_kernel_audit(
        theta=args.theta,
        lambdas=args.lambdas,
        max_samples=args.max_samples,
        data_seed=args.data_seed,
        kernel_type=args.kernel_type,
        gamma=args.gamma,
    )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.groupby(["target_model", "target_group", "lambda"], as_index=False)[["convex_pier", "kernel_pier", "pier_drop"]].mean())
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
