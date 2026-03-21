from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[2]
KERNEL_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(KERNEL_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(KERNEL_PROJECT_DIR))

from isqed.geometry import DISCOSolver
from isqed.real_world import AdversarialFGSMIntervention, ImageModelWrapper
from kernel_isqed import solve_kernel_residuals


class TinyVisionBackbone(nn.Module):
    def __init__(self, seed: int, hidden_scale: float, n_classes: int = 10):
        super().__init__()
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, n_classes, bias=True)
        with torch.no_grad():
            self.conv.weight.mul_(hidden_scale)
            self.fc.weight.mul_(hidden_scale)
            self.fc.bias.mul_(hidden_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.act(h)
        h = self.pool(h).flatten(1)
        out = self.fc(h)
        return out


class TinyRobustBackbone(nn.Module):
    def __init__(self, seed: int, n_classes: int = 10):
        super().__init__()
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.final = nn.Linear(8, n_classes, bias=True)
        with torch.no_grad():
            self.conv.weight.mul_(0.6)
            self.final.weight.mul_(0.6)
            self.final.bias.mul_(0.6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(x)
        h = torch.tanh(self.conv(h))
        h = h.mean(dim=(2, 3))
        out = self.final(h)
        return out


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
    if pairs.shape[0] > 12000:
        rng = np.random.RandomState(0)
        pairs = pairs[rng.choice(pairs.shape[0], size=12000, replace=False)]
    diffs = X[pairs[:, 0]] - X[pairs[:, 1]]
    d2 = np.sum(diffs * diffs, axis=1)
    med = float(np.median(d2))
    if med <= 1e-12:
        return 1.0
    return 1.0 / (2.0 * med)


def build_synthetic_samples(max_samples: int, seed: int) -> List[Tuple[torch.Tensor, int]]:
    rng = np.random.RandomState(seed)
    samples: List[Tuple[torch.Tensor, int]] = []
    for _ in range(max_samples):
        x = torch.tensor(rng.normal(0.0, 1.0, size=(3, 224, 224)), dtype=torch.float32)
        y = int(rng.randint(0, 10))
        samples.append((x, y))
    return samples


def split_fit_eval(
    samples: List[Tuple[torch.Tensor, int]],
    seed: int,
    fit_frac: float = 0.5,
) -> tuple[List[Tuple[torch.Tensor, int]], List[Tuple[torch.Tensor, int]], np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    n = len(samples)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_fit = int(n * fit_frac)
    fit_idx = idx[:n_fit]
    eval_idx = idx[n_fit:]
    fit_samples = [samples[i] for i in fit_idx]
    eval_samples = [samples[i] for i in eval_idx]
    return fit_samples, eval_samples, fit_idx, eval_idx


def load_real_imagefolder_samples(data_root: str, max_samples: int, seed: int):
    try:
        import torchvision.datasets as datasets
        import torchvision.transforms as T
    except Exception:
        return None
    root = Path(data_root)
    if not root.exists():
        return None
    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    try:
        dataset = datasets.ImageFolder(root=str(root), transform=transform)
    except Exception:
        return None
    n = len(dataset)
    if n == 0:
        return None
    rng = np.random.RandomState(seed)
    if max_samples < n:
        ids = rng.choice(n, size=max_samples, replace=False)
    else:
        ids = np.arange(n)
    samples = []
    for i in ids:
        x, y = dataset[int(i)]
        samples.append((x, int(y)))
    return samples


def load_real_models(device: str):
    try:
        import torchvision.models as tv_models
    except Exception:
        return None
    models = []
    loaders = [
        ("ResNet18", lambda: tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)),
        ("ResNet50", lambda: tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)),
        ("MobileNetV3Small", lambda: tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)),
        ("EfficientNetB0", lambda: tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)),
        ("ConvNeXtTiny", lambda: tv_models.convnext_tiny(weights=tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)),
    ]
    for name, fn in loaders:
        try:
            m = fn()
            models.append(ImageModelWrapper(m, name, device, mode="tanh_margin"))
        except Exception:
            continue
    try:
        robust = TinyRobustBackbone(seed=99, n_classes=1000)
        models.append(ImageModelWrapper(robust, "RobustResNet50", device, mode="tanh_margin"))
    except Exception:
        pass
    return models if len(models) >= 4 else None


def build_fallback_models(device: str):
    models = [
        ImageModelWrapper(TinyVisionBackbone(seed=11, hidden_scale=1.0, n_classes=10), "ResNet18", device, mode="tanh_margin"),
        ImageModelWrapper(TinyVisionBackbone(seed=13, hidden_scale=0.9, n_classes=10), "ResNet50", device, mode="tanh_margin"),
        ImageModelWrapper(TinyVisionBackbone(seed=17, hidden_scale=1.1, n_classes=10), "EfficientNetB0", device, mode="tanh_margin"),
        ImageModelWrapper(TinyVisionBackbone(seed=19, hidden_scale=1.2, n_classes=10), "ConvNeXtTiny", device, mode="tanh_margin"),
        ImageModelWrapper(TinyRobustBackbone(seed=23, n_classes=10), "RobustResNet50", device, mode="tanh_margin"),
    ]
    return models


def run_experiment(
    data_root: str,
    max_samples: int,
    sample_seed: int,
    doses_fit: Iterable[float],
    doses_eval: Iterable[float],
    lambdas: Iterable[float],
    kernel_type: str,
    gamma: Optional[float],
) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    samples = load_real_imagefolder_samples(data_root=data_root, max_samples=max_samples, seed=sample_seed)
    backend_data = "imagefolder"
    if samples is None:
        samples = build_synthetic_samples(max_samples=max_samples, seed=sample_seed)
        backend_data = "synthetic"

    models = None
    backend_model = "fallback_tiny"
    if backend_data == "imagefolder":
        models = load_real_models(device=device)
        backend_model = "torchvision"
    if models is None:
        models = build_fallback_models(device=device)
        backend_model = "fallback_tiny"

    fit_samples, eval_samples, fit_ids, eval_ids = split_fit_eval(samples=samples, seed=sample_seed, fit_frac=0.5)
    fit_size, eval_size = assert_honesty_split(fit_ids=fit_ids, eval_ids=eval_ids)

    ref_model = None
    for m in models:
        if "ResNet50" in m.name and "Robust" not in m.name:
            ref_model = m.model
            break
    if ref_model is None:
        ref_model = models[0].model

    adv = AdversarialFGSMIntervention(ref_model=ref_model, device=device)
    lambda_list = [float(v) for v in lambdas]
    fit_dose_list = [float(v) for v in doses_fit]
    eval_dose_list = [float(v) for v in doses_eval]

    rows = []
    for t_idx, target in enumerate(models):
        peers = [m for i, m in enumerate(models) if i != t_idx]

        y_fit_list = []
        Y_fit_list = []
        for sample in fit_samples:
            for eps in fit_dose_list:
                adv_sample = adv.apply(sample, epsilon=float(eps))
                y_t = target._forward(adv_sample)
                y_p = [p._forward(adv_sample) for p in peers]
                y_fit_list.append(float(y_t))
                Y_fit_list.append([float(v) for v in y_p])
        y_fit = np.asarray(y_fit_list, dtype=float)
        Y_p_fit = np.asarray(Y_fit_list, dtype=float)

        _, w_hat = DISCOSolver.solve_weights_and_distance(y_fit.reshape(-1, 1), Y_p_fit)
        w_hat = np.asarray(w_hat, dtype=float).reshape(-1)

        for eps in eval_dose_list:
            y_eval_list = []
            Y_eval_list = []
            for sample in eval_samples:
                adv_sample = adv.apply(sample, epsilon=float(eps))
                y_t = target._forward(adv_sample)
                y_p = [p._forward(adv_sample) for p in peers]
                y_eval_list.append(float(y_t))
                Y_eval_list.append([float(v) for v in y_p])
            y_eval = np.asarray(y_eval_list, dtype=float)
            Y_p_eval = np.asarray(Y_eval_list, dtype=float)

            y_mix = Y_p_eval @ w_hat
            convex_pier = float(np.mean(np.abs(y_eval - y_mix)))

            X_fit_k, X_eval_k = standardize_by_fit(Y_p_fit, Y_p_eval)
            gamma_used = median_heuristic_gamma(X_fit_k) if gamma is None else float(gamma)
            k_res = solve_kernel_residuals(
                y_fit=y_fit,
                Y_p_fit=X_fit_k,
                y_eval=y_eval,
                Y_p_eval=X_eval_k,
                lambdas=lambda_list,
                kernel_type={"name": kernel_type, "gamma": gamma_used},
            )

            for lam in lambda_list:
                ker_pier = float(np.mean(np.abs(k_res.residuals_by_lambda[float(lam)])))
                rows.append(
                    {
                        "target_model": target.name,
                        "target_group": "Robust" if "Robust" in target.name else "Standard",
                        "dose_epsilon": float(eps),
                        "lambda": float(lam),
                        "budget": float(1.0 / float(lam)),
                        "convex_pier": convex_pier,
                        "kernel_pier": ker_pier,
                        "pier_drop": float(convex_pier - ker_pier),
                        "pier_drop_ratio": float((convex_pier - ker_pier) / max(1e-12, convex_pier)),
                        "n_fit": int(len(fit_samples)),
                        "n_eval": int(len(eval_samples)),
                        "honesty_fit_size": int(fit_size),
                        "honesty_eval_size": int(eval_size),
                        "n_models_total": int(len(models)),
                        "n_peers": int(len(peers)),
                        "kernel_type": kernel_type,
                        "gamma": float(gamma_used),
                        "data_backend": backend_data,
                        "model_backend": backend_model,
                    }
                )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Exp3b: Image ecosystem kernel audit under FGSM doses")
    parser.add_argument("--data-root", type=str, default="./data/imagenet/val")
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--doses-fit", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.04])
    parser.add_argument("--doses-eval", type=float, nargs="+", default=[0.0, 0.02, 0.04, 0.06, 0.08, 0.1])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
    parser.add_argument("--kernel-type", type=str, default="rbf")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument(
        "--out-csv",
        type=str,
        default="results/tables/exp3b_imagenet_kernel.csv",
    )
    args = parser.parse_args()

    df = run_experiment(
        data_root=args.data_root,
        max_samples=args.max_samples,
        sample_seed=args.sample_seed,
        doses_fit=args.doses_fit,
        doses_eval=args.doses_eval,
        lambdas=args.lambdas,
        kernel_type=args.kernel_type,
        gamma=args.gamma,
    )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    summary = (
        df.groupby(["target_model", "target_group", "dose_epsilon", "lambda"], as_index=False)[
            ["convex_pier", "kernel_pier", "pier_drop"]
        ]
        .mean()
        .sort_values(["target_group", "target_model", "dose_epsilon", "lambda"])
    )
    print(summary.head(40))
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
