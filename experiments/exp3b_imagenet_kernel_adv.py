from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
KERNEL_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(KERNEL_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(KERNEL_PROJECT_DIR))

from isqed.geometry import DISCOSolver
from isqed.real_world import AdversarialFGSMIntervention, ImageModelWrapper
from kernel_isqed import solve_kernel_residuals


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_model_provenance(
    wrapper: ImageModelWrapper,
    architecture: str,
    weight_source: str,
    checkpoint_sha256: str,
    robust_training: bool,
) -> ImageModelWrapper:
    wrapper.architecture = architecture
    wrapper.weight_source = weight_source
    wrapper.checkpoint_sha256 = checkpoint_sha256
    wrapper.pretrained = True
    wrapper.robust_training = bool(robust_training)
    return wrapper


def model_provenance(wrapper: ImageModelWrapper) -> dict[str, object]:
    return {
        "architecture": str(wrapper.architecture),
        "weight_source": str(wrapper.weight_source),
        "checkpoint_sha256": str(wrapper.checkpoint_sha256),
        "pretrained": bool(wrapper.pretrained),
        "robust_training": bool(wrapper.robust_training),
    }


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Robust checkpoint must contain a state dictionary.")
    current = checkpoint
    for key in ("state_dict", "model"):
        value = current.get(key)
        if isinstance(value, dict):
            current = value
            break
    if not current or not all(isinstance(key, str) for key in current):
        raise ValueError("Robust checkpoint state dictionary is empty or malformed.")
    return current


def compatible_state_dict(
    checkpoint_state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    candidates = [checkpoint_state]
    for prefix in (
        "module.",
        "model.",
        "module.model.",
        "attacker.model.",
        "module.attacker.model.",
    ):
        candidate = {
            key[len(prefix):]: value
            for key, value in checkpoint_state.items()
            if key.startswith(prefix)
        }
        if candidate:
            candidates.append(candidate)
    expected = set(model_state)
    for candidate in candidates:
        filtered = {key: value for key, value in candidate.items() if key in model_state}
        if set(filtered) != expected:
            continue
        if all(tuple(filtered[key].shape) == tuple(model_state[key].shape) for key in expected):
            return filtered
    missing_best = min(
        (expected - set(candidate) for candidate in candidates),
        key=len,
    )
    raise ValueError(
        f"Robust checkpoint is not a complete torchvision ResNet-50 state dict; missing {len(missing_best)} keys."
    )


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


def split_fit_eval(
    samples: list[tuple[torch.Tensor, int]],
    seed: int,
    fit_frac: float = 0.5,
) -> tuple[
    list[tuple[torch.Tensor, int]],
    list[tuple[torch.Tensor, int]],
    np.ndarray,
    np.ndarray,
]:
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


def load_real_imagefolder_samples(
    data_root: str,
    max_samples: int,
    seed: int,
    expected_class_count: int = 1000,
):
    try:
        import torchvision.datasets as datasets
        import torchvision.transforms as T
    except Exception as exc:
        raise RuntimeError("torchvision is required for the vision audit.") from exc
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ImageFolder root does not exist: {root}")
    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(root=str(root), transform=transform)
    n = len(dataset)
    if n == 0:
        raise ValueError(f"ImageFolder is empty: {root}")
    if len(dataset.classes) != int(expected_class_count):
        raise ValueError(
            f"Expected {expected_class_count} ImageNet classes, found {len(dataset.classes)} in {root}."
        )
    rng = np.random.RandomState(seed)
    if max_samples < n:
        ids = rng.choice(n, size=max_samples, replace=False)
    else:
        ids = np.arange(n)
    samples = []
    for i in ids:
        x, y = dataset[int(i)]
        samples.append((x, int(y)))
    class_index_json = json.dumps(dataset.class_to_idx, ensure_ascii=True, sort_keys=True)
    metadata = {
        "data_root": str(root),
        "dataset_class_count": int(len(dataset.classes)),
        "dataset_class_index_sha256": hashlib.sha256(class_index_json.encode("utf-8")).hexdigest(),
    }
    return samples, metadata


def load_robust_resnet50(device: str, robust_checkpoint: str) -> ImageModelWrapper:
    try:
        import torchvision.models as tv_models
    except Exception as exc:
        raise RuntimeError("torchvision is required for the vision audit.") from exc
    checkpoint_path = Path(robust_checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Robust ResNet-50 checkpoint does not exist: {checkpoint_path}")
    model = tv_models.resnet50(weights=None)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = extract_state_dict(checkpoint)
    state = compatible_state_dict(state, model.state_dict())
    model.load_state_dict(state, strict=True)
    wrapper = ImageModelWrapper(model, "RobustResNet50", device, mode="tanh_margin")
    return set_model_provenance(
        wrapper=wrapper,
        architecture="torchvision.models.resnet50",
        weight_source=str(checkpoint_path),
        checkpoint_sha256=sha256_file(checkpoint_path),
        robust_training=True,
    )


def load_real_models(device: str, robust_checkpoint: str):
    try:
        import torchvision.models as tv_models
    except Exception as exc:
        raise RuntimeError("torchvision is required for the vision audit.") from exc
    models: list[ImageModelWrapper] = []
    loaders = [
        (
            "ResNet18",
            "torchvision.models.resnet18",
            "torchvision:ResNet18_Weights.IMAGENET1K_V1",
            lambda: tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1),
        ),
        (
            "ResNet50",
            "torchvision.models.resnet50",
            "torchvision:ResNet50_Weights.IMAGENET1K_V2",
            lambda: tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2),
        ),
        (
            "MobileNetV3Small",
            "torchvision.models.mobilenet_v3_small",
            "torchvision:MobileNet_V3_Small_Weights.IMAGENET1K_V1",
            lambda: tv_models.mobilenet_v3_small(
                weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            ),
        ),
        (
            "EfficientNetB0",
            "torchvision.models.efficientnet_b0",
            "torchvision:EfficientNet_B0_Weights.IMAGENET1K_V1",
            lambda: tv_models.efficientnet_b0(
                weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
            ),
        ),
        (
            "ConvNeXtTiny",
            "torchvision.models.convnext_tiny",
            "torchvision:ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
            lambda: tv_models.convnext_tiny(
                weights=tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            ),
        ),
    ]
    for name, architecture, weight_source, loader in loaders:
        wrapper = ImageModelWrapper(loader(), name, device, mode="tanh_margin")
        models.append(
            set_model_provenance(
                wrapper=wrapper,
                architecture=architecture,
                weight_source=weight_source,
                checkpoint_sha256="torchvision-managed",
                robust_training=False,
            )
        )
    models.append(load_robust_resnet50(device=device, robust_checkpoint=robust_checkpoint))
    return models


def run_experiment(
    data_root: str,
    robust_checkpoint: str,
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

    samples, data_metadata = load_real_imagefolder_samples(
        data_root=data_root,
        max_samples=max_samples,
        seed=sample_seed,
    )
    backend_data = "imagefolder"
    models = load_real_models(device=device, robust_checkpoint=robust_checkpoint)
    backend_model = "torchvision"

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
        target_metadata = model_provenance(target)
        peer_weight_sources = "|".join(str(model_provenance(peer)["weight_source"]) for peer in peers)
        peer_checkpoint_sha256s = "|".join(
            str(model_provenance(peer)["checkpoint_sha256"]) for peer in peers
        )

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
                        "target_group": "Robust" if target_metadata["robust_training"] else "Standard",
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
                        "target_architecture": target_metadata["architecture"],
                        "target_weight_source": target_metadata["weight_source"],
                        "target_checkpoint_sha256": target_metadata["checkpoint_sha256"],
                        "target_pretrained": int(bool(target_metadata["pretrained"])),
                        "target_robust_training": int(bool(target_metadata["robust_training"])),
                        "peer_weight_sources": peer_weight_sources,
                        "peer_checkpoint_sha256s": peer_checkpoint_sha256s,
                        "data_root": data_metadata["data_root"],
                        "dataset_class_count": data_metadata["dataset_class_count"],
                        "dataset_class_index_sha256": data_metadata["dataset_class_index_sha256"],
                    }
                )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Exp3b: Image ecosystem kernel audit under FGSM doses")
    parser.add_argument("--data-root", type=str, default="./data/imagenet/val")
    parser.add_argument("--robust-checkpoint", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=500)
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
        robust_checkpoint=args.robust_checkpoint,
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
