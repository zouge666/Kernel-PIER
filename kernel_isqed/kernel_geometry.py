from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

KernelSpec = Union[str, Mapping[str, Any]]


@dataclass(frozen=True)
class KernelDISCOResult:
    lambdas: np.ndarray
    residuals_by_lambda: Dict[float, np.ndarray]
    mean_abs_residual_by_lambda: Dict[float, float]
    predictions_by_lambda: Dict[float, np.ndarray]
    best_lambda: float
    best_residuals: np.ndarray
    best_predictions: np.ndarray


class KernelDISCOSolver:
    @staticmethod
    def _as_vector(y: np.ndarray, name: str) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if y_arr.size == 0:
            raise ValueError(f"{name} must be non-empty.")
        return y_arr

    @staticmethod
    def _as_matrix(x: np.ndarray, name: str) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        if x_arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D array-like object.")
        if x_arr.shape[0] == 0 or x_arr.shape[1] == 0:
            raise ValueError(f"{name} must be non-empty.")
        return x_arr

    @staticmethod
    def _normalize_kernel_spec(
        kernel_type: KernelSpec,
        kernel_params: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        if isinstance(kernel_type, str):
            kernel_name = kernel_type
            params = dict(kernel_params or {})
            return kernel_name, params

        if isinstance(kernel_type, Mapping):
            kernel_name = str(kernel_type.get("name", "rbf"))
            params = {k: v for k, v in kernel_type.items() if k != "name"}
            if kernel_params:
                params.update(dict(kernel_params))
            return kernel_name, params

        raise ValueError("kernel_type must be a string or a mapping.")

    @staticmethod
    def solve_kernel_residuals(
        y_fit: np.ndarray,
        Y_p_fit: np.ndarray,
        y_eval: np.ndarray,
        Y_p_eval: np.ndarray,
        lambdas: Iterable[float],
        kernel_type: KernelSpec = "rbf",
        kernel_params: Optional[Mapping[str, Any]] = None,
        approximation_mode: str = "exact",
        n_components: Optional[int] = None,
        random_state: Optional[int] = 0,
    ) -> KernelDISCOResult:
        y_fit_vec = KernelDISCOSolver._as_vector(y_fit, "y_fit")
        y_eval_vec = KernelDISCOSolver._as_vector(y_eval, "y_eval")
        x_fit = KernelDISCOSolver._as_matrix(Y_p_fit, "Y_p_fit")
        x_eval = KernelDISCOSolver._as_matrix(Y_p_eval, "Y_p_eval")

        if x_fit.shape[0] != y_fit_vec.shape[0]:
            raise ValueError(
                f"Row mismatch: Y_p_fit has {x_fit.shape[0]} rows but y_fit has {y_fit_vec.shape[0]} values."
            )
        if x_eval.shape[0] != y_eval_vec.shape[0]:
            raise ValueError(
                f"Row mismatch: Y_p_eval has {x_eval.shape[0]} rows but y_eval has {y_eval_vec.shape[0]} values."
            )
        if x_fit.shape[1] != x_eval.shape[1]:
            raise ValueError(
                f"Column mismatch: Y_p_fit has {x_fit.shape[1]} features but Y_p_eval has {x_eval.shape[1]}."
            )

        lambda_list = np.asarray(list(lambdas), dtype=float)
        if lambda_list.size == 0:
            raise ValueError("lambdas must contain at least one value.")
        if np.any(lambda_list <= 0):
            raise ValueError("All lambda values must be strictly positive.")

        kernel_name, params = KernelDISCOSolver._normalize_kernel_spec(kernel_type, kernel_params)

        residuals_by_lambda: Dict[float, np.ndarray] = {}
        mean_abs_residual_by_lambda: Dict[float, float] = {}
        predictions_by_lambda: Dict[float, np.ndarray] = {}

        for lam in lambda_list:
            lam_f = float(lam)
            if approximation_mode == "exact":
                model = KernelRidge(alpha=lam_f, kernel=kernel_name, **params)
                model.fit(x_fit, y_fit_vec)
                pred_eval = np.asarray(model.predict(x_eval), dtype=float).reshape(-1)
            elif approximation_mode == "nystrom":
                comp = int(n_components) if n_components is not None else int(min(512, x_fit.shape[0]))
                mapper = Nystroem(kernel=kernel_name, n_components=comp, random_state=random_state, **params)
                z_fit = mapper.fit_transform(x_fit)
                z_eval = mapper.transform(x_eval)
                model = Ridge(alpha=lam_f, fit_intercept=False)
                model.fit(z_fit, y_fit_vec)
                pred_eval = np.asarray(model.predict(z_eval), dtype=float).reshape(-1)
            elif approximation_mode == "rff":
                if kernel_name != "rbf":
                    raise ValueError("rff mode only supports rbf kernel.")
                comp = int(n_components) if n_components is not None else int(min(1024, x_fit.shape[1] * 64))
                gamma = float(params.get("gamma", 1.0 / max(1, x_fit.shape[1])))
                mapper = RBFSampler(gamma=gamma, n_components=comp, random_state=random_state)
                z_fit = mapper.fit_transform(x_fit)
                z_eval = mapper.transform(x_eval)
                model = Ridge(alpha=lam_f, fit_intercept=False)
                model.fit(z_fit, y_fit_vec)
                pred_eval = np.asarray(model.predict(z_eval), dtype=float).reshape(-1)
            else:
                raise ValueError("approximation_mode must be one of: exact, nystrom, rff.")
            residual_eval = y_eval_vec - pred_eval

            residuals_by_lambda[lam_f] = residual_eval
            predictions_by_lambda[lam_f] = pred_eval
            mean_abs_residual_by_lambda[lam_f] = float(np.mean(np.abs(residual_eval)))

        best_lambda = min(mean_abs_residual_by_lambda, key=mean_abs_residual_by_lambda.get)
        best_residuals = residuals_by_lambda[best_lambda]
        best_predictions = predictions_by_lambda[best_lambda]

        return KernelDISCOResult(
            lambdas=lambda_list,
            residuals_by_lambda=residuals_by_lambda,
            mean_abs_residual_by_lambda=mean_abs_residual_by_lambda,
            predictions_by_lambda=predictions_by_lambda,
            best_lambda=float(best_lambda),
            best_residuals=best_residuals,
            best_predictions=best_predictions,
        )


def solve_kernel_residuals(
    y_fit: np.ndarray,
    Y_p_fit: np.ndarray,
    y_eval: np.ndarray,
    Y_p_eval: np.ndarray,
    lambdas: Iterable[float],
    kernel_type: KernelSpec = "rbf",
    approximation_mode: str = "exact",
    n_components: Optional[int] = None,
    random_state: Optional[int] = 0,
) -> KernelDISCOResult:
    return KernelDISCOSolver.solve_kernel_residuals(
        y_fit=y_fit,
        Y_p_fit=Y_p_fit,
        y_eval=y_eval,
        Y_p_eval=Y_p_eval,
        lambdas=lambdas,
        kernel_type=kernel_type,
        approximation_mode=approximation_mode,
        n_components=n_components,
        random_state=random_state,
    )
