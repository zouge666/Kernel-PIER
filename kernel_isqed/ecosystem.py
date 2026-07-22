from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from isqed.core import Intervention, ModelUnit


class Ecosystem:
    def __init__(self, target: ModelUnit, peers: Iterable[ModelUnit]):
        self.target = target
        self.peers = list(peers)

    def batched_query(self, X: Iterable[Any], Thetas, intervention: Intervention, seeds=None):
        inputs = list(X)
        n = len(inputs)
        if isinstance(Thetas, (int, float)):
            theta_list = [float(Thetas)] * n
        else:
            theta_list = list(Thetas)
            if len(theta_list) != n:
                raise ValueError("Length of Thetas must match length of X.")

        if seeds is None:
            seed_list = [None] * n
        elif isinstance(seeds, (int, float)):
            seed_list = [int(seeds)] * n
        else:
            seed_list = list(seeds)
            if len(seed_list) != n:
                raise ValueError("Length of seeds must match length of X.")

        y_target = []
        y_peers = []
        for x, theta, seed in zip(inputs, theta_list, seed_list):
            try:
                perturbed_x = intervention.apply(x, theta, seed)
            except TypeError:
                perturbed_x = intervention.apply(x, theta)

            raw_target = self.target._forward(perturbed_x)
            target_scalarizer = getattr(self.target, "scalarizer", None)
            target_value = target_scalarizer(raw_target) if target_scalarizer is not None else raw_target
            peer_values = []
            for peer in self.peers:
                raw_peer = peer._forward(perturbed_x)
                peer_scalarizer = getattr(peer, "scalarizer", None)
                peer_value = peer_scalarizer(raw_peer) if peer_scalarizer is not None else raw_peer
                peer_values.append(float(peer_value))
            y_target.append(float(target_value))
            y_peers.append(peer_values)

        return np.asarray(y_target, dtype=float), np.asarray(y_peers, dtype=float)
