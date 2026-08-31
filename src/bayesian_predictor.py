"""Bayesian cycle-length predictor for Girls & Cycles.

This module provides a lightweight Bayesian estimator that returns
both a posterior mean and a credible interval for cycle length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import math


@dataclass
class PredictionResult:
    mean: float
    std: float
    credible_interval: Tuple[float, float]


class BayesianCyclePredictor:
    """Normal-Normal conjugate Bayesian model for cycle length prediction.

    Prior:
        mu ~ Normal(prior_mean, prior_std^2)
    Likelihood:
        x_i ~ Normal(mu, obs_std^2)

    Posterior for mu remains Normal and is updated in closed form.
    """

    def __init__(
        self,
        prior_mean: float = 30.17,
        prior_std: float = 6.88,
        obs_std: float = 3.5,
    ) -> None:
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.obs_std = obs_std

    def update(self, observations: Iterable[float]) -> PredictionResult:
        obs: List[float] = [float(x) for x in observations]
        if len(obs) == 0:
            return self._result(self.prior_mean, self.prior_std)

        n = len(obs)
        sample_mean = sum(obs) / n

        prior_precision = 1.0 / (self.prior_std ** 2)
        likelihood_precision = n / (self.obs_std ** 2)
        posterior_precision = prior_precision + likelihood_precision

        posterior_var = 1.0 / posterior_precision
        posterior_std = math.sqrt(posterior_var)

        posterior_mean = (
            prior_precision * self.prior_mean
            + likelihood_precision * sample_mean
        ) / posterior_precision

        return self._result(posterior_mean, posterior_std)

    def predict_next_cycle(self, observations: Iterable[float]) -> PredictionResult:
        """Alias for update() for semantic readability."""
        return self.update(observations)

    @staticmethod
    def _result(mean: float, std: float) -> PredictionResult:
        # Approx. 95% credible interval for Normal posterior
        low = mean - 1.96 * std
        high = mean + 1.96 * std
        return PredictionResult(mean=mean, std=std, credible_interval=(low, high))


if __name__ == "__main__":
    # Example usage
    historical_cycles = [28, 30, 31, 29, 32]

    model = BayesianCyclePredictor()
    result = model.predict_next_cycle(historical_cycles)

    print(f"Posterior mean: {result.mean:.2f} days")
    print(f"Posterior std: {result.std:.2f} days")
    print(
        f"95% credible interval: "
        f"[{result.credible_interval[0]:.2f}, {result.credible_interval[1]:.2f}] days"
    )
