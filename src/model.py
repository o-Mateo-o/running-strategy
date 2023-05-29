import numpy as np


class ExtendedKellerApproxModel:
    def short(D: float, tau: float, F: float) -> float:
        return D / F / tau + tau

    def mid(D: float, E0: float, sigma: float, tau: float, F: float) -> float:
        return (np.sqrt(E0**2 + 4 * sigma * D**2 / tau) - E0) / sigma / 2

    def long(
        D: float, gamma: float, E0: float, sigma: float, tau: float, F: float
    ) -> float:
        return (
            1
            / 2
            / sigma
            / tau
            * (
                np.sqrt((E0 * tau - gamma * D**2) ** 2 + 4 * sigma * tau * D**2)
                - E0 * tau
                + gamma * D**2
            )
        )
