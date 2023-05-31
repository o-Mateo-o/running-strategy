import numpy as np


class ExtendedKellerApproxModel:
    """Approximated formulas of the extended Keller model."""

    def short(D: float, tau: float, F: float) -> float:
        """Calculate a record time on a "short" distance and the given physiological.

        Args:
            D (float): Distance.
            tau (float): Time constant
            F (float): Maximal propulsive force per unit mass.

        Returns:
            float: A record time.
        """
        return D / F / tau + tau

    def mid(D: float, E0: float, sigma: float, tau: float, F: float) -> float:
        """Calculate a record time on a "mid" (actually - long) distance and the given physiological.

        Args:
            D (float): Distance.
            E0 (float): Initial energy per unit mass.
            sigma (float): Energy supply rate per unit mass.
            tau (float): Time constant
            F (float): Maximal propulsive force per unit mass.

        Returns:
            float: A record time.
        """
        return (np.sqrt(E0**2 + 4 * sigma * D**2 / tau) - E0) / sigma / 2

    def long(
        D: float, gamma: float, E0: float, sigma: float, tau: float, F: float
    ) -> float:
        """Calculate a record time on a "long" (actually - very long) distance and the given physiological.

        Args:
            D (float): Distance.
            gamma (float): Fatigue factor.
            E0 (float): Initial energy per unit mass.
            sigma (float): Energy supply rate per unit mass.
            tau (float): Time constant
            F (float): Maximal propulsive force per unit mass.

        Returns:
            float: A record time.
        """
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
