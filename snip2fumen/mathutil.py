from typing import Tuple

import numpy as np


class MathUtil:
    @staticmethod
    def polar_to_cartesian(rho: float, theta: float) -> Tuple[float, float]:
        return rho * np.cos(theta), rho * np.sin(theta)

    @staticmethod
    def cartesian_to_polar(x, y) -> Tuple[float, float]:
        return np.sqrt(x**2 + y**2), np.arctan2(y, x)

    @staticmethod
    def intersect(line1: Tuple[float, float], line2: Tuple[float, float]) -> Tuple[int, int]:
        """
        Compute intersection point of two lines in polar coordinates

        From : https://stackoverflow.com/a/46572063

        :return: intersection point (x,y coordinates)
        """
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return x0, y0
