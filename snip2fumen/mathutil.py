"""
Math util
"""
from typing import Tuple, List

import numpy as np


class MathUtil:
    """
    Static geometric math util
    """
    @staticmethod
    def polar_to_cartesian(rho: float, theta: float) -> Tuple[float, float]:
        """
        convert to cartesian coordinates
        """
        return rho * np.cos(theta), rho * np.sin(theta)

    @staticmethod
    def cartesian_to_polar(x, y) -> Tuple[float, float]:
        """
        convert to polar coordinates
        """
        return np.sqrt(x**2 + y**2), np.arctan2(y, x)

    @staticmethod
    def intersect(line1: List[Tuple[float, float]], line2: List[Tuple[float, float]]) -> Tuple[int, int]:
        """
        Compute intersection point of two lines in polar coordinates

        From : https://stackoverflow.com/a/46572063

        :return: intersection point (x,y coordinates)
        """
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        matrix = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x_0, y_0 = np.linalg.solve(matrix, b)
        x_0, y_0 = int(np.round(x_0)), int(np.round(y_0))
        return x_0, y_0
