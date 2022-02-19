"""
cv2 drawing file
"""
import cv2
import numpy as np

from snip2fumen.colors import ColorUtil
from snip2fumen.mathutil import MathUtil


class DrawUtil:
    """
    static drawing util class
    """
    @staticmethod
    def draw_lines(img, lines):
        """
        Draw lines on image
        :param img: image to draw on
        :param lines: lines to draw
        """
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x_0, y_0 = MathUtil.polar_to_cartesian(rho, theta)
            x_1 = int(x_0 + 1000 * (-b))
            y_1 = int(y_0 + 1000 * a)
            x_2 = int(x_0 - 1000 * (-b))
            y_2 = int(y_0 - 1000 * a)

            cv2.line(img, (x_1, y_1), (x_2, y_2), (0, 0, 255), 2)

    @staticmethod
    def draw_lines_from_points(img, points):
        """
        draw grid lines from grid points

        :param img: image to draw on
        :param points: grid points
        """
        for i in range(points.shape[0]):
            cv2.line(img, points[i, 0], points[i, -1], color=(0, 0, 250), thickness=1)

        for j in range(points.shape[1]):
            cv2.line(img, points[0, j], points[-1, j], color=(0, 0, 250), thickness=1)

    @staticmethod
    def draw_points(img, points):
        """
        draw grid points

        :param img: image to draw on
        :param points: grid points
        """
        for line in points:
            for point in line:
                cv2.circle(img, point, radius=2, color=(0, 255, 0), thickness=-1)

    @staticmethod
    def draw_blocks(img, points, grid):
        """
        Draw grid and colors

        :param img: image to draw on
        :param points: points grid
        :param grid: colors
        """
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                cv2.rectangle(img, points[i, j], points[i + 1, j + 1],
                              color=ColorUtil.hashable(grid[i][j]), thickness=-1)

    @staticmethod
    def show_wait_destroy(winname, img):
        """
        display image in window

        :param winname: window name
        :param img: image to display
        """
        cv2.imshow(winname, img)
        cv2.moveWindow(winname, 500, 0)
        wait_time = 1000
        while cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) >= 1:
            key_code = cv2.waitKey(wait_time)
            if (key_code & 0xFF) == ord("q"):
                cv2.destroyWindow(winname)
                break
