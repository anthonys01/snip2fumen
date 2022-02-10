import cv2
import numpy as np

from snip2fumen.colors import ColorUtil
from snip2fumen.mathutil import MathUtil


class DrawUtil:
    @staticmethod
    def draw_lines(img, lines):
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0, y0 = MathUtil.polar_to_cartesian(rho, theta)
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    @staticmethod
    def draw_lines_from_points(img, points):
        for i in range(points.shape[0]):
            cv2.line(img, points[i, 0], points[i, -1], color=(0, 0, 250), thickness=1)

        for j in range(points.shape[1]):
            cv2.line(img, points[0, j], points[-1, j], color=(0, 0, 250), thickness=1)

    @staticmethod
    def draw_points(img, points):
        for l in points:
            for point in l:
                cv2.circle(img, point, radius=2, color=(0, 255, 0), thickness=-1)

    @staticmethod
    def draw_blocks(img, points, grid):
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                cv2.rectangle(img, points[i, j], points[i + 1, j + 1],
                              color=ColorUtil.hashable(grid[i][j]), thickness=-1)

    @staticmethod
    def show_wait_destroy(winname, img):
        cv2.imshow(winname, img)
        cv2.moveWindow(winname, 500, 0)
        wait_time = 1000
        while cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE) >= 1:
            keyCode = cv2.waitKey(wait_time)
            if (keyCode & 0xFF) == ord("q"):
                cv2.destroyWindow(winname)
                break
