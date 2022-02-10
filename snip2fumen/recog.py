import sys
from typing import List, Tuple, Union

import cv2
import numpy as np
from math import isclose
from collections import defaultdict
import pyperclip
from snip2fumen import MIN_HEIGHT, MIN_WIDTH, p, VISUALIZE, ACCEPTABLE_DIFF, MAX_LINES_TAKEN
from snip2fumen.pieces import *
import snip2fumen.mini_fumen as mini_fumen
from snip2fumen.colors import ColorUtil, Color
from snip2fumen.draw import DrawUtil
from snip2fumen.mathutil import MathUtil


Line = Tuple[float, float]


class FumenEncoder:
    @staticmethod
    def to_fumen_block(color: Color):
        if color == EMPTY:
            return mini_fumen.EMPTY
        if color == GARBAGE:
            return mini_fumen.GRAY
        if color == L_PIECE:
            return mini_fumen.L
        if color == J_PIECE:
            return mini_fumen.J
        if color == S_PIECE:
            return mini_fumen.S
        if color == Z_PIECE:
            return mini_fumen.Z
        if color == I_PIECE:
            return mini_fumen.I
        if color == T_PIECE:
            return mini_fumen.T
        if color == O_PIECE:
            return mini_fumen.O
        return mini_fumen.EMPTY

    @staticmethod
    def to_fumen(grid: np.ndarray[(int, int), Color]) -> str:
        """
        Generate the fumen url associated with given normalized board
        (also put the generated link into clipboard)
        """
        field = mini_fumen.Field()

        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                field.set_number_field_at(col,  grid.shape[0] - row - 1,
                                          FumenEncoder.to_fumen_block(grid[row, col]))

        fumen_url = "http://fumen.zui.jp/?" + mini_fumen.encode(field)
        print(fumen_url)
        pyperclip.copy(fumen_url)
        return fumen_url


class BoardRecognizer:

    def __init__(self):
        self.size_hint: int = 0
        self.MIN_HEIGHT = MIN_HEIGHT
        self.MIN_WIDTH = MIN_WIDTH
        self.VISUALIZE = VISUALIZE
        self.ACCEPTABLE_DIFF = ACCEPTABLE_DIFF
        self.MAX_LINES_TAKEN = MAX_LINES_TAKEN

    def recognize_file(self, file_path: str) -> np.ndarray[(int, int), Color]:
        return self.recognize(cv2.imread(file_path))

    def recognize(self, img) -> np.ndarray[(int, int), Color]:
        """
        recognize the tetris board from image array

        For Hough lines first processing : https://stackoverflow.com/a/48963987

        :param img: BGR colored array
        :return: block grid
        """
        height = img.shape[0]
        width = img.shape[1]

        print(f"Processing {height}x{width} img...")

        if height < self.MIN_HEIGHT or width < self.MIN_WIDTH:
            print("Image too small. The lowest resolution accepted is 120x250.")
            sys.exit()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 20, 20, apertureSize=3)

        horizontal = np.copy(edges)
        vertical = np.copy(edges)

        p("\n\nProcessing vertical lines...")

        s = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 30))
        vertical = cv2.erode(vertical, s)
        vertical = cv2.dilate(vertical, s)
        lines_ver: Union[np.ndarray[List[Line]], List[List[Line]]] = \
            cv2.HoughLines(vertical, 1, np.pi / 180, height // 4)
        if lines_ver is not None and lines_ver.any():
            p('number of vertical Hough lines:', len(lines_ver))
            lines_ver = self.filter_lines(lines_ver)
            if self.VISUALIZE:
                img_tmp = np.copy(img)
                DrawUtil.draw_lines(img_tmp, lines_ver)
                DrawUtil.show_wait_destroy("Verticals 1", img_tmp)
            lines_ver = self.filter_verticals(lines_ver)
            # should be sorted now
            lines_ver = list(reversed(lines_ver))
        else:
            lines_ver = []

        p("\n\nProcessing horizontal lines...")

        s = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 30, 1))
        horizontal = cv2.erode(horizontal, s)
        horizontal = cv2.dilate(horizontal, s)
        lines_hor: Union[np.ndarray[List[Line]], List[List[Line]]] = \
            cv2.HoughLines(horizontal, 1, np.pi / 180, width // 4)
        if lines_hor is not None and lines_hor.any():
            p('number of horizontal Hough lines:', len(lines_hor))
            lines_hor = self.filter_lines(lines_hor)
            if self.VISUALIZE:
                img_tmp = np.copy(img)
                DrawUtil.draw_lines(img_tmp, lines_hor)
                DrawUtil.show_wait_destroy("Horizontals 1", img_tmp)
            lines_hor = self.filter_horizontals(lines_hor)
            # should be sorted now
            lines_hor = list(reversed(lines_hor))
        else:
            lines_hor = []

        points, _ = np.meshgrid([None] * len(lines_ver), [None] * len(lines_hor))
        grid, _ = np.meshgrid([None] * (len(lines_ver) - 1), [None] * (len(lines_hor) - 1))

        for i in range(len(lines_hor)):
            for j in range(len(lines_ver)):
                points[i, j] = MathUtil.intersect(lines_hor[i], lines_ver[j])

        # get grid colors from block middle points
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                x1, y1 = points[i, j]
                x2, y2 = points[i + 1, j + 1]
                x, y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                grid[i, j] = img[y, x]

        lines = lines_hor + lines_ver
        p('\n\nTotal number of Hough lines:', len(lines))

        if self.VISUALIZE:
            DrawUtil.draw_lines(img, lines)
            DrawUtil.draw_points(img, points)
            DrawUtil.show_wait_destroy("Found lines", img)

            res = np.zeros(img.shape, np.uint8)
            DrawUtil.draw_blocks(res, points, grid)
            DrawUtil.draw_lines_from_points(res, points)

            DrawUtil.show_wait_destroy("Grid before mapping", res)

        ColorUtil.map_colors(grid)

        if self.VISUALIZE:
            res = np.zeros(img.shape, np.uint8)
            DrawUtil.draw_blocks(res, points, grid)
            DrawUtil.draw_lines_from_points(res, points)

            DrawUtil.show_wait_destroy("Result", res)
        return grid

    def filter_lines(self, lines: List[List[Line]]) -> List[List[Line]]:
        rho_threshold = 15
        theta_threshold = 0.5
    
        # how many lines are similar to a given one
        similar_lines = {i: [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue
    
                rho_i, theta_i = lines[i][0]
                rho_j, theta_j = lines[j][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    similar_lines[i].append(j)
    
        # ordering the INDECES of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x: len(similar_lines[x]))
    
        # line flags is the base for the filtering
        line_flags = len(lines)*[True]
        for i in range(len(lines) - 1):
            if not line_flags[indices[i]]:
                continue
    
            for j in range(i + 1, len(lines)):
                if not line_flags[indices[j]]:
                    continue
    
                rho_i, theta_i = lines[indices[i]][0]
                rho_j, theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    line_flags[indices[j]] = False
        
        filtered_lines = []
        for i in range(len(lines)):
            if line_flags[i]:
                filtered_lines.append(lines[i])
        
        p('Number of filtered lines:', len(filtered_lines))
        return filtered_lines

    def is_acceptable_line_diff(self, wanted_diff, given_diff) -> bool:
        return np.abs(given_diff - wanted_diff) <= self.ACCEPTABLE_DIFF

    def _find_best_line_diff(self, sorted_lines_desc: List[float], max_line_nb: int) -> int:
        line_diff = [int(sorted_lines_desc[i] - sorted_lines_desc[i + 1]) for i in range(len(sorted_lines_desc) - 1)]
        p(f"Line diff : {line_diff}")
        line_diff_count = defaultdict(lambda: 0)
        for d in line_diff:
            line_diff_count[d] += 1
            line_diff_count[d+1] += 1
            line_diff_count[d-1] += 1
        p(f"Line diff count : {dict(line_diff_count)}")

        # first guess
        max_line_diff = int((sorted_lines_desc[0] - sorted_lines_desc[-1]) // max_line_nb)
        p(f"First guess for grid size : {max_line_diff}")

        if self.size_hint and abs(max_line_diff - self.size_hint) / max_line_diff <= 0.2:
            # good hint
            max_line_diff = self.size_hint

        second_guess = max(line_diff_count, key=line_diff_count.get)

        if abs(max_line_diff - second_guess) / max_line_diff <= 0.2:
            # close enough
            max_line_diff = second_guess
        else:
            close_candidate = None
            for acceptable_delta in range(-self.ACCEPTABLE_DIFF, self.ACCEPTABLE_DIFF + 1):
                candidate = max_line_diff + acceptable_delta
                if candidate in line_diff_count:
                    if not close_candidate or line_diff_count[close_candidate] < line_diff_count[candidate]:
                        close_candidate = candidate
            if close_candidate:
                # found a correct one
                max_line_diff = close_candidate
            else:
                # no good candidate found from board size
                max_line_diff = second_guess

        return max_line_diff

    def _find_best_begin_line(self, sorted_lines: List[float], max_line_diff: int):
        line_diff = [abs(sorted_lines[i] - sorted_lines[i+1]) for i in range(len(sorted_lines)-1)]
        max_diff_cummulative = [0] * len(line_diff)
        c = 0
        i = 0
        while i < len(line_diff):
            j = i
            while j < len(line_diff) - 1 and self.is_acceptable_line_diff(max_line_diff, line_diff[j]):
                for k in range(i, j + 1):
                    max_diff_cummulative[k] += 1
                j += 1

            i = j + 1

        p(f"Successive max diff appearance : {max_diff_cummulative}")
        best_line_start_index = max_diff_cummulative.index(max(max_diff_cummulative))
        best_line_start = sorted_lines[best_line_start_index]
        p(f"Best line start for now : {best_line_start}")

        earliest_line_match = None
        for i in range(best_line_start_index):
            diff = abs(sorted_lines[i] - best_line_start)
            matches = False
            for acceptable_delta in range(-self.ACCEPTABLE_DIFF, self.ACCEPTABLE_DIFF + 1):
                matches |= diff % (max_line_diff + acceptable_delta) in [0, 1, max_line_diff - 1 + acceptable_delta]
            if matches:
                earliest_line_match = sorted_lines[i]
                p(f"New best line start : {sorted_lines[i]}")
                break

        if not earliest_line_match:
            earliest_line_match = best_line_start
        return earliest_line_match

    def _find_line_grid_from_latest(self, sorted_lines_desc, max_line_nb, max_line_diff, first_line):
        new_lines = [first_line]
        current_line = first_line - max_line_diff
        current_existing_line_index = sorted_lines_desc.index(first_line)
        created = 0
        for _ in range(max_line_nb - 1):
            if current_line + self.ACCEPTABLE_DIFF < sorted_lines_desc[-1]:
                break

            while sorted_lines_desc[current_existing_line_index] > current_line + self.ACCEPTABLE_DIFF:
                current_existing_line_index += 1

            distance_to_closest = abs(sorted_lines_desc[current_existing_line_index] - current_line)
            if distance_to_closest <= self.ACCEPTABLE_DIFF:
                current_line = sorted_lines_desc[current_existing_line_index]
                p(f"Closest existing line at {distance_to_closest}, using it {current_line}")
            else:
                created += 1
                p(f"Closest existing line at {distance_to_closest}, creating new line {current_line}")

            new_lines.append(current_line)
            current_line -= max_line_diff
        p(f"Lines from latest {new_lines}")
        return new_lines, created

    def _find_line_grid_from_earliest(self, sorted_lines_asc, max_line_nb, max_line_diff, first_line):
        new_lines = [first_line]
        current_line = first_line + max_line_diff
        current_existing_line_index = sorted_lines_asc.index(first_line)
        created = 0
        for _ in range(max_line_nb - 1):
            if current_line - self.ACCEPTABLE_DIFF > sorted_lines_asc[-1]:
                break

            while sorted_lines_asc[current_existing_line_index] < current_line - self.ACCEPTABLE_DIFF:
                current_existing_line_index += 1

            distance_to_closest = abs(sorted_lines_asc[current_existing_line_index] - current_line)
            if distance_to_closest <= self.ACCEPTABLE_DIFF:
                current_line = sorted_lines_asc[current_existing_line_index]
                p(f"Closest existing line at {distance_to_closest}, using it {current_line}")
            else:
                created += 1
                p(f"Closest existing line at {distance_to_closest}, creating new line {current_line}")

            new_lines.append(current_line)
            current_line += max_line_diff
        p(f"Lines from earliest {new_lines}")
        return list(reversed(new_lines)), created

    def find_line_grid(self, sorted_lines_desc: List[float], max_line_nb: int):
        if len(sorted_lines_desc) < 2:
            p("Not enough lines to compute further")
            return sorted_lines_desc

        max_line_diff = self._find_best_line_diff(sorted_lines_desc, max_line_nb)
        self.size_hint = max_line_diff

        p(f"Most likely grid interval : {max_line_diff}")

        latest_line_match = self._find_best_begin_line(sorted_lines_desc, max_line_diff)
        new_lines, created = self._find_line_grid_from_latest(
            sorted_lines_desc, max_line_nb, max_line_diff, latest_line_match)

        sorted_lines_asc = list(reversed(sorted_lines_desc))
        earliest_line_match = self._find_best_begin_line(sorted_lines_asc, max_line_diff)
        new_lines_2, created_2 = self._find_line_grid_from_earliest(
            sorted_lines_asc, max_line_nb, max_line_diff, earliest_line_match)

        if created_2 < created:
            p("Starting from early is better in this case")
            new_lines = new_lines_2

        p(f"New lines : {new_lines}")
        return new_lines
    
    def filter_verticals(self, lines: List[List[Line]]) -> List[List[Line]]:
        res: List[List[Line]] = []
        # keep only truly vertical lines (theta = 0)
        for line in lines:
            rho, theta = line[0]
            if isclose(theta, 0.0, abs_tol=1e-5):
                res.append(line)
        
        res_sorted = sorted(res, key=lambda l: MathUtil.polar_to_cartesian(l[0][0], l[0][1])[0], reverse=True)
        x_sorted = [MathUtil.polar_to_cartesian(l[0][0], l[0][1])[0] for l in res_sorted]
        p(x_sorted)
        new_lines = self.find_line_grid(x_sorted, 11)
        res = [[MathUtil.cartesian_to_polar(x, 0)] for x in new_lines]
        p('Number of vertical filtered lines:', len(res))
        return res
    
    def filter_horizontals(self, lines: List[List[Line]]) -> List[List[Line]]:
        res: List[List[Line]] = []
        # keep only truly horizontal lines (theta = pi / 2)
        for line in lines:
            rho, theta = line[0]
            if isclose(theta, np.pi / 2, abs_tol=1e-5):
                res.append(line)
        
        res_sorted: List[List[Line]] = \
            sorted(res, key=lambda l: MathUtil.polar_to_cartesian(l[0][0], l[0][1])[1], reverse=True)
        y_sorted = [MathUtil.polar_to_cartesian(l[0][0], l[0][1])[1] for l in res_sorted]
        p(y_sorted)
        new_lines = self.find_line_grid(y_sorted, self.MAX_LINES_TAKEN + 1)
        res = [[MathUtil.cartesian_to_polar(0, y)] for y in new_lines]
        p('Number of horizontal filtered lines:', len(res))
        return res


if __name__ == "__main__":
    ...
