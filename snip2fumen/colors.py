"""
Colors, and color processing
"""
from typing import Union, Tuple, List, Dict

import numpy as np

from snip2fumen import prnt
from snip2fumen.pieces import EMPTY, S_PIECE, Z_PIECE, O_PIECE, J_PIECE, L_PIECE, T_PIECE, I_PIECE, GARBAGE

Color = Union[Tuple[float, float, float], Tuple[int, int, int]]

COLORS = {
    EMPTY: "empty",
    GARBAGE: "garbage",
    S_PIECE: "S",
    Z_PIECE: "Z",
    O_PIECE: "O",
    J_PIECE: "J",
    L_PIECE: "L",
    T_PIECE: "T",
    I_PIECE: "I"
}


class ColorUtil:
    """
    color static class util
    """
    @staticmethod
    def bgr_to_ycc(color_bgr: Color) -> Color:
        """
        convert BGR to YCC color

        :param color_bgr: BGR color
        :return: YCC color
        """
        blue, green, red = color_bgr
        blue, green, red = blue/255.0, green/255.0, red/255.0
        y = .299 * red + .587 * green + .114 * blue
        c_b = 128 - .168736 * red - .331364 * green + .5 * blue
        c_r = 128 + .5 * red - .418688 * green - .081312 * blue
        return y, c_b, c_r

    @staticmethod
    def color_dist(c_1: Color, c_2: Color) -> float:
        """
        Find the euclidean distance of two BGR colors in YUV color space

        Code from : https://stackoverflow.com/a/21886236

        :param c_1: bgr color
        :param c_2: bgr color
        :return: euclidean distance
        """
        return sum((a - b) ** 2 for a, b in zip(ColorUtil.bgr_to_ycc(c_1), ColorUtil.bgr_to_ycc(c_2)))

    @staticmethod
    def hashable(color_array: Union[np.ndarray[3, float], Color]) -> Color:
        """
        Ensure to have a hashable color tuple (mostly to silently convert numpy color array)
        """
        return int(color_array[0]), int(color_array[1]), int(color_array[2])

    @staticmethod
    def _min_zero_row(zero_mat: np.ndarray[(int, int), bool], mark_zero: List[Tuple[int, int]]):
        # Find the row
        min_row = [99999, -1]

        for row_num in range(zero_mat.shape[0]):
            if 0 < np.sum(zero_mat[row_num]) < min_row[0]:
                min_row = [np.sum(zero_mat[row_num]), row_num]

        # Marked the specific row and column as False
        zero_index = np.where(zero_mat[min_row[1]])[0][0]
        mark_zero.append((min_row[1], zero_index))
        zero_mat[min_row[1], :] = False
        zero_mat[:, zero_index] = False

    # pylint: disable=[too-many-locals,too-many-branches]
    @staticmethod
    def _mark_matrix(mat: np.ndarray[(int, int)], valid_rows: int):
        # Transform the matrix to boolean matrix(0 = True, others = False)
        zero_bool_mat: np.ndarray[(int, int), bool] = (mat == 0)
        zero_bool_mat_copy: np.ndarray[(int, int), bool] = zero_bool_mat.copy()

        # Recording possible answer positions by marked_zero
        marked_zero = []
        while True in zero_bool_mat_copy:
            ColorUtil._min_zero_row(zero_bool_mat_copy, marked_zero)

        # Recording the row and column indexes separately.
        marked_zero_row: List[int] = []
        marked_zero_col: List[int] = []
        for marked in marked_zero:
            marked_zero_row.append(marked[0])
            marked_zero_col.append(marked[1])
        # step 2-2-1
        non_marked_row = list(set(range(valid_rows)) - set(marked_zero_row))

        marked_cols = []
        check_switch = True
        while check_switch:
            check_switch = False
            for row in non_marked_row:
                row_array = zero_bool_mat[row, :]
                for j in range(row_array.shape[0]):
                    # step 2-2-2
                    if row_array[j] and j not in marked_cols:
                        # step 2-2-3
                        marked_cols.append(j)
                        check_switch = True

            for row_num, col_num in marked_zero:
                # step 2-2-4
                if row_num not in non_marked_row and col_num in marked_cols:
                    # step 2-2-5
                    non_marked_row.append(row_num)
                    check_switch = True
        # step 2-2-6
        marked_rows = list(set(range(valid_rows)) - set(non_marked_row))

        return marked_zero, marked_rows, marked_cols

    @staticmethod
    def _adjust_matrix(mat, cover_rows, cover_cols, valid_rows, valid_cols):
        cur_mat = mat
        non_zero_element = []

        # Step 4-1
        for row in range(valid_rows):
            if row not in cover_rows:
                for i in range(valid_cols):
                    if i not in cover_cols:
                        non_zero_element.append(cur_mat[row][i])

        min_num = min(non_zero_element)

        # Step 4-2
        for row in range(valid_rows):
            if row not in cover_rows:
                for i in range(valid_cols):
                    if i not in cover_cols:
                        cur_mat[row, i] = cur_mat[row, i] - min_num
        # Step 4-3
        for row in cover_rows:
            for col in cover_cols:
                cur_mat[row, col] = cur_mat[row, col] + min_num

        return cur_mat

    # pylint: disable=[too-many-locals,too-many-branches]
    @staticmethod
    def _hungarian_algorithm(color_families: list) -> Dict[Color, Color]:
        """
        Match given colors to the normalized colors

        Based on : https://python.plainenglish.io/hungarian-algorithm-introduction-python-implementation-93e7c0890e15

        :return: mapping from given colors to normalized colors
        """
        colors_list = list(COLORS.keys())

        valid_rows = len(color_families)
        valid_columns = len(colors_list)

        cost_matrix_size = max(valid_rows, valid_columns)
        cost_matrix = np.inf + np.zeros((cost_matrix_size, cost_matrix_size))

        for i in range(valid_rows):
            for j in range(valid_columns):
                cost_matrix[i, j] = ColorUtil.color_dist(color_families[i], colors_list[j])

        # substract rows and columns internal minimum
        for row in range(valid_rows):
            cost_matrix[row] = cost_matrix[row] - np.min(cost_matrix[row])

        for col in range(valid_columns):
            cost_matrix[:, col] = cost_matrix[:, col] - np.min(cost_matrix[:, col])

        zero_count = 0
        ans_pos = []
        stop_condition = min(valid_rows, valid_columns)
        while zero_count < stop_condition:
            # Step 2 & 3
            ans_pos, marked_rows, marked_cols = ColorUtil._mark_matrix(cost_matrix, valid_rows)
            zero_count = len(marked_rows) + len(marked_cols)
            prnt(ans_pos, marked_rows, marked_cols)

            if zero_count < stop_condition:
                cost_matrix = ColorUtil._adjust_matrix(cost_matrix, marked_rows, marked_cols, valid_rows, valid_columns)

        mapping = {}
        for color, guess in ans_pos:
            mapping[color_families[color]] = colors_list[guess]

        return mapping

    @staticmethod
    def _fix_misattribution(color_to_guess: Dict[Color, Color]):
        """
        If one color was assigned to GARBAGE, but there is still unused colors, reassign it.

        If distance between the color and its attribution is too big, set it to EMPTY
        """
        assigned_to_garbage = None
        color_pieces = [I_PIECE, T_PIECE, S_PIECE, Z_PIECE, J_PIECE, L_PIECE, O_PIECE]
        for color in color_to_guess:
            prnt(f"Dist({color}, {color_to_guess[color]}) = {ColorUtil.color_dist(color, color_to_guess[color])}")
            if color_to_guess[color] == EMPTY:
                continue
            if color_to_guess[color] == GARBAGE:
                assigned_to_garbage = color
            else:
                color_pieces.remove(color_to_guess[color])

        if assigned_to_garbage and color_pieces:
            color_to_guess[assigned_to_garbage] = \
                min(color_pieces, key=lambda c: ColorUtil.color_dist(assigned_to_garbage, c))
        elif assigned_to_garbage:
            dist = ColorUtil.color_dist(assigned_to_garbage, GARBAGE)
            if dist > 0.1:
                color_to_guess[assigned_to_garbage] = EMPTY

    # pylint: disable=[too-many-locals,too-many-branches]
    @staticmethod
    def map_colors(grid: np.ndarray[(int, int), Color]):
        """
        Fill color grid with normalized block colors (that translate into board state)

        Color comparisons are done in YUV color space.

        First regroup all colors in the grid into several color "families".

        Then, map these families to the block colors (all families with no association is mapped to EMPTY by default)

        Finally, map the old colors to the new in the grid.
        """
        color_families = {}
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                color = ColorUtil.hashable(grid[i, j])
                if not color_families:
                    color_families[color] = [color]
                else:
                    found = False
                    for family, members in color_families.items():
                        if ColorUtil.color_dist(color, family) < 1e-2:
                            members.append(color)
                            found = True
                            break
                    if not found:
                        color_families[color] = [color]

        prnt(f"Color families found : {color_families.keys()}")

        families = list(color_families.keys())

        map_color_to_guess = ColorUtil._hungarian_algorithm(families)
        ColorUtil._fix_misattribution(map_color_to_guess)

        for found_family in map_color_to_guess:
            families.remove(found_family)

        for left in families:
            prnt(f"Family {left} was not matched anything defaulting to EMPTY")
            map_color_to_guess[left] = EMPTY

        for family, members in color_families.items():
            guess = map_color_to_guess[family]
            for col in members:
                map_color_to_guess[col] = guess

        prnt(f"Final color mapping : {map_color_to_guess}")

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                grid[i, j] = map_color_to_guess[ColorUtil.hashable(grid[i, j])]
