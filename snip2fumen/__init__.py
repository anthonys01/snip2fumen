ACCEPTABLE_DIFF = 3
MIN_HEIGHT = 120
MIN_WIDTH = 230
COLORS_WANTED = True
MAX_LINES_TAKEN = 20
VISUALIZE = False
LOGS = False


def p(*args):
    if LOGS:
        print(*args)
