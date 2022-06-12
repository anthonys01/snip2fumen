"""
    Method to adjust the default colors
"""
from snip2fumen.colors import COLORS, ColorUtil
from snip2fumen.recog import BoardRecognizer


img_to_answers = {
    "./img/jstris1.png": {
        (0, 0, 0): 'empty', (1, 177, 89): 'S', (1, 89, 45): 'garbage', (138, 41, 175): 'T', (198, 65, 33): 'J',
        (2, 91, 227): 'L', (2, 159, 227): 'O', (215, 155, 15): 'I', (55, 15, 215): 'Z'
    },
    "./img/jstris2.png": {
        (0, 0, 0): 'empty', (1, 80, 114): 'garbage', (138, 41, 175): 'T', (198, 65, 33): 'J',
        (55, 15, 215): 'Z', (2, 91, 227): 'L', (2, 159, 227): 'O', (1, 177, 89): 'S', (215, 155, 15): 'I'
    },
    "./img/jstris3.png": {
        (0, 0, 0): 'empty', (138, 41, 175): 'T', (2, 91, 227): 'L', (215, 155, 15): 'I', (1, 177, 89): 'S',
        (55, 15, 215): 'Z', (2, 159, 227): 'O', (198, 65, 33): 'J', (153, 153, 153): 'garbage'
    },
    "./img/jstris4.png": {
        (63, 248, 255): 'O', (0, 0, 0): 'empty', (31, 124, 130): 'empty', (20, 0, 255): 'Z',
        (253, 255, 0): 'I', (248, 63, 0): 'J', (249, 56, 255): 'T', (170, 170, 170): 'garbage'
    },
    # "./img/jstris5.png": {
    #     (0, 0, 0): 'empty', (64, 62, 220): 'Z', (33, 31, 110): 'empty', (58, 243, 131): 'S',
    #     (137, 245, 15): 'I', (241, 17, 137): 'T', (229, 152, 35): 'J', (29, 88, 245): 'L', (41, 186, 231): 'O'
    # },
    "./img/jstris6.png": {
        (0, 0, 0): 'empty', (253, 7, 66): 'J', (239, 0, 239): 'T',
        (0, 239, 72): 'S', (4, 139, 242): 'L', (8, 70, 127): 'empty', (160, 160, 160): 'garbage'
    },
    "./img/fourtris1.png": {
        (0, 0, 0): 'empty', (32, 64, 255): 'Z', (32, 128, 255): 'L', (64, 208, 64): 'S', (32, 224, 255): 'O',
        (255, 128, 64): 'J', (255, 208, 0): 'I', (240, 64, 160): 'T', (187, 187, 187): 'garbage'
    },
    "./img/tetrio1.png": {
        (8, 8, 10): 'empty', (58, 51, 178): 'Z', (154, 61, 164): 'T', (50, 99, 179): 'L',
        (50, 153, 179): 'O', (51, 180, 132): 'S', (132, 179, 50): 'I', (168, 65, 82): 'J'
    },
    "./img/tetrio2.png": {
        (12, 9, 12): 'empty', (132, 180, 51): 'I', (58, 51, 178): 'Z', (49, 98, 178): 'L', (50, 153, 179): 'O',
        (51, 180, 132): 'S', (76, 76, 76): 'garbage', (154, 61, 164): 'T', (164, 62, 79): 'J'
    },
    "./img/fourtris2.png": {
        (0, 0, 0): 'empty',
        (32, 64, 255): 'Z',
        (32, 128, 255): 'L',
        (32, 224, 255): 'O',
        (64, 208, 64): 'S',
        (240, 64, 160): 'T',
        (255, 128, 64): 'J',
        (255, 208, 0): 'I'
    }
}


def calc_error(answer, results):
    error = 0
    for color, res in results.items():
        if color in answer:
            error += answer[color] != res
        else:
            print(f'{color=} is not in answer [{res=}]')
    return error


def adjust():
    """
    adjust
    """
    board_recog = BoardRecognizer()
    board_recog.raw_colors = True

    var = list(range(0, 100, 5))
    var = [0]
    delta_var = []

    for i in var:
        errors_by_block = {
            'empty': 0,
            'garbage': 0,
            'Z': 0,
            'L': 0,
            'O': 0,
            'S': 0,
            'T': 0,
            'J': 0,
            'I': 0
        }
        total_errors = 0
        delta_sum = 0
        for img, answer in img_to_answers.items():
            current_color_map = {
                (0, 0, 0): "empty",
                (45, 255, 100): "S",
                (255, 65, 185): "T",
                (255, 100, 65): "J",
                (65, 45, 255): "Z",
                (45, 255, 255): "O",
                (35, 125, 225): "L",
                (180, 180, 15): "I"
            }
            raw_grid = board_recog.recognize_file(img)
            families_map = ColorUtil.extract_color_families(raw_grid)
            families = list(families_map.keys())

            need_garbage = False
            garbage_color = None

            for color in families:
                if color[0] > 0 and color[0] == color[1] == color[2]:
                    need_garbage = True
                    garbage_color = color
                    break

            guesses = ColorUtil.hungarian_algorithm(list(current_color_map.keys()), families)
            delta = sum((ColorUtil.color_dist(color, guess) for color, guess in guesses.items())) / len(guesses)
            results = {}
            for color, guess in guesses.items():
                results[color] = current_color_map[guess]
            print(f'{img} {results=}')
            errors = calc_error(answer, results)

            if need_garbage:
                print('Need garbage')
                guesses[garbage_color] = (120, 120, 120)
                delta = sum((ColorUtil.color_dist(color, guess) for color, guess in guesses.items())) / len(guesses)
                results[garbage_color] = "garbage"
                errors = calc_error(answer, results)
            else:
                current_color_map[(120, 120, 120)] = "garbage"
                guesses_with_garbage = ColorUtil.hungarian_algorithm(list(current_color_map.keys()), families)
                delta_with_garbage = sum((ColorUtil.color_dist(color, guess)
                                          for color, guess in guesses_with_garbage.items())) / len(guesses_with_garbage)
                results_with_garbage = {}
                for color, guess in guesses_with_garbage.items():
                    results_with_garbage[color] = current_color_map[guess]
                print(f'{img} {results_with_garbage=}')
                errors_with_garbage = calc_error(answer, results_with_garbage)
                print(f'{errors_with_garbage=} | {errors=}')
                print(f'{delta_with_garbage=} | {delta=}')
                if delta_with_garbage < delta:
                    print('With garbage has better delta')
                    guesses = guesses_with_garbage
                    delta = delta_with_garbage
                    results = results_with_garbage
                    errors = errors_with_garbage
            delta_sum += delta
            # print(f'{img} {delta=}')
            # print(f'{img} {errors=}')
            if errors:
                for color, res in results.items():
                    if answer[color] != res:
                        errors_by_block[answer[color]] += 1
                        print(f'{img=} guessed : {results[color]=}, answer {answer[color]=}')
                total_errors += errors
        print(f'{delta_sum=}')
        print(f'{total_errors=}')
        print(f'{errors_by_block=}')
        delta_var.append(delta_sum)
    print(f'{var=} {delta_var=}')
    print(f'{var[delta_var.index(min(delta_var))]} {min(delta_var)}')


if __name__ == '__main__':
    adjust()
