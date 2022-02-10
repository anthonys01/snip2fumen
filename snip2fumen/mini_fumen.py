"""
Mini fumen encoder only for this use case (original encoder https://github.com/knewjade/tetris-fumen)

Only support single board, no comments, action, garbage...
"""

ENCODE_TABLE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'


EMPTY = 0
I = 1
L = 2
O = 3
Z = 4
T = 5
J = 6
S = 7
GRAY = 8


class Buffer:
    tableLength = len(ENCODE_TABLE)

    def __init__(self):
        self.values = list()

    def __len__(self):
        return len(self.values)

    def push(self, value: int, split_count: int = 1):
        current = value
        for count in range(split_count):
            self.values.append(current % Buffer.tableLength)
            current = current // Buffer.tableLength

    def merge(self, b):
        for v in b.values:
            self.values.append(v)

    def get(self, i):
        return self.values[i]

    def set(self, i, v):
        self.values[i] = v

    def get_encoded_string(self):
        return "".join([ENCODE_TABLE[v] for v in self.values])


FIELD_TOP = 23
FIELD_MAX_HEIGHT = FIELD_TOP + 1
FIELD_WIDTH = 10
FIELD_BLOCKS = FIELD_MAX_HEIGHT * FIELD_WIDTH


class Field:
    def __init__(self):
        self.field = [0] * (FIELD_TOP * FIELD_WIDTH)
        self.garbage = [0] * FIELD_WIDTH

    @staticmethod
    def index_for(x, y):
        return x + y * FIELD_WIDTH

    def get_number_at(self, x: int, y: int):
        return self.field[Field.index_for(x, y)] if y >= 0 else self.garbage[Field.index_for(x, -(y + 1))]

    def set_number_field_at(self, x: int, y: int, value: int):
        self.field[Field.index_for(x, y)] = value

    def clear_line(self):
        new_field = self.field[:]
        top = len(new_field) // FIELD_WIDTH - 1
        for y in reversed(range(top + 1)):
            line = self.field[y * FIELD_WIDTH: (y + 1) * FIELD_WIDTH]
            is_filled = True
            for b in line:
                if b == EMPTY:
                    is_filled = False
                    break
            if is_filled:
                bottom = new_field[:y * FIELD_WIDTH]
                over = new_field[(y + 1) * FIELD_WIDTH:]
                new_field = bottom + over
        self.field = new_field


def encode_field(prev: Field, current: Field):
    buffer = Buffer()

    def get_diff(xIndex, yIndex):
        y = FIELD_TOP - yIndex - 1
        return current.get_number_at(xIndex, y) - prev.get_number_at(xIndex, y) + 8

    def record_block_counts(d, c):
        buffer.push(d * FIELD_BLOCKS + c, 2)

    prev_diff = get_diff(0, 0)
    counter = -1
    changed = False
    for yIndex in range(FIELD_MAX_HEIGHT):
        for xIndex in range(FIELD_WIDTH):
            diff = get_diff(xIndex, yIndex)
            if diff != prev_diff:
                record_block_counts(prev_diff, counter)
                counter = 0
                prev_diff = diff
                changed = True
            else:
                counter += 1
    record_block_counts(prev_diff, counter)

    return changed, buffer


def encode(field: Field):
    last_repeat = -1
    buffer = Buffer()

    def update_field(last_repeat_index, prev: Field, current: Field):
        changed, values = encode_field(prev, current)

        if changed:
            buffer.merge(values)
            last_repeat_index = -1
        elif last_repeat_index < 0 or buffer.get(last_repeat_index) == Buffer.tableLength - 1:
            buffer.merge(values)
            buffer.push(0)
            last_repeat_index = len(buffer) - 1
        elif buffer.get(last_repeat_index) < Buffer.tableLength - 1:
            current_repeat_value = buffer.get(last_repeat_index)
            buffer.set(last_repeat_index, current_repeat_value + 1)
        return last_repeat_index

    empty_piece = {
        "type": EMPTY,
        "rotation": 0,  # Rotation.REVERSE is 2 but is encoded as 0 in actionEncoder
        "x": 0,
        "y": 22
    }

    action = {
        "piece": empty_piece,
        "rise": False,
        "lock": True,
        "mirror": False,
        "comment": False,
        "colorize": True
    }

    def encode_action(a):
        val = int(not a["lock"])
        val *= 2
        val += int(a["comment"])
        val *= 2
        val += int(a["colorize"])
        val *= 2
        val += int(a["mirror"])
        val *= 2
        val += int(a["rise"])
        val *= FIELD_WIDTH * FIELD_MAX_HEIGHT
        val += (FIELD_TOP - 22 - 1) * FIELD_WIDTH + 0  # because piece type is EMPTY
        val *= 4
        val += a["piece"]["rotation"]
        val *= 8
        val += a["piece"]["type"]
        return val

    last_repeat = update_field(last_repeat, Field(), field)
    buffer.push(encode_action(action), 3)

    if action["lock"]:
        field.clear_line()

    res = buffer.get_encoded_string()
    if len(res) < 41:
        return "v115@" + res

    head = res[:42]
    tail = res[42:]
    split = [tail[x:x+47] for x in range(0, len(tail), 47)]
    return "v115@" + "?".join([head] + split)


if __name__ == "__main__":
    print(encode(Field()))
