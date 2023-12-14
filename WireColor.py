from enum import Enum
import random


class WireColor(Enum):
    GREEN = (1, 0, 0, 0)
    YELLOW = (0, 1, 0, 0)
    BLUE = (0, 0, 1, 0)
    RED = (0, 0, 0, 1)
    NO_WIRE = (0, 0, 0, 0)


if __name__ == '__main__':
    # example
    print(WireColor.RED.value)
