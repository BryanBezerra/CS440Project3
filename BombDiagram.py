import numpy as np
import random
from WireColor import WireColor


class BombDiagram:
    """Generates a random bomb diagram according to the guidelines.

    Attributes:
        image: A 3D numpy array of the same (image_size, image_size, 4) representing
        the wire configuration.
        isDangerous: true if the configuration is dangerous, otherwise false
        wireToCut: which WireColor should be cut, None if the bomb is not dangerous
    """
    NUMBER_OF_COLORS = 4

    def __init__(self, image_size):
        """Initializes the instance based on image size.

        Args:
            image_size: the number of pixels on each side of the diagram
        """
        self.isDangerous = False
        self.wireToCut = None
        self.image = np.zeros((image_size, image_size, self.NUMBER_OF_COLORS), dtype=np.uint8)
        self.create_random_image()

    def create_random_image(self):
        """Creates a random image conforming to the generation guidelines.

        Four wires of different colors are laid down alternating rows and
        columns, half the time starting with rows and half the time with columns.
        """
        image_size = self.image.shape[0]
        remaining_colors = [WireColor.RED, WireColor.BLUE, WireColor.YELLOW, WireColor.GREEN]
        remaining_first_indices = [i for i in range(image_size)]
        remaining_second_indices = remaining_first_indices.copy()
        r = random.random()
        if r < .5:
            first_action = self.color_column
            second_action = self.color_row
        else:
            first_action = self.color_row
            second_action = self.color_column

        while len(remaining_colors) > 0:
            r = random.randrange(len(remaining_first_indices))
            index = remaining_first_indices.pop(r)
            r = random.randrange(len(remaining_colors))
            color = remaining_colors.pop(r)
            first_action(index, color)

            if WireColor.RED not in remaining_colors and WireColor.YELLOW in remaining_colors:
                self.isDangerous = True

            if self.isDangerous and len(remaining_colors) == 1:
                self.wireToCut = color

            r = random.randrange(len(remaining_second_indices))
            index = remaining_second_indices.pop(r)
            r = random.randrange(len(remaining_colors))
            color = remaining_colors.pop(r)
            second_action(index, color)

            if WireColor.RED not in remaining_colors and WireColor.YELLOW in remaining_colors:
                self.isDangerous = True

    def color_column(self, column, color):
        """Colors one column in the image.

        Args:
            column: The column's index.
            color: A WireColor i.e. WireColor.RED.
        """
        if type(color) != WireColor:
            raise TypeError("'color' must be a WireColor")
        self.image[:, column] = np.array(color.value)

    def color_row(self, row, color):
        """Colors one row in the image.

        Args:
            row: The row's index.
            color: A WireColor i.e. WireColor.RED.
        """
        if type(color) != WireColor:
            raise TypeError("'color' must be a WireColor")
        self.image[row, :] = np.array(color.value)

    def __str__(self):
        """Shows the diagram, whether it's dangerous, and which wire to cut"""
        RED = "\033[31m█ \033[0m"
        YELLOW = "\33[33m█ \033[0m"
        GREEN = "\033[32m█ \033[0m"
        BLUE = "\033[34m█ \033[0m"
        NEUTRAL = "\33[90m█ \033[0m"

        result = ""
        for row in range(self.image.shape[0]):
            for col in range(self.image.shape[1]):
                pixel = tuple(map(tuple, self.image[[row], [col]]))[0]
                if pixel == WireColor.RED.value:
                    result += RED
                elif pixel == WireColor.YELLOW.value:
                    result += YELLOW
                elif pixel == WireColor.GREEN.value:
                    result += GREEN
                elif pixel == WireColor.BLUE.value:
                    result += BLUE
                else:
                    result += NEUTRAL
            result += "\n"
        result += "Is dangerous: " + str(self.isDangerous) + "\n"
        if self.isDangerous:
            result += "Wire to cut: " + str(self.wireToCut.name)
        else:
            result += "No need to cut a wire"
        return result

    def is_dangerous(self):
        return self.isDangerous

    def get_wire_to_cut(self):
        return self.wireToCut

    def get_image(self):
        return self.image.copy()

    def get_flat_image(self):
        """Returns a flat numpy array representing the image.

        An extra 1 value is appended to the start of the array to make dot product easier during training.
        """
        extra_one = np.array([1])
        return np.hstack((extra_one, self.image.flatten()))


if __name__ == '__main__':
    """Example"""
    # Checking a pixel's color
    test = BombDiagram(20)
    print(test)
    print(test.get_flat_image())

