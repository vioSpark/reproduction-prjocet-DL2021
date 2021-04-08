from SESN_parts import convTH
import numpy as np


def dummy_image():
    x_values = np.linspace(-10, 10, 50)
    y_values = np.linspace(-10, 10, 50)
    return np.add.outer(x_values,y_values)


cth = convTH(1, 1, 5, 5, 10)
dummy_pic = dummy_image()
cth.forward(dummy_pic)

# TODO: move py files to src
