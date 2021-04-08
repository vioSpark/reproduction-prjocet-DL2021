from SESN_parts import convTH
import numpy as np
import torch


def dummy_image():
    x_values = np.linspace(-10, 10, 50)
    y_values = np.linspace(-10, 10, 50)
    z_values = (1, 2, 3)
    base_channel = np.add.outer(x_values, y_values)
    img = np.multiply.outer(z_values, base_channel)
    batch_maker = (1, 1)
    img = np.multiply.outer(batch_maker, img)
    return torch.from_numpy(img)


cth = convTH(3, 2, 7, 5, 6)
dummy_pic = dummy_image()
layer2 = cth.forward(dummy_pic)
print('requires grad:', cth.weights.requires_grad)

external_grad = torch.rand_like(layer2)
layer2.backward(gradient=external_grad)
grads = cth.weights.grad
print(grads.shape)
print(grads)

# TODO: move py files to src
