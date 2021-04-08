import torch
import math
import numpy as np
from scipy.special import hermite

"""
    Equation 7
    choose a complete basis of Nb steerable functions
        -  Ψ = {ψs−1σ,i}^{Nb}_{i=1} and represent convolutional filter as a linear combination of basis functions with
        trainable parameters w = {wi}^{Nb}_{i=1}
        -  In other words, we do the following substitution in Equation 7: ψσ → κ =sum( wi * Ψi ) over i
"""


class convTH:
    def __init__(self, Cin, Cout, V, S, Nb):
        """
        :param Cin: number of input channels (regular conv)
        :param Cout: number of output channels (regular conv)
        :param V: how many pixel the filters in the filter bank should be [dimension V]
        :param S:
        :param Nb:
        """

        # save parameters
        self.Cin = Cin
        self.Cout = Cout
        self.V = V
        self.S = S
        self.Nb = Nb
        # bases have the shape [Nb, S, bases_size, bases_size]
        # TODO: hermite bases
        self.bases = hermite_bases()

        # weights
        # w is an array of shape [Cout, Cin, Nb]
        # Nb is the number of steerable functions we represent a filter with
        # To say a metaphor, the bigger Nb, the more terms we've fetched in a Taylor approximation ..
        #       if this is chinese, ask Mark about it
        self.weights = torch.Tensor(self.Cout, self.Cin, Nb)

        # biases
        # TODO: where to add bias? after the regular conv into every channel (Cout and S separately)?
        # shapes of bias:
        self.biases = torch.Tensor()

        self.init_params()

    def init_params(self, std=0.7071):
        self.weights = std * torch.randn_like(self.weights)
        # TODO: std for biases?
        self.biases = torch.rand_like(self.biases)

    def forward(self, x):
        """
        Conv T → H
            -   If the input signal is a function on T (aka regular image)
                -   stored as an array of shape [Cin, U, U]
                -   Equation 7 can be simplified
                    -   The summation over S degenerates
                    -   convTH(f, w, Ψ) = squeeze(conv2d(f, expand(w × Ψ)))
                        -   w is an array of shape [Cout, Cin, Nb]
                        -   compute filter w×Ψ of shape [Cout, Cin, S, V, V]
                        -   expand it to shape [Cout, Cin*S, V, V ]
                        -   use standard 2D convolution to produce the output with Cout*S channels
                        -   squeeze it to shape [Cout, S, U, U]

        This translates to:

        :param: x: input tensor which has a shape of [Cin, U, U] (channel-in, height, width)

        Returns:
            y: output tensor which has a shape of H - (Cout, S, U, U) where
                Cout = Output channels
                S = ???
                U = image size

        """

        # Extract dimensions
        _, U, _ = x.shape()

        # TODO: autograd
        # TODO: add batch size

        # compute filter w×Ψ of shape [Cout, Cin, S, V, V ]
        # w is weights, Ψ are the hermite stuff
        # w:        [Cout, Cin, Nb]
        # filter:   [Cout, Cin, S, V, V ]
        # TODO: check if torch.matmul is the function for our case (see annotated figure 1)
        filters = torch.matmul(self.weights, self.bases)
        # filters should be [Cout, Cin, S, V, V]
        print(filters.shape)

        # expand
        expanded_filters = filters.view(self.Cout, self.Cin * self.S, self.V, self.V)
        # expanded_filters should be [Cout, Cin*S, V, V]
        print(expanded_filters.shape)

        # convolve
        # TODO: bias?
        # todo: stride, padding, ... ?
        # TODO: is this the same as https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html ?
        conv_output = torch.conv2d(x, expanded_filters)
        # conv_output should be shaped [Cout, S, U, U] ... except when striding/padding is not normal
        print(conv_output.shape)

        # squeeze
        # todo: stride, padding, ... - fix at U
        # todo: calculate U at runtime
        y = conv_output.view(self.Cout, self.S, U, U)
        # y should be [Cout, S, U, U] ... expect when striding/padding...
        print(y.shape)

        return y


class convHH:
    """
    Conv H → H
        -   The function on H has a scale axis; two options for choosing weights of the convolutional filter.
            -   filter with just one scale
                -   does not capture the correlations between different scales of the input function
                -   w has shape [Cout, Cin, Nb] and Equation 7 degenerates in the same way as before
                -   Is it right here???   convHH(f, w, Ψ) = squeeze(conv2d(expand(f), expand(w × Ψ)))
                    -   We expand f to an array of shape [CinS, U, U]
                    -   expand w × Ψ to have shape [CoutS, CinS, V, V]
                    -   The result of the convolution squeezed in the same way as before

            -   it may have a non-unitary extent KS in the scale axis (WTF is that?)
                -   capture the correlation between KS neighboring scales
                -   called interscale interaction
                    -   w has shape [Cout, Cin, KS, Nb]
                    -   We iterate over all scales in interaction
                        -   shift f for each scale
                        -   choose a corresponding part of w
                        -   apply convHH to them
                        -   We sum the obtained KS results afterwards
    """
    pass


def hermite_bases(x, n, y, m, A, sigma):
    """
    basis
        -   2D Hermite polynomials with 2D Gaussian envelope,
        -   The basis is pre-calculated for all scales and fixed.
        -   For filters of size V × V , the basis is stored as an array of shape of [Nb, S, V, V ]
            -   Appendix C for more details
    """
    p1 = A / (sigma ** 2)
    h1 = hermite(n, x / sigma)
    h2 = hermite(m, y / sigma)
    gaussian_envelope = math.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    psi = p1 * h1 * h2 * gaussian_envelope
    return psi
