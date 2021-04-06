import torch
import math
import numpy as np
from scipy.special import hermite


class convTH:
    """
    Equation 7
    choose a complete basis of Nb steerable functions
        -  Ψ = {ψs−1σ,i}^{Nb}_{i=1} and represent convolutional filter as a linear combination of basis functions with
        trainable parameters w = {wi}^{Nb}_{i=1}
        -  In other words, we do the following substitution in Equation 7: ψσ → κ =sum( wiΨi ) over i

    basis
        -   2D Hermite polynomials with 2D Gaussian envelope,
        -   The basis is pre-calculated for all scales and fixed.
        -   For filters of size V × V , the basis is stored as an array of shape of [Nb, S, V, V ]
            -   Appendix C for more details

    Conv T → H
        -   If the input signal is a function on T (aka regular image)
            -   stored as an array of shape [Cin, U, U]
            -   Equation 7 can be simplified
                -   The summation over S degenerates
                -   convTH(f, w, Ψ) = squeeze(conv2d(f, expand(w × Ψ)))
                    -   w is an array of shape [Cout, Cin, Nb]
                    -   compute filter w×Ψ of shape [Cout, Cin, S, V, V ]
                    -   expand it to shape [Cout, CinS, V, V ]
                    -   use standard 2D convolution to produce the output with Cout*S channels
                    -   squeeze it to shape [Cout, S, U, U]
    """

    def __init__(self, number_of_scales):
        #precalculate basis
        basis=basis()
        pass

    def init_params(self):
        pass

    def forward(self, x):
        """ x: input tensor which has a shape of (N, C, H, W)
                (sample size, channel-in, height, width)

        Returns:
            y: output tensor which has a shape of (N, F, H', W') where
                H' = 1 + (H + 2 * padding - kernel_size) / stride
                W' = 1 + (W + 2 * padding - kernel_size) / stride
        """
        # sum over channels
        y = tensor()
        y.requiregrad = True
        y = torch.conv2d(x, ourweights, ourbias)
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
                        -   We sum the obtained KS results afterwards"""
    pass


def basis(x, n, y, m, A, sigma):
    p1 = A / (sigma ** 2)
    h1 = hermite(n, x / sigma)
    h2 = hermite(m, y / sigma)
    gaussian_envelope = math.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    psi = p1 * h1 * h2 * gaussian_envelope
    return psi
