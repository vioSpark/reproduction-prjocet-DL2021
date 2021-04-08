import torch
import math
import numpy as np
from scipy.special import eval_hermite

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
        # bases have the shape [Nb, S, V, V]
        self.bases = hermite_bases(Nb, S, V)

        # weights
        # w is an array of shape [Cout, Cin, Nb]
        # Nb is the number of steerable functions we represent a filter with
        # To say a metaphor, the bigger Nb, the more terms we've fetched in a Taylor approximation ..
        #       if this is chinese, ask Mark about it
        self.weights = torch.Tensor(self.Cout, self.Cin, Nb).double()
        self.weights.requires_grad = True

        # biases
        # TODO: where to add bias? after the regular conv into every channel (Cout and S separately)?
        #  -- Mark's tip: at the conv2d line
        # shapes of bias:
        self.biases = torch.Tensor()
        self.biases.requires_grad = True

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

        :param: x: input tensor which has a shape of [N,Cin, U, U] (batch-size, channel-in, height, width)

        Returns:
            y: output tensor which has a shape of H - (Cout, S, U, U) where
                Cout = Output channels
                S = ???
                U = image size

        """

        # Extract dimensions
        N, C_in, U, _ = x.shape

        # TODO: autograd
        # TODO: add batch size

        # compute filter w×Ψ of shape [Cout, Cin, S, V, V ]
        # w is weights, Ψ are the hermite stuff
        # w:        [Cout, Cin, Nb]
        # filter:   [Cout, Cin, S, V, V ]
        # TODO: check if torch.matmul is the function for our case (see annotated figure 1)
        # TODO: cross-check this ein-sum
        filters = torch.einsum('ijk, klmn -> ijlmn', self.weights, self.bases)
        # filters should be [Cout, Cin, S, V, V]
        check = (filters.shape == (self.Cout, self.Cin, self.S, self.V, self.V))
        print(check)

        # expand
        # TODO: notify the authors that the paper might have a pretty significant 'bug' (I'm 95% sure):
        # "...expand it to shape[Cout, CinS, V, V]. Then we use standard..." here it should be [CoutS, Cin, V, V]
        expanded_filters = filters.view(self.Cout * self.S, self.Cin, self.V, self.V)
        # expanded_filters should be [Cout, Cin*S, V, V] -> wrong, see above
        check = (expanded_filters.shape == (self.Cout * self.S, self.Cin, self.V, self.V))
        print(check)

        # convolve
        # TODO: bias?
        # todo: stride, padding, ... ? -- atm temporary fixed for retaining image shape w padding
        # TODO: is this the same as https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html ?
        conv_output = torch.conv2d(x, expanded_filters, padding=math.floor(self.V / 2))
        # conv_output should be shaped [Cout * S, U, U] ... except when striding/padding is not normal
        # plus batch-size -> [N, Cout * S, U, U]
        check = (conv_output.shape == (N, self.Cout * self.S, U, U))
        print(check)

        # squeeze
        # todo: stride, padding, ... - fix at U
        # todo: calculate U at runtime
        y = conv_output.view(N, self.Cout, self.S, U, U)
        # y should be [Cout, S, U, U] ... expect when striding/padding...
        check = (y.shape == (N, self.Cout, self.S, U, U))
        print(check)

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


def hermite_bases(Nb, S, V, A=1, sigma=5):
    """
    :param: Nb: number of bases ... pay attention to the pyramid form (image below)
            to get a complete layer of the pyramid, you need Nb=n(n+1)/2 where n is an element of N
    :param: S: number of how many scales we will have
    :param: V: x*y size of each base
    :param: A: global scale factor for filter
    :param: sigma: The sigma that's on the power of S
                    sigma is the biggest scale, and exponentially converges to 1 in S steps
    basis
        -   2D Hermite polynomials with 2D Gaussian envelope,
        -   The basis is pre-calculated for all scales and fixed.
        -   For filters of size V × V , the basis is stored as an array of shape of [Nb, S, V, V ]
            -   Appendix C for more details

        -   sigma seems to be the scale parameter -> corresponds to S
        -   Nb seems to be the polynomial order -> n and m parameters?
        -   see this pic: https://www.researchgate.net/publication/308494563/figure/fig1/AS:560765801566208@1510708393139/Two-dimensional-Hermite-TDH-functions-of-rank-0-to-7-in-A-polar-form-and-B.png
        -   what on earth should A be?
    """
    bases = torch.Tensor(Nb, S, V, V).double()
    # determine n-m from Nb
    # TODO: optimize this for loop
    config_list = generate_hermite_list(Nb)
    i = 0
    for element in config_list:
        n, m = element
        # TODO: check if it's okay to run sigma like this
        for s in range(S):
            # sigma_inner runs from sigma ** 1 -> sigma ** (1/S)
            # TODO: double check if sigma_inner is conceptually right
            sigma_inner = sigma ** ((s + 1) ** -1)
            x = np.linspace(int(math.ceil(-V / 2)), int(math.floor(V / 2)), V, dtype=np.double)
            y = np.linspace(int(math.ceil(-V / 2)), int(math.floor(V / 2)), V, dtype=np.double)
            p1 = A / (sigma_inner ** 2)
            # note: using "physicist's Hermite polynomials" by using scipy
            h1 = eval_hermite(n, x / sigma_inner)
            h2 = eval_hermite(m, y / sigma_inner)
            # TODO: double check if this is the gaussian?
            #  ... might be the case that gaussian can be defined in multiple ways?? WTF??
            gaussian_envelope = np.exp(-np.add.outer(x ** 2, y ** 2) / (2 * sigma_inner ** 2))
            bases[i, s] = torch.from_numpy(p1 * np.outer(h1, h2) * gaussian_envelope)
        i += 1

    return bases


def generate_hermite_list(Nb):
    # this code is ugly as fuck
    counter = 0
    config = []
    while True:
        for i in range(counter + 1):
            config.append((i, counter - i))
            if len(config) == Nb:
                return config
        counter += 1
