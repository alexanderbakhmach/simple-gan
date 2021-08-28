import torch
import logging


from torch import nn


class Generator(nn.Module):
    """The generator to generate the
    even number from noise vector.
    """

    def __init__(self, input_length: int):
        """The constructor. Takes the input
        length to construct the correct network layers.

        Args:
            input_length (int): The size of the input vector
                                that our generator network can take as
                                an input.
        """
        if not isinstance(input_length, int):
            raise ValueError("The input_length must be of type int.")

        super(Generator, self).__init__()

        output_length = input_length

        self._dense_layer = nn.Linear(input_length, output_length)
        self._avtivation = nn.Sigmoid()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Takes the vector x and pass it throught network

        Args:
            x (torch.Tensor): The input tensor with some size.

        Returns:
            torch.Tensor: The output tensor with the same size as an input one.
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError("The x must be torch Tensor")

        logging.debug("Passing the X throught network.")

        return self._avtivation(self._dense_layer(x))


class Discriminator(nn.Module):
    """The discriminator to compare
    generated data with the generator and
    the real data.
    """

    OUTPUT_LENGTH = 1

    def __init__(self, input_length: int):
        """The constructor. Takes the input
        length to construct the correct network layers.

        Args:
            input_length (int): The size of the input vector
                                that our discriminator network can take as
                                an input.
        """
        if not isinstance(input_length, int):
            raise ValueError("The input_length must be of type int.")

        super(Discriminator, self).__init__()

        self._dense_layer = nn.Linear(input_length, self.OUTPUT_LENGTH)
        self._activation = nn.Sigmoid()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Takes the vector x and pass it throught network

        Args:
            x (torch.Tensor): The input tensor with some size.

        Returns:
            torch.Tensor: The output tensor with the size 1 that tels
                          about categoty of the input tensor.
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError("The x must be torch Tensor")

        logging.debug("Passing the X throught network.")

        return self._avtivation(self._dense_layer(x))
