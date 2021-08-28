import numpy as np
import math

from typing import List, Tuple


def int2binlist(number: int) -> List[int]:
    """Create the list of integers that forms the
    binary representation of the given number.

    Args:
        number (int): The integer number we want to convert to the list
                          of integers which represents ints binary form.

    Returns:
            List[int]: The list of integers.
    """
    return list(map(int, list(bin(number))[2:]))


def input_len_from_number(number: int) -> int:
    """Calculate the number of array elements
    needed to contains all bytes digist from the binary
    representation of the given number.

    Args:
        number (int): The number we want to get binary rep list size.

    Returns:
        int: The number of list elements to containt the number bytes digits.
    """
    return len(int2binlist(number))


def generate_even_datasets_batch(
    max_int: int, batch_size: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """Create the batch of the dataset wich consists of numbers.
    The labels of this dataset is the boolean value idicating is this number
    even or not.

    Args:
        max_int (int): The maximum integer in the dataset.
        batch_size (int): The number of data present in the one generated batch.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The tuple where the first element is the array of values
                                       and the second is the array of labels.

    """
    max_bytes_number = input_len_from_number(max_int)
    max_int = int(max_int / 2)

    sampled_integers = np.random.randint(1, max_int, batch_size)
    sampled_integers *= 2

    labels = [1] * batch_size

    data = np.zeros((batch_size, max_bytes_number))

    for i, sample in enumerate(sampled_integers):
        sample = np.array(int2binlist(sample))
        data[i][-sample.shape[0] :] = sample

    return labels, data


def convert_float_matrix_to_int_list(
    float_matrix: np.array, threshold: float = 0.5
) -> List[int]:
    """Converts generated output in binary list form to a list of integers
    Args:
        float_matrix: A matrix of values between 0 and 1 which we want to threshold and convert to
            integers
        threshold: The cutoff value for 0 and 1 thresholding.
    Returns:
        A list of integers.
    """
    return [
        int("".join([str(int(y)) for y in x]), 2) for x in float_matrix >= threshold
    ]
