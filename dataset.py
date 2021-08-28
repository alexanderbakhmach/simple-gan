import logging
import numpy as np
import math

from typing import List, Tuple


def get_logger() -> logging.Logger:
    """Intanciate logger for current app.

    Returns:
        logging.Logger: The new logger to use.
    """
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    return logger


logger = get_logger()


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
    max_int = int(max_int / 2)
    sampled_integers = np.random.randint(0, max_int, batch_size)
    sampled_integers *= 2

    labels = [1] * batch_size

    max_sample_int = sampled_integers.max()
    max_bytes_number = len(int2binlist(max_sample_int))

    data = np.zeros((batch_size, max_bytes_number))

    for i, sample in enumerate(sampled_integers):
        sample = np.array(int2binlist(sample))
        data[i][-sample.shape[0] :] = sample

    return labels, data


if __name__ == "__main__":
    logger.info("Start creating the dataset")

    logger.info(generate_even_datasets_batch(10))
