import math
import logging
import torch
import time
import torch.nn as nn


from dataset import (
    int2binlist,
    input_len_from_number,
    generate_even_datasets_batch,
    convert_float_matrix_to_int_list,
)
from model import Generator, Discriminator


def get_logger(level: str = None) -> logging.Logger:
    """Intanciate logger for current app.

    Args:
        level (str): The level of the logging.

    Returns:
        logging.Logger: The new logger to use.
    """
    if level is None:
        level = logging.INFO

    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    return logger


logger = get_logger()


def train(max_int: int = 128, batch_size: int = 16, training_steps: int = 500) -> None:
    """Train the generator network and discriminator network together.

    Args:
        max_int (int): The maximum number to use during the batch generation.
        batch_size (int): The batch size to use during ints generation.
        trainint_steps (int): The number of the training iterations.

    Returns:
            None
    """
    input_length = input_len_from_number(max_int)

    # Models
    generator = Generator(input_length)
    discriminator = Discriminator(input_length)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # loss
    loss = nn.BCELoss()

    for i in range(training_steps):
        # zero the gradients on each iteration
        generator_optimizer.zero_grad()

        # Create noisy input for generator
        # Need float type instead of int
        noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
        generated_data = generator(noise)

        # Generate examples of even real data
        true_labels, true_data = generate_even_datasets_batch(
            max_int, batch_size=batch_size
        )
        true_labels = torch.tensor(true_labels).float().view(-1, 1)
        true_data = torch.tensor(true_data).float()

        # Train the generator
        # We invert the labels here and don't train the discriminator because we want the generator
        # to make things the discriminator classifies as true.
        generator_discriminator_out = discriminator(generated_data)
        generator_loss = loss(generator_discriminator_out, true_labels)
        generator_loss.backward()
        generator_optimizer.step()

        # Train the discriminator on the true/generated data
        discriminator_optimizer.zero_grad()
        true_discriminator_out = discriminator(true_data)
        true_discriminator_loss = loss(true_discriminator_out, true_labels)

        # add .detach() here think about this
        generator_discriminator_out = discriminator(generated_data.detach())
        generator_discriminator_loss = loss(
            generator_discriminator_out, torch.zeros(batch_size).view(-1, 1)
        )
        discriminator_loss = (
            true_discriminator_loss + generator_discriminator_loss
        ) / 2

        discriminator_loss.backward()
        discriminator_optimizer.step()

        if i % 10 == 0:
            data = convert_float_matrix_to_int_list(generated_data)

            logger.info(
                f"\n|-The epoch pass #{i}\n\t--> Generator loss: {generator_loss}. "
                f"\n\t--> Discriminator loss: {discriminator_loss}. "
                f"\n\t--> Generator - Discriminator loss: {generator_discriminator_loss}. "
                f"\n\t--> Data: {data}. \n\t--> Even numbers in list: {sum(1-n%2 for n in data)}"
            )


if __name__ == "__main__":
    train(training_steps=300)
