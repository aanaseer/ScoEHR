"""The training loop for the autoencoder."""

import os
import time

import numpy as np
import torch
import tqdm


def train_autoencoder(
    n_epochs,
    dataloader,
    optimiser,
    loss_fn,
    autoencoder,
    output_step,
    checkpoint_save_step,
    checkpoint_dir_path,
    data_type,
    binary_indices=None,
    cts_indices=None,
):
    start_time = time.time()
    print(f"==> Training the autoencoder with {n_epochs} epochs.")
    autoencoder_learning_rates = []
    autoencoder_losses = []

    if data_type == "mixed":
        (
            binary_loss_fn_instance_autoencoder,
            continuous_loss_fn_instance_autoencoder,
        ) = loss_fn
    else:
        pass

    for epoch in range(n_epochs):
        all_loss = 0
        for x in tqdm.tqdm(dataloader):
            optimiser.zero_grad()
            encode_and_decode = autoencoder(x)

            if data_type == "mixed":
                loss_binary = (
                    binary_loss_fn_instance_autoencoder(
                        encode_and_decode[:, binary_indices], x[:, binary_indices]
                    )
                    / x[:, binary_indices].shape[0]
                )
                loss_continuous = (
                    continuous_loss_fn_instance_autoencoder(
                        encode_and_decode[:, cts_indices], x[:, cts_indices]
                    )
                    / x[:, cts_indices].shape[0]
                )
                loss = loss_binary + loss_continuous
            else:
                loss = loss_fn(encode_and_decode, x) / x.shape[0]

            all_loss += loss.item()
            loss.backward()

            optimiser.step()

        learning_rate_used = optimiser.param_groups[0]["lr"]
        avg_loss = all_loss / len(dataloader)
        autoencoder_learning_rates.append(learning_rate_used)
        autoencoder_losses.append(avg_loss)

        if (epoch + 1) % output_step == 0 or epoch == 0 or epoch == n_epochs:
            time_elapsed = time.time() - start_time
            print(
                f"Epoch: {epoch}  |  Time: {time_elapsed / output_step:.2f}s/epoch  \
                    | Loss: {avg_loss:.5f} | Learning rate: {learning_rate_used}"
            )
            start_time = time.time()

        if (epoch + 1) % checkpoint_save_step == 0 or epoch == n_epochs:
            print(f"==> Saving checkpoint to: {checkpoint_dir_path}")
            checkpoint_file_name_and_path = os.path.join(
                checkpoint_dir_path, f"checkpoint_autoenc_epoch_{epoch}.pt"
            )
            torch.save([autoencoder, optimiser], checkpoint_file_name_and_path)

            # Save the losses and learning rates
            np.save(
                checkpoint_dir_path + "autoencoder_learning_rates",
                arr=autoencoder_learning_rates,
            )
            np.save(checkpoint_dir_path + "autoencoder_losses", arr=autoencoder_losses)

            print("==> Checkpoint saved.")

    print("==> Autoencoder training completed.")
