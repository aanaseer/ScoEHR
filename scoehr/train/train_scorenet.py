"""The training loop for the score network."""

import os
import time

import numpy as np
import torch
import tqdm
from torch.optim import lr_scheduler


def train_score_net(
    n_epochs,
    dataloader,
    optimiser,
    loss_fn_instance,
    score_net,
    output_step,
    checkpoint_save_step,
    checkpoint_dir_path,
):
    start_time = time.time()
    print(f"==> Training the score net with {n_epochs} epochs.")

    def lambda_func(epoch):
        return 1.0 - max(0, epoch - 20) / float(n_epochs - 10)

    scheduler = lr_scheduler.LambdaLR(optimiser, lr_lambda=lambda_func)

    score_net_learning_rates = []
    score_net_losses = []
    for epoch in range(n_epochs):
        all_loss = 0
        for x in tqdm.tqdm(dataloader):
            optimiser.zero_grad()
            loss = loss_fn_instance.loss_fn(x).mean()
            all_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1)
            optimiser.step()

        learning_rate_used = optimiser.param_groups[0]["lr"]
        avg_loss = all_loss / len(dataloader)

        score_net_learning_rates.append(learning_rate_used)
        score_net_losses.append(avg_loss)

        scheduler.step()

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
                checkpoint_dir_path, f"checkpoint_score_epoch_{epoch}.pt"
            )
            torch.save([loss_fn_instance, optimiser], checkpoint_file_name_and_path)

            # Save the losses and learning rates
            np.save(
                checkpoint_dir_path + "score_net_learning_rates",
                arr=score_net_learning_rates,
            )
            np.save(checkpoint_dir_path + "score_net_losses", arr=score_net_losses)

            print("==> Checkpoint saved.")
    print("==> Score net training completed.")
