"""Main file to run the training and generation process."""

import argparse
import math
import os
import time

import numpy as np
import torch
import torchsde
import yaml
from datasets import HongData, MIMIC3_ICD
from models.autoencoder import Autoencoder
from models.unet import UNet
from score_matching.dsm import DenoisingScoreMatching
from score_matching.sde_library import ReverseSDE, VPSDE, WrapperForTorchSDE
from torch.utils.data import DataLoader
from train.train_autoencoder_utils import train_autoencoder
from train.train_scorenet import train_score_net
from utils.convert_data_to_binary import convert_to_binary
from utils.create_directory import create
from utils.initialise_weights import weights_init


# Selecting a dataset
def choose_dataset(
    dataset_name,
    batch_size,
    test_size,
    save_test_train_data,
    device,
    data_type,
    data_dir,
):
    if dataset_name == "mimic3_icd_binary":
        train_data, test_data = MIMIC3_ICD(data_dir=data_dir).data(
            use_train_test_split=True, test_size=test_size
        )
        train_data = train_data.to(device)  # 32564 x 1071
        test_data = test_data.to(device)  # 13956 x 1071
        train_dataloader = DataLoader(
            dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True
        )

    elif dataset_name == "hong":
        hong_data = HongData(data_dir=data_dir)
        binary_indices = hong_data.binary_indices
        cts_indices = hong_data.cts_indices
        train_data, test_data = hong_data.data(use_train_test_split=True)
        train_data = train_data.to(device)
        test_data = test_data.to(device)
        train_dataloader = DataLoader(
            dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True
        )

    else:
        raise NotImplementedError(f"{dataset_name} does not exist.")

    if save_test_train_data:
        print("==> Saving a copy of the test and train data.")
        create(test_train_data_dump_path)
        train_data_file_name_path = os.path.join(
            test_train_data_dump_path, "train_data.npy"
        )
        test_data_file_name_path = os.path.join(
            test_train_data_dump_path, "test_data.npy"
        )
        np.save(train_data_file_name_path, train_data.to("cpu"))
        np.save(test_data_file_name_path, test_data.to("cpu"))
        print(
            f"==> A copy of the test and train data saved to {test_train_data_dump_path}."
        )

    enc_in_dim = train_data.shape[1]

    if data_type == "mixed":
        return train_dataloader, binary_indices, cts_indices, enc_in_dim
    else:
        return train_dataloader, enc_in_dim


def choose_score_net_architecture(
    score_net_architecture,
    score_net_in_dim,
    channel_mult,
    num_res_blocks,
    dropout,
):
    if dropout is None:
        dropout = 0
    if channel_mult is None:
        channel_mult = (1, 2, 2)
    if num_res_blocks is None:
        num_res_blocks = 2

    if score_net_architecture == "sdeflow_unet":
        score_net = UNet(
            input_channels=1,
            encoded_latent_embedding_dim=score_net_in_dim,
            ch=128,
            ch_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=(16,),
            resamp_with_conv=True,
            dropout=dropout,
        )
    else:
        raise NotImplementedError(f"{score_net_architecture} does not exist.")

    return score_net


# Initialise SDE, score_net, optimiser
def initialise_model(
    score_net_architecture,
    score_net_in_dim,
    device,
    T,
    lr_score_net,
    checkpoint_dir_path,
    load_checkpoint_score_net,
    loading_checkpoint_filepath_score_net,
    dataset,
    use_autoencoder,
    enc_out_dim,
    lr_autoencoder,
    load_checkpoint_autoencoder,
    loading_checkpoint_filepath_autoencoder,
    data_type,
    batch_size,
    padding_required,
    channel_mult,
    num_res_blocks,
    dropout,
    test_size,
    save_test_train_data,
    data_dir,
):
    print("==> Initialising SDEs, models, optimisers, loss functions and dataloaders.")
    sde = VPSDE()
    score_net = choose_score_net_architecture(
        score_net_architecture=score_net_architecture,
        score_net_in_dim=score_net_in_dim,
        channel_mult=channel_mult,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
    )
    score_net.to(device)
    loss_fn_instance_dsm = DenoisingScoreMatching(
        sde=sde, score_net=score_net, T=T, padding_required=padding_required
    )
    optimiser_score_net = torch.optim.Adam(
        loss_fn_instance_dsm.parameters(), lr=lr_score_net
    )

    if load_checkpoint_score_net:
        print("==> Loading checkpoint for score net found in: {checkpoint_dir_path}")

        loss_fn_instance_dsm, optimiser_score_net = torch.load(
            loading_checkpoint_filepath_score_net, map_location=device
        )
        score_net = loss_fn_instance_dsm.score_net

        print("==> Checkpoint for score net loaded.")

    reverse_sde = ReverseSDE(sde=sde, score_net=score_net, T=T)

    if data_type == "mixed":
        train_dataloader, binary_indices, cts_indices, enc_in_dim = choose_dataset(
            dataset_name=dataset,
            batch_size=batch_size,
            test_size=test_size,
            save_test_train_data=save_test_train_data,
            device=device,
            data_type=data_type,
            data_dir=data_dir,
        )
        init_objects = [
            sde,
            score_net,
            reverse_sde,
            optimiser_score_net,
            train_dataloader,
            loss_fn_instance_dsm,
            binary_indices,
            cts_indices,
        ]
    if data_type != "mixed":
        train_dataloader, enc_in_dim = choose_dataset(
            dataset_name=dataset,
            batch_size=batch_size,
            test_size=test_size,
            save_test_train_data=save_test_train_data,
            device=device,
            data_type=data_type,
            data_dir=data_dir,
        )
        init_objects = [
            sde,
            score_net,
            reverse_sde,
            optimiser_score_net,
            train_dataloader,
            loss_fn_instance_dsm,
        ]

    if use_autoencoder:
        autoencoder = Autoencoder(enc_in_dim, enc_out_dim)
        autoencoder.to(device)
        autoencoder.apply(weights_init)
        optimiser_autoencoder = torch.optim.Adam(
            autoencoder.parameters(), lr=lr_autoencoder, weight_decay=0.0001
        )

        if data_type == "binary":
            print("==> Choosing BCE loss for the autoencoder.")
            loss_fn_instance_autoencoder = torch.nn.BCELoss(
                reduction="sum"
            )  # WORKS FOR BINARY DATA!
        elif data_type == "mixed":
            binary_loss_fn_instance_autoencoder = torch.nn.BCELoss(reduction="sum")
            continuous_loss_fn_instance_autoencoder = torch.nn.MSELoss(reduction="sum")
            loss_fn_instance_autoencoder = (
                binary_loss_fn_instance_autoencoder,
                continuous_loss_fn_instance_autoencoder,
            )
        else:
            raise ValueError(f"{data_type} is incorrect.")

        if load_checkpoint_autoencoder:
            print(
                f"==> Loading checkpoint for autoencoder found in: {checkpoint_dir_path}"
            )

            autoencoder, optimiser_autoencoder = torch.load(
                loading_checkpoint_filepath_autoencoder, map_location=device
            )

            print("==> Checkpoint for autoencoder loaded.")
        decoder = autoencoder.decoder
        init_objects.extend(
            [autoencoder, decoder, optimiser_autoencoder, loss_fn_instance_autoencoder]
        )

    return init_objects


def initiate_training(
    n_epochs_score_net,
    output_step_score_net,
    checkpoint_save_step_score_net,
    checkpoint_dir_path,
    use_autoencoder,
    n_epochs_autoencoder,
    output_step_autoencoder,
    checkpoint_save_step_autoencoder,
    batch_size,
    data_type,
    dataset,
    score_net_architecture,
    score_net_in_dim,
    padding_required,
    channel_mult,
    num_res_blocks,
    dropout,
    lr_score_net,
    lr_autoencoder,
    T,
    load_checkpoint_score_net,
    loading_checkpoint_filepath_score_net,
    load_checkpoint_autoencoder,
    loading_checkpoint_filepath_autoencoder,
    data_dir,
    test_size,
    save_test_train_data,
):
    os.makedirs(checkpoint_dir_path, exist_ok=True)
    print("==> Commencing training.")

    if use_autoencoder:
        if data_type == "mixed":
            (
                sde,
                score_net,
                reverse_sde,
                optimiser_score_net,
                train_dataloader_autoencoder,
                loss_fn_instance_dsm,
                binary_indices,
                cts_indices,
                autoencoder,
                decoder,
                optimiser_autoencoder,
                loss_fn_instance_autoencoder,
            ) = initialise_model(
                score_net_architecture=score_net_architecture,
                score_net_in_dim=score_net_in_dim,
                device=device,
                T=T,
                lr_score_net=lr_score_net,
                checkpoint_dir_path=checkpoint_dir_path,
                load_checkpoint_score_net=load_checkpoint_score_net,
                loading_checkpoint_filepath_score_net=loading_checkpoint_filepath_score_net,
                dataset=dataset,
                use_autoencoder=use_autoencoder,
                enc_out_dim=score_net_in_dim,
                lr_autoencoder=lr_autoencoder,
                load_checkpoint_autoencoder=load_checkpoint_autoencoder,
                loading_checkpoint_filepath_autoencoder=loading_checkpoint_filepath_autoencoder,
                data_type=data_type,
                batch_size=batch_size,
                padding_required=padding_required,
                channel_mult=channel_mult,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                test_size=test_size,
                save_test_train_data=save_test_train_data,
                data_dir=data_dir,
            )

            train_autoencoder(
                n_epochs=n_epochs_autoencoder,
                dataloader=train_dataloader_autoencoder,
                optimiser=optimiser_autoencoder,
                loss_fn=loss_fn_instance_autoencoder,
                autoencoder=autoencoder,
                output_step=output_step_autoencoder,
                checkpoint_save_step=checkpoint_save_step_autoencoder,
                checkpoint_dir_path=checkpoint_dir_path,
                data_type=data_type,
                binary_indices=binary_indices,
                cts_indices=cts_indices,
            )
        else:
            (
                sde,
                score_net,
                reverse_sde,
                optimiser_score_net,
                train_dataloader_autoencoder,
                loss_fn_instance_dsm,
                autoencoder,
                decoder,
                optimiser_autoencoder,
                loss_fn_instance_autoencoder,
            ) = initialise_model(
                score_net_architecture=score_net_architecture,
                score_net_in_dim=score_net_in_dim,
                device=device,
                T=T,
                lr_score_net=lr_score_net,
                checkpoint_dir_path=checkpoint_dir_path,
                load_checkpoint_score_net=load_checkpoint_score_net,
                loading_checkpoint_filepath_score_net=loading_checkpoint_filepath_score_net,
                dataset=dataset,
                use_autoencoder=use_autoencoder,
                enc_out_dim=score_net_in_dim,
                lr_autoencoder=lr_autoencoder,
                load_checkpoint_autoencoder=load_checkpoint_autoencoder,
                loading_checkpoint_filepath_autoencoder=loading_checkpoint_filepath_autoencoder,
                data_type=data_type,
                batch_size=batch_size,
                padding_required=padding_required,
                channel_mult=channel_mult,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                test_size=test_size,
                save_test_train_data=save_test_train_data,
                data_dir=data_dir,
            )

            train_autoencoder(
                n_epochs=n_epochs_autoencoder,
                dataloader=train_dataloader_autoencoder,
                optimiser=optimiser_autoencoder,
                loss_fn=loss_fn_instance_autoencoder,
                autoencoder=autoencoder,
                output_step=output_step_autoencoder,
                checkpoint_save_step=checkpoint_save_step_autoencoder,
                checkpoint_dir_path=checkpoint_dir_path,
                data_type=data_type,
                binary_indices=None,
                cts_indices=None,
            )

        data_for_score_net = autoencoder.encode(
            train_dataloader_autoencoder.dataset
        ).detach()
        train_dataloader_score_net = DataLoader(
            dataset=data_for_score_net,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

    else:
        if data_type == "mixed":
            (
                sde,
                score_net,
                reverse_sde,
                optimiser_score_net,
                train_dataloader_score_net,
                loss_fn_instance_dsm,
                binary_indices,
                cts_indices,
            ) = initialise_model(
                score_net_architecture=score_net_architecture,
                score_net_in_dim=score_net_in_dim,
                device=device,
                T=T,
                lr_score_net=lr_score_net,
                checkpoint_dir_path=checkpoint_dir_path,
                load_checkpoint_score_net=load_checkpoint_score_net,
                loading_checkpoint_filepath_score_net=loading_checkpoint_filepath_score_net,
                dataset=dataset,
                use_autoencoder=use_autoencoder,
                enc_out_dim=score_net_in_dim,
                lr_autoencoder=lr_autoencoder,
                load_checkpoint_autoencoder=load_checkpoint_autoencoder,
                loading_checkpoint_filepath_autoencoder=loading_checkpoint_filepath_autoencoder,
                data_type=data_type,
                batch_size=batch_size,
                padding_required=padding_required,
                channel_mult=channel_mult,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                test_size=test_size,
                save_test_train_data=save_test_train_data,
                data_dir=data_dir,
            )

        else:
            (
                sde,
                score_net,
                reverse_sde,
                optimiser_score_net,
                train_dataloader_score_net,
                loss_fn_instance_dsm,
            ) = initialise_model(
                score_net_architecture=score_net_architecture,
                score_net_in_dim=score_net_in_dim,
                device=device,
                T=T,
                lr_score_net=lr_score_net,
                checkpoint_dir_path=checkpoint_dir_path,
                load_checkpoint_score_net=load_checkpoint_score_net,
                loading_checkpoint_filepath_score_net=loading_checkpoint_filepath_score_net,
                dataset=dataset,
                use_autoencoder=use_autoencoder,
                enc_out_dim=score_net_in_dim,
                lr_autoencoder=lr_autoencoder,
                load_checkpoint_autoencoder=load_checkpoint_autoencoder,
                loading_checkpoint_filepath_autoencoder=loading_checkpoint_filepath_autoencoder,
                data_type=data_type,
                batch_size=batch_size,
                padding_required=padding_required,
                channel_mult=channel_mult,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                test_size=test_size,
                save_test_train_data=save_test_train_data,
                data_dir=data_dir,
            )

    train_score_net(
        n_epochs=n_epochs_score_net,
        dataloader=train_dataloader_score_net,
        optimiser=optimiser_score_net,
        loss_fn_instance=loss_fn_instance_dsm,
        score_net=score_net,
        output_step=output_step_score_net,
        checkpoint_save_step=checkpoint_save_step_score_net,
        checkpoint_dir_path=checkpoint_dir_path,
    )


def generate_samples(
    time_steps,
    device,
    col_dim,
    num_samples_to_generate,
    generated_data_dir_path,
    save_generated_data,
    checkpoint_dir_path,
    use_autoencoder,
    batch_size,
    data_type,
    dataset,
    score_net_architecture,
    score_net_in_dim,
    padding_required,
    channel_mult,
    num_res_blocks,
    dropout,
    lr_score_net,
    lr_autoencoder,
    T,
    load_checkpoint_score_net,
    loading_checkpoint_filepath_score_net,
    load_checkpoint_autoencoder,
    loading_checkpoint_filepath_autoencoder,
    data_dir,
    test_size,
    save_test_train_data,
):
    os.makedirs(generated_data_dir_path, exist_ok=True)

    if not load_checkpoint_score_net:
        print(
            "The load_checkpoint argument for score net should be True to generate samples."
        )
        exit()

    print("==> Commencing sample generation.")

    if use_autoencoder:
        if not load_checkpoint_autoencoder:
            print(
                "The load_checkpoint argument for autoencoder should be True to generate samples."
            )
            exit()
        else:
            if data_type == "mixed":
                (
                    sde,
                    score_net,
                    reverse_sde,
                    _,
                    train_dataloader,
                    _,
                    binary_indices,
                    cts_indices,
                    autoencoder,
                    _,
                    _,
                    _,
                ) = initialise_model(
                    score_net_architecture=score_net_architecture,
                    score_net_in_dim=score_net_in_dim,
                    device=device,
                    T=T,
                    lr_score_net=lr_score_net,
                    checkpoint_dir_path=checkpoint_dir_path,
                    load_checkpoint_score_net=load_checkpoint_score_net,
                    loading_checkpoint_filepath_score_net=loading_checkpoint_filepath_score_net,
                    dataset=dataset,
                    use_autoencoder=use_autoencoder,
                    enc_out_dim=score_net_in_dim,
                    lr_autoencoder=lr_autoencoder,
                    load_checkpoint_autoencoder=load_checkpoint_autoencoder,
                    loading_checkpoint_filepath_autoencoder=loading_checkpoint_filepath_autoencoder,
                    data_type=data_type,
                    batch_size=batch_size,
                    padding_required=padding_required,
                    channel_mult=channel_mult,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    test_size=test_size,
                    save_test_train_data=save_test_train_data,
                    data_dir=data_dir,
                )
            else:
                (
                    sde,
                    score_net,
                    reverse_sde,
                    _,
                    train_dataloader,
                    _,
                    _,
                    _,
                    autoencoder,
                    _,
                    _,
                    _,
                ) = initialise_model(
                    score_net_architecture=score_net_architecture,
                    score_net_in_dim=score_net_in_dim,
                    device=device,
                    T=T,
                    lr_score_net=lr_score_net,
                    checkpoint_dir_path=checkpoint_dir_path,
                    load_checkpoint_score_net=load_checkpoint_score_net,
                    loading_checkpoint_filepath_score_net=loading_checkpoint_filepath_score_net,
                    dataset=dataset,
                    use_autoencoder=use_autoencoder,
                    enc_out_dim=score_net_in_dim,
                    lr_autoencoder=lr_autoencoder,
                    load_checkpoint_autoencoder=load_checkpoint_autoencoder,
                    loading_checkpoint_filepath_autoencoder=loading_checkpoint_filepath_autoencoder,
                    data_type=data_type,
                    batch_size=batch_size,
                    padding_required=padding_required,
                    channel_mult=channel_mult,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    test_size=test_size,
                    save_test_train_data=save_test_train_data,
                    data_dir=data_dir,
                )

            autoencoder.eval()
    elif not use_autoencoder:
        if data_type == "mixed":
            (
                sde,
                score_net,
                reverse_sde,
                _,
                train_dataloader,
                _,
                _,
                _,
            ) = initialise_model(
                score_net_architecture=score_net_architecture,
                score_net_in_dim=score_net_in_dim,
                device=device,
                T=T,
                lr_score_net=lr_score_net,
                checkpoint_dir_path=checkpoint_dir_path,
                load_checkpoint_score_net=load_checkpoint_score_net,
                loading_checkpoint_filepath_score_net=loading_checkpoint_filepath_score_net,
                dataset=dataset,
                use_autoencoder=use_autoencoder,
                enc_out_dim=score_net_in_dim,
                lr_autoencoder=lr_autoencoder,
                load_checkpoint_autoencoder=load_checkpoint_autoencoder,
                loading_checkpoint_filepath_autoencoder=loading_checkpoint_filepath_autoencoder,
                data_type=data_type,
                batch_size=batch_size,
                padding_required=padding_required,
                channel_mult=channel_mult,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                test_size=test_size,
                save_test_train_data=save_test_train_data,
                data_dir=data_dir,
            )
        else:
            (sde, score_net, reverse_sde, _, train_dataloader, _,) = initialise_model(
                score_net_architecture=score_net_architecture,
                score_net_in_dim=score_net_in_dim,
                device=device,
                T=T,
                lr_score_net=lr_score_net,
                checkpoint_dir_path=checkpoint_dir_path,
                load_checkpoint_score_net=load_checkpoint_score_net,
                loading_checkpoint_filepath_score_net=loading_checkpoint_filepath_score_net,
                dataset=dataset,
                use_autoencoder=use_autoencoder,
                enc_out_dim=score_net_in_dim,
                lr_autoencoder=lr_autoencoder,
                load_checkpoint_autoencoder=load_checkpoint_autoencoder,
                loading_checkpoint_filepath_autoencoder=loading_checkpoint_filepath_autoencoder,
                data_type=data_type,
                batch_size=batch_size,
                padding_required=padding_required,
                channel_mult=channel_mult,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                test_size=test_size,
                save_test_train_data=save_test_train_data,
                data_dir=data_dir,
            )

    print("==> Commencing reverse solve using torchsde Euler Maruyama.")
    torchsde_SDE = WrapperForTorchSDE(
        reverse_sde=reverse_sde, noise_type="diagonal", sde_type="ito"
    )
    ts = torch.linspace(0, 1, time_steps + 1) * reverse_sde.T
    ts = ts.to(device)
    with torch.no_grad():
        assert (
            num_samples_to_generate >= batch_size
        ), "Num to generate should be greater than or equal to batch size."
        batch_iter = math.ceil(num_samples_to_generate / batch_size)
        print(f"==> Able to generate {batch_iter * batch_size} samples.")

        xs_batches = []
        if use_autoencoder:
            for i in range(0, batch_iter):
                print(f"==> Batch {i} out of {batch_iter}.")
                x_0 = torch.randn(batch_size, col_dim, device=device)
                x_batch_solved = torchsde.sdeint(torchsde_SDE, x_0, ts, method="euler")
                if data_type == "binary":
                    print(f"==> NOTE: the data type chosen is {data_type}")
                    with torch.no_grad():
                        print(
                            f"==> Decoding data generated in batch {i} out of {batch_iter}."
                        )
                        x_batch_solved = torch.stack([x_batch_solved])
                        for x_encoded in x_batch_solved:
                            out = autoencoder.decode(x_encoded.to(device))
                            out = out.cpu().detach()
                            xs_batches.append(out)
                            del out
                            torch.cuda.empty_cache()

                elif data_type == "mixed":
                    print(f"==> NOTE: the data type chosen is {data_type}")
                    with torch.no_grad():
                        print(
                            f"==> Decoding data generated in batch {i} out of {batch_iter}."
                        )
                        x_batch_solved = torch.stack(
                            [x_batch_solved]
                        )  # torch.Size([1, 101, 10, 64])

                        for (
                            x_encoded
                        ) in (
                            x_batch_solved
                        ):  # x_encoded.shape = torch.Size([101, 10, 64])
                            out = autoencoder.decode(x_encoded.to(device))
                            out = (
                                out.cpu().detach()
                            )  # out.shape = torch.Size([101, 10, 625])

                            temp_out = [
                                torch.concat(
                                    (
                                        torch.tensor(out[i][:, cts_indices]),
                                        out[i][:, binary_indices],
                                    ),
                                    dim=1,
                                )
                                for i in range(out.shape[0])
                            ]
                            out = torch.stack(temp_out, dim=0)
                            xs_batches.append(out)
                            del out
                            del temp_out
                            torch.cuda.empty_cache()

            xs = torch.cat(
                xs_batches, 1
            )  # this requires xs_batches = [tensor, tensor,..]
        else:
            for i in range(0, batch_iter):
                print(f"==> Batch {i} out of {batch_iter}.")
                x_0 = torch.randn(batch_size, col_dim, device=device)
                out = torchsde.sdeint(torchsde_SDE, x_0, ts, method="euler")

                def unpad(x, pad=(79, 80, 0, 0)):
                    if pad[0] + pad[1] > 0:
                        x = x[:, :, pad[0] : -pad[1]]
                    return x

                out = unpad(out)
                out = out.cpu().detach()
                xs_batches.append(out)
                del out
                torch.cuda.empty_cache()

            xs = torch.cat(xs_batches, 1)

        print("==> Reverse solve using torchsde Euler Maruyama completed.")

    if not use_autoencoder:
        xs = torch.stack(xs)

    if use_autoencoder:
        xs_decoded = xs.clone().detach()  # torch.Size([101, 20, 625])
        if data_type == "binary":
            print("==> Generated samples have been decoded.")
            print("==> Converting decoded generated samples to binary data.")
            xs = convert_to_binary(data=xs_decoded)

            print("==> Binary data conversion completed.")

        elif data_type == "mixed":
            print("==> Generated samples have been decoded.")
            print("==> Converting decoded generated samples to binary data.")
            temp_xs_decoded = [
                torch.concat(
                    (
                        xs_decoded[i][:, cts_indices].clone().detach(),
                        convert_to_binary(xs_decoded[i][:, binary_indices]),
                    ),
                    dim=1,
                )
                for i in range(xs_decoded.shape[0])
            ]
            xs = torch.stack(temp_xs_decoded, dim=0)

    generated_data_at_every_time_step = xs.numpy()
    generated_data_final = generated_data_at_every_time_step[-1]
    filepath_generated_data_final = os.path.join(
        generated_data_dir_path, "generated_data_final"
    )

    if save_generated_data:
        create(generated_data_dir_path)
        print("==> Saving generated data.")
        np.savez_compressed(filepath_generated_data_final, a=generated_data_final)
        print(f"==> Generated data saved to {generated_data_dir_path}.")

    return generated_data_at_every_time_step, generated_data_final


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exptime = time.strftime("%Y%m%d-%H%M%S")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--experiment", type=str)
    expname = arg_parser.parse_args().experiment

    configs = yaml.safe_load_all(open("configs/experiments.yaml"))
    _, configs = configs

    if expname not in configs:
        raise ValueError(f"Experiment {expname} not found in config file.")

    args = argparse.Namespace(exptime=exptime, **configs[expname])

    if args.mode == "train":
        load_checkpoint_score_net = False
        load_checkpoint_autoencoder = False
    else:
        load_checkpoint_score_net = True
        load_checkpoint_autoencoder = True

    if args.dataset == "mimic3_icd_binary":
        data_type = "binary"
        if args.use_autoencoder:
            score_net_in_dim = args.score_net_in_dim
        else:
            score_net_in_dim = 1071
    elif args.dataset == "hong":
        data_type = "mixed"
        if args.use_autoencoder:
            score_net_in_dim = args.score_net_in_dim
        else:
            score_net_in_dim = 625
    else:
        raise ValueError(f"{args.dataset} data_type and enc_dim not been chosen.")

    if not args.use_autoencoder:
        folder_name_keys = [
            "exptime",
            "mode",
            "dataset",
            "score_net_architecture",
            "n_epochs_score_net",
            "lr_score_net",
            "batch_size",
        ]
    else:
        folder_name_keys = [
            "exptime",
            "mode",
            "dataset",
            "score_net_architecture",
            "n_epochs_score_net",
            "lr_score_net",
            "batch_size",
            "n_epochs_autoencoder",
            "lr_autoencoder",
        ]

    # Setting up the output directory
    folder_name = "-".join([str(getattr(args, k)) for k in folder_name_keys])
    create(args.output_dir, folder_name)
    output_dir_path = os.path.join(args.output_dir, folder_name)

    checkpoint_dir_path = os.path.join(output_dir_path, args.checkpoint_dir)
    generated_data_dir_path = os.path.join(output_dir_path, "generated_data")
    test_train_data_dump_path = os.path.join(output_dir_path, "test_train_data_used")
    create(checkpoint_dir_path)

    if args.mode == "train":
        initiate_training(
            n_epochs_score_net=args.n_epochs_score_net,
            output_step_score_net=args.output_step_score_net,
            checkpoint_save_step_score_net=args.checkpoint_save_step_score_net,
            checkpoint_dir_path=checkpoint_dir_path,
            use_autoencoder=args.use_autoencoder,
            n_epochs_autoencoder=args.n_epochs_autoencoder,
            output_step_autoencoder=args.output_step_autoencoder,
            checkpoint_save_step_autoencoder=args.checkpoint_save_step_autoencoder,
            batch_size=args.batch_size,
            data_type=data_type,
            dataset=args.dataset,
            score_net_architecture=args.score_net_architecture,
            score_net_in_dim=score_net_in_dim,
            padding_required=args.padding_required,
            channel_mult=None,
            num_res_blocks=None,
            dropout=None,
            lr_score_net=args.lr_score_net,
            lr_autoencoder=args.lr_autoencoder,
            T=args.T,
            load_checkpoint_score_net=args.load_checkpoint_score_net,
            loading_checkpoint_filepath_score_net=args.loading_checkpoint_filepath_score_net,
            load_checkpoint_autoencoder=args.load_checkpoint_autoencoder,
            loading_checkpoint_filepath_autoencoder=args.loading_checkpoint_filepath_autoencoder,
            data_dir=args.data_dir,
            test_size=args.test_train_size,
            save_test_train_data=args.save_test_train_data,
        )
    if args.mode == "generate":
        xs, final_samples = generate_samples(
            time_steps=args.euler_maruyama_time_steps,
            device=device,
            col_dim=score_net_in_dim,
            num_samples_to_generate=args.num_samples_to_generate,
            generated_data_dir_path=generated_data_dir_path,
            save_generated_data=args.save_generated_samples,
            checkpoint_dir_path=checkpoint_dir_path,
            use_autoencoder=args.use_autoencoder,
            batch_size=args.batch_size,
            data_type=data_type,
            dataset=args.dataset,
            score_net_architecture=args.score_net_architecture,
            score_net_in_dim=score_net_in_dim,
            padding_required=args.padding_required,
            channel_mult=None,
            num_res_blocks=None,
            dropout=None,
            lr_score_net=args.lr_score_net,
            lr_autoencoder=args.lr_autoencoder,
            T=args.T,
            load_checkpoint_score_net=args.load_checkpoint_score_net,
            loading_checkpoint_filepath_score_net=args.loading_checkpoint_filepath_score_net,
            load_checkpoint_autoencoder=args.load_checkpoint_autoencoder,
            loading_checkpoint_filepath_autoencoder=args.loading_checkpoint_filepath_autoencoder,
            data_dir=args.data_dir,
            test_size=args.test_train_size,
            save_test_train_data=args.save_test_train_data,
        )
