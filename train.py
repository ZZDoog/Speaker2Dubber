import argparse
import os
import shutil
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num, vocoder_infer
from utils.tools import to_device, log, synth_one_sample
from model import Speaker2Dubber_Loss
from dataset import Dataset, PretrainDataset

from evaluate import evaluate, evaluate_v2

import sys 
sys.path.append("..")
from resemblyzer import VoiceEncoder
import shutil


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


def main(args, configs):
    print("Prepare training Speaker2Dubber model...")
    preprocess_config, model_config, train_config = configs
    if model_config['train_mode'] == 'pretrain':
        dataset = PretrainDataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True, diff_audio=False)
    else:
        dataset = Dataset("train.txt", preprocess_config, train_config, sort=False, drop_last=True, diff_audio=False)

    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 1   # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)

    # Load vocoder (HiFiGAN, https://github.com/jik876/hifi-gan)
    vocoder = get_vocoder(model_config, device)

    encoder_spk = VoiceEncoder().to(device)
    encoder_emo = VoiceEncoder().to(device)
    encoder_speaker = VoiceEncoder().to(device)
    encoder_spk.eval()
    encoder_emo.eval()
    encoder_speaker.train()

    Loss = Speaker2Dubber_Loss(preprocess_config, model_config).to(device)
    print("Number of Speaker2Dubber Parameters:", num_param)

    # Load vocie encoder (compute the wav embedding for accuracy only)
    # encoder_spk = VoiceEncoder(weights_fpath=\
    #     "/home/conggaoxiang/Desktop/Avatar2/V2C/audio_encoder/MovieAnimation_bak_1567500.pt").to(device)
    # encoder_emo = VoiceEncoder(weights_fpath=\
    #     "/home/conggaoxiang/Desktop/Avatar2/V2C/audio_encoder/MovieAnimation_bak_1972500.pt").to(device)
    # The emotion and speaker encoder is provided by V2C-Net (chenqi et.al, https://github.com/chenqi008/V2C).
    

    # Init logger
    for p in train_config["path"].values():

        # protect the experiment log and checkpoints, except debug
        if train_config['expname'] == 'debug':
            os.makedirs(p.format(train_config['expname']), exist_ok=True)
        else:
            os.makedirs(p.format(train_config['expname']), exist_ok=True)

    script_path = train_config['path']['script_path'].format(train_config['expname'])
    train_log_path = os.path.join(train_config["path"]["log_path"].format(train_config['expname']), "train")
    val_log_path = os.path.join(train_config["path"]["log_path"].format(train_config['expname']), "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Copy the Script and config
    file_to_save = ['train.py', 'Inference.py', 'dataset.py','evaluate.py',
                     'mcd.py', 'prepare_align.py', 'preprocess.py']
    folder_to_copy = ['model', 'transformer', 'config']

    for x in file_to_save:
        shutil.copy2(x, script_path)
    
    for x in folder_to_copy:
        if os.path.exists('{}/{}'.format(script_path, x)):
            shutil.rmtree('{}/{}'.format(script_path, x))
        shutil.copytree(x, '{}/{}'.format(script_path, x))

    # rec and gen samples
    train_samples_path = os.path.join(train_log_path, "samples")
    val_samples_path = os.path.join(val_log_path, "samples")
    if os.path.exists(train_samples_path):
        shutil.rmtree(train_samples_path)
    if os.path.exists(val_samples_path):
        shutil.rmtree(val_samples_path)
    os.makedirs(train_samples_path, exist_ok=True)
    os.makedirs(val_samples_path, exist_ok=True)

    # Training
    step = args.restore_step
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    loss_model = model_config["loss_function"]["model"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()
    while True:
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                output = model(*(batch[2:]), useGT=True, train_mode=model_config['train_mode'], epoch=epoch)

                # Cal Loss
                losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    if loss_model == "V2C":
                        message2 = "Epoch:{} Total: {:.4f}, Mel_mae: {:.4f}, Mel_Post_mae: {:.4f}, Pitch MSE: {:.4f}, Energy MSE: {:.4f}, Pitch MAE: {:.4f}, Energy MAE: {:.4f}, Mel_mse: {:.4f}, Mel_Post_mse: {:.4f}, Emo_cross: {:.4f}, CTC_MDA_video: {:.4f}, CTC_MEL: {:.4f}, speaker_loss: {:.4f}, Duration_MSE: {:.4f}, Duration_MAE: {:.4f}".format(epoch,
                            *losses)
                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses, LM=loss_model)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]

                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0 and step!=0:
                    model.eval()
                    if model_config['train_mode'] == 'pretrain':
                        message= evaluate_v2(model, step, configs, val_logger, \
                            vocoder, Loss)
                    else:
                        message= evaluate(model, step, configs, val_logger, \
                            vocoder, encoder_spk, encoder_emo, \
                            train_samples_path, val_samples_path, useGT=False)
                    
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)
                    model.train()

                if train_config['expname'] != 'debug':
                    if step % save_step == 0 and step!=0:
                        torch.save(
                            {
                                # "model": model.state_dict(),  
                                "model": model.module.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict(),
                            },
                            os.path.join(
                                train_config["path"]["ckpt_path"].format(train_config['expname']),
                                "{}.pth.tar".format(step),
                            ),
                        )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
    
