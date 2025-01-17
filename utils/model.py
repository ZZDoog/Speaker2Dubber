import os
import json

import torch
import numpy as np

from utils.env import AttrDict
from hifi_gan.models import Generator
from utils.istft_models import istft_Generator

from utils.stft import TorchSTFT
from model import Speaker2Dubber, ScheduledOptim

MAX_WAV_VALUE = 32768.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(args, configs, device, train=False):
    # (preprocess_config, model_config, train_config) = configs
    (preprocess_config, model_config, train_config) = configs
    model = Speaker2Dubber(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"].format(train_config['expname']),
            "{}.pth.tar".format(args.restore_step),
        )

        print("{} loaded".format(ckpt_path))
        
        ckpt = torch.load(ckpt_path)
        # remove keys of pretrained model that are not in our model (i.e., embeeding layer)
        model_dict = model.state_dict()
        if model_config["learn_speaker"]:
            speaker_emb_weight = ckpt["model"]["speaker_emb.weight"]
            s, d = speaker_emb_weight.shape
        ckpt["model"] = {k: v for k, v in ckpt["model"].items() \
                         if k in model_dict and k != "speaker_emb.weight"}
        model.load_state_dict(ckpt["model"], strict=False)
        if model_config["learn_speaker"] and s <= model.state_dict()["speaker_emb.weight"].shape[0]:
            model.state_dict()["speaker_emb.weight"][:s, :] = speaker_emb_weight
    
    if model_config['train_mode'] == 'finetune' and train:
        ckpt_path = model_config['pretrain_ckpt_path']
        ckpt = torch.load(ckpt_path)

        # Load the pretrained phoneme encoder from the ckpt
        # pretrained_encoder_state_dict = {k.replace('MDA.encoder.', ''): v
        #                   for k, v in ckpt['model'].items() if k.startswith('MDA.encoder.')}
        # phoneme_encoder_state_dict = model.phoneme_encoder.state_dict()
        # for k in pretrained_encoder_state_dict:
        #     if k in phoneme_encoder_state_dict:
        #         phoneme_encoder_state_dict[k] = pretrained_encoder_state_dict[k]
        #     else:
        #         print(f"Warning: {k} not found in Speaker2Dubber.encoder. Skipping.")
        # model.phoneme_encoder.load_state_dict(phoneme_encoder_state_dict)

        # Load the pretrained decoder from the ckpt
        # pretrained_encoder_state_dict = {k.replace('decoder.', ''): v
        #                   for k, v in ckpt['model'].items() if k.startswith('decoder.')}
        # phoneme_encoder_state_dict = model.decoder.state_dict()
        # for k in pretrained_encoder_state_dict:
        #     if k in phoneme_encoder_state_dict:
        #         phoneme_encoder_state_dict[k] = pretrained_encoder_state_dict[k]
        #     else:
        #         print(f"Warning: {k} not found in Speaker2Dubber.encoder. Skipping.")
        # model.decoder.load_state_dict(phoneme_encoder_state_dict)
        
        # remove keys of pretrained model that are not in our model (i.e., embeeding layer)
        model_dict = model.state_dict()
        if model_config["learn_speaker"]:
            speaker_emb_weight = ckpt["model"]["speaker_emb.weight"]
            s, d = speaker_emb_weight.shape
        ckpt["model"] = {k: v for k, v in ckpt["model"].items() \
                         if k in model_dict and k != "speaker_emb.weight"}
        model.load_state_dict(ckpt["model"], strict=False)
        if model_config["learn_speaker"] and s <= model.state_dict()["speaker_emb.weight"].shape[0]:
            model.state_dict()["speaker_emb.weight"][:s, :] = speaker_emb_weight

        print("{} loaded".format(ckpt_path))

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model



def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param



# 2024.1.3 new hifigan (hop length=256 version)

def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)

    # elif name == "HiFi-GAN":
    #     if speaker.split('_')[1] == '22k':
    #         with open("hifigan/config_22k.json", "r") as f:
    #             config = json.load(f)
    #     elif speaker.split('_')[1] == '16k':
    #         with open("hifigan/config_16k.json", "r") as f:
    #             config = json.load(f)

    #     config = hifigan.AttrDict(config)
    #     vocoder = hifigan.Generator(config)

    #     if speaker == "LibriTTS_22k":
    #         ckpt = torch.load("hifigan/pretrained/generator_universal.pth.tar")
    #     elif speaker == "AISHELL3_22k":
    #         ckpt = torch.load("hifigan/pretrained/generator_aishell3.pth.tar")
        
    #     vocoder.load_state_dict(ckpt["generator"])
    #     vocoder.eval()
    #     vocoder.remove_weight_norm()
    #     vocoder.to(device)
    
    elif name == "realHiFi-GAN_UniverVersion":
        config_file = os.path.join("./hifi_gan/UNIVERSAL_V1", "config.json")
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(h.seed)
        vocoder = Generator(h).to(device)
        state_dict_g = load_checkpoint(os.path.join("./hifi_gan/UNIVERSAL_V1", "g_02500000"),
                                       device)
        vocoder.load_state_dict(state_dict_g['generator'])
        vocoder.eval()
        vocoder.remove_weight_norm()

    return vocoder

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)
        elif name == "realHiFi-GAN_UniverVersion":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
