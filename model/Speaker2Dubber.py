import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Decoder, PostNet, Encoder, Lip_Encoder
from .modules import LengthRegulator, MelStyleEncoder, Prosody_Consistency_Learning, CTC_classifier_MDA
from utils.tools import get_mask_from_lengths, generate_square_subsequent_mask
LRELU_SLOPE = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from monotonic_align import mask_from_lens, maximum_path

class Speaker2Dubber(nn.Module):
    """ From Speaker to Dubber """
    def __init__(self, preprocess_config, model_config):
        super(Speaker2Dubber, self).__init__()
        self.model_config = model_config
        self.dataset_name = preprocess_config["dataset"]
        # self.style_encoder = MelStyleEncoder(model_config)  # In fact, during conducting expriment, we remove this auxiliary style encoder (V2C-Net from Chenqi, et.al). Specifically, we only use the pre-trained GE2E model to gurateen only style information without content information, following the setting of paper. 
        self.phoneme_encoder = Encoder(model_config)
        self.lip_encoder = Lip_Encoder(model_config)
        self.phoneme_proj_con = nn.Conv1d(256, 256, kernel_size=1, padding=0, bias=False)
        self.CTC_classifier_MDA = CTC_classifier_MDA(model_config["Symbols"]["phonemenumber"])  # len(symbols)
        self.PCL = Prosody_Consistency_Learning(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.pre_net_bottleneck = model_config["transformer"]["pre_net_bottleneck"]
        self.postnet = PostNet()
        self.proj_fusion = nn.Conv1d(768, 256, kernel_size=1, padding=0, bias=False)
        self.length_regulator=LengthRegulator()
        self.use_mel_style_encoder = model_config['transformer']["use_mel_style_encoder"]
        if self.use_mel_style_encoder:
            self.mel_style_encoder = MelStyleEncoder(preprocess_config, model_config)
        
        self.Identity_enhancement = model_config["Enhancement"]["Identity_enhancement"]
        self.Content_enhancement = model_config["Enhancement"]["Content_enhancement"]

        self.pro_output_os = nn.Conv1d(512, 256, kernel_size=1, padding=0, bias=False)
        self.n_speaker = 1
        if model_config["multi_speaker"]:
            with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
            ) as f:
                self.n_speaker = len(json.load(f))
            with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
            ) as f:
                self.n_speaker += len(json.load(f))
            self.speaker_emb = nn.Embedding(
                self.n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )
        self.pro_output_emo = nn.Conv1d(512, 256, kernel_size=1, padding=0, bias=False)
        
        self.CTC_classifier_mel = CTC_classifier_mel(model_config["Symbols"]["phonemenumber"])  # len(symbols)
        self.n_emotion = 1

        if model_config["with_emotion"]:
            self.n_emotion = preprocess_config["preprocessing"]["emotion"]["n_emotion"]
            self.emotion_emb = nn.Embedding(
                self.n_emotion + 1,
                model_config["transformer"]["encoder_hidden"],
                padding_idx=self.n_emotion,
            )
        self.duration_search = False
        
        
    def forward(
            self,
            speakers,
            texts, # 3
            src_lens, # 4
            max_src_len, # 5
            mels=None,  #  6
            mel_lens=None,
            max_mel_len=None,
            p_targets=None,
            e_targets=None,
            d_targets=None,
            spks=None,
            emotions=None,
            emos=None,
            Feature_256=None,
            lip_lens = None,
            max_lip_lens = None,
            lip_embedding = None,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
            useGT=None,
            train_mode=None,
            epoch=None,
    ):
        """===========mask for voice, text, lip-movement========"""
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        visual_masks = get_mask_from_lengths(lip_lens, max_lip_lens)

        if self.use_mel_style_encoder:
            style_vector = self.mel_style_encoder(mels, get_mask_from_lengths(mel_lens, max_mel_len))
            spks = style_vector.squeeze()


        """===========get phoneme embedding and lip motion embedding========"""
        phoneme_embeddings = self.phoneme_encoder(texts, spks, None, src_masks)
        lip_motion_embedding = self.lip_encoder(lip_embedding, visual_masks)

        output_phoneme = self.phoneme_proj_con(phoneme_embeddings.transpose(1, 2))
        output_phoneme = output_phoneme.transpose(1, 2)

        ctc_pred_MDA_video = self.CTC_classifier_MDA(output_phoneme)

    
        """=========Prosody Consistency Learning========="""
        (output, p_predictions, e_predictions,) = self.PCL(output_phoneme, src_masks, visual_masks, p_targets, e_targets,
                                                        Feature_256, p_control, e_control, useGT, train_mode=train_mode)

        
        """=========Duration Consistency Reasoning========="""

        mean_cof = (d_targets.sum(dim=-1) / lip_lens).mean()
        norm_lip = lip_motion_embedding / torch.norm(lip_motion_embedding, dim=2, keepdim=True)
        text_norm_base = torch.norm(output_phoneme, dim=2, keepdim=True)
        norm_text = output_phoneme / (text_norm_base+1)
        norm_text = torch.nan_to_num(norm_text, nan=1)
        similarity = torch.bmm(norm_lip, norm_text.permute(0, 2, 1))
        similarity = similarity.permute(0, 2, 1)

        cofs = d_targets.sum(dim=1)/lip_lens
        lip_targets = torch.round(d_targets / cofs.unsqueeze(1))
        dif = lip_lens - lip_targets.sum(dim=1)
        max_duration_idx = torch.argmax(lip_targets, dim=1)

        # finetune the length
        for i in range(lip_lens.shape[0]):
            lip_targets[i][max_duration_idx[i]] += dif[i]

        gt_similarity = torch.zeros_like(similarity)
        for i in range(similarity.shape[0]):
            begin=0
            for line in range(similarity.shape[1]):
                end = begin + int(lip_targets[i][line])
                gt_similarity[i, line, begin:end] = 1
                begin = end

        mask_sim = mask_from_lens(similarity, src_lens, lip_lens)
        alignment = maximum_path(similarity.contiguous(), mask_sim)  # (B, S, T)
        d_predictions = alignment.sum(axis=-1).detach() * mean_cof

        if useGT:
            prosody, mel_lens = self.length_regulator(output, d_targets, None)
            # mel_lens = torch.sum(d_predictions, dim=1, dtype=torch.int)
            mel_masks = get_mask_from_lengths(mel_lens)
            ctc_pred_MDA_video, mel_len = self.length_regulator(ctc_pred_MDA_video, d_targets, max_mel_len)

        else:
            prosody, mel_lens = self.length_regulator(output, d_predictions * d_control, None)
            # mel_lens = torch.sum(d_predictions, dim=1, dtype=torch.int)
            mel_masks = get_mask_from_lengths(mel_lens)
            ctc_pred_MDA_video, mel_len = self.length_regulator(ctc_pred_MDA_video, d_predictions, None)

            
        fusion_output = prosody

        """=========Mel-Generator========="""
        fusion_output, mel_masks = self.decoder(fusion_output, mel_masks, spks, None)
        ctc_pred_mel = self.CTC_classifier_mel(fusion_output)
        
        fusion_output = self.mel_linear(fusion_output)
        postnet_output = self.postnet(fusion_output) + fusion_output
        ctc_loss_all = [ctc_pred_MDA_video, ctc_pred_mel]

        return (
            fusion_output,
            postnet_output,
            p_predictions,
            e_predictions,
            # d_predictions,
            [similarity, gt_similarity],
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            ctc_loss_all,
            max_src_len,
        )

class CTC_classifier_mel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Linear(256, num_classes)
    def forward(self, Dub):
        size = Dub.size()
        Dub = Dub.reshape(-1, size[2]).contiguous()
        Dub = self.classifier(Dub)
        return Dub.reshape(size[0], size[1], -1) 
        




