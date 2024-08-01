import os
import json
import copy
import math
from collections import OrderedDict
from torch.nn.utils import weight_norm
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from text.symbols import symbols

from utils.tools import get_mask_from_lengths, pad
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from utils.tools import init_weights, get_padding
from transformer import Encoder, Lip_Encoder
LRELU_SLOPE = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from monotonic_align import mask_from_lens, maximum_path

from .blocks import (
    Mish,
    FCBlock,
    Conv1DBlock,
    SALNFFTBlock,
    MultiHeadAttention,
)


class MelStyleEncoder(nn.Module):
    """ Mel-Style Encoder """

    def __init__(self, preprocess_config, model_config):
        super(MelStyleEncoder, self).__init__()
        n_position = model_config["max_seq_len"] + 1
        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        d_melencoder = model_config["melencoder"]["encoder_hidden"]
        n_spectral_layer = model_config["melencoder"]["spectral_layer"]
        n_temporal_layer = model_config["melencoder"]["temporal_layer"]
        n_slf_attn_layer = model_config["melencoder"]["slf_attn_layer"]
        n_slf_attn_head = model_config["melencoder"]["slf_attn_head"]
        d_k = d_v = (
            model_config["melencoder"]["encoder_hidden"]
            // model_config["melencoder"]["slf_attn_head"]
        )
        kernel_size = model_config["melencoder"]["conv_kernel_size"]
        dropout = model_config["melencoder"]["encoder_dropout"]

        self.max_seq_len = model_config["max_seq_len"]
        self.d_melencoder = d_melencoder

        self.fc_1 = FCBlock(n_mel_channels, d_melencoder)

        self.spectral_stack = nn.ModuleList(
            [
                FCBlock(
                    d_melencoder, d_melencoder, activation=Mish()
                )
                for _ in range(n_spectral_layer)
            ]
        )

        self.temporal_stack = nn.ModuleList(
            [
                nn.Sequential(
                    Conv1DBlock(
                        d_melencoder, 2 * d_melencoder, kernel_size, activation=Mish(), dropout=dropout
                    ),
                    nn.GLU(),
                )
                for _ in range(n_temporal_layer)
            ]
        )

        self.slf_attn_stack = nn.ModuleList(
            [
                MultiHeadAttention(
                    n_slf_attn_head, d_melencoder, d_k, d_v, dropout=dropout, layer_norm=True
                )
                for _ in range(n_slf_attn_layer)
            ]
        )

        self.fc_2 = FCBlock(d_melencoder, d_melencoder)

    def forward(self, mel, mask):

        max_len = mel.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        enc_output = self.fc_1(mel)

        # Spectral Processing
        for _, layer in enumerate(self.spectral_stack):
            enc_output = layer(enc_output)

        # Temporal Processing
        for _, layer in enumerate(self.temporal_stack):
            residual = enc_output
            enc_output = layer(enc_output)
            enc_output = residual + enc_output

        # Multi-head self-attention
        for _, layer in enumerate(self.slf_attn_stack):
            residual = enc_output
            enc_output, _ = layer(
                enc_output, enc_output, enc_output, mask=slf_attn_mask
            )
            enc_output = residual + enc_output

        # Final Layer
        enc_output = self.fc_2(enc_output) # [B, T, H]

        # Temporal Average Pooling
        enc_output = torch.mean(enc_output, dim=1, keepdim=True) # [B, 1, H]

        return enc_output


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len
    

class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class CTC_classifier_MDA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # B, S, 512
        size = x.size()
        x = x.reshape(-1, size[2]).contiguous()
        x = self.classifier(x)
        return x.reshape(size[0], size[1], -1)  
    

class Prosody_Consistency_Learning(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(Prosody_Consistency_Learning, self).__init__()
        self.proj_con = nn.Conv1d(512, 256, kernel_size=1, padding=0, bias=False)
        
        self.dataset_name = preprocess_config["dataset"]
        self.emo_fc_2_val = nn.Sequential(nn.Linear(256, 256),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(256, 256),
                                      nn.ReLU(inplace=True),
                                      )
        self.emo_fc_2_aro = nn.Sequential(nn.Linear(256, 256),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(256, 256),
                                      nn.ReLU(inplace=True),
                                      )

        self.W = nn.Linear(256, 256)
        self.Uo = nn.Linear(256, 256)
        self.Um = nn.Linear(256, 256)

        self.bo = nn.Parameter(torch.ones(256), requires_grad=True)
        self.bm = nn.Parameter(torch.ones(256), requires_grad=True)

        self.wo = nn.Linear(256, 1)
        self.wm = nn.Linear(256, 1)
        self.inf = 1e5
        
        self.valence_attention = nn.MultiheadAttention(256, 8, dropout=0.1)
        self.arousal_attention = nn.MultiheadAttention(256, 8, dropout=0.1)

        self.loss_model = model_config["loss_function"]["model"]

        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.scale_fusion = model_config["Affective_Prosody_Adaptor"]["Use_Scale_attention"]
        self.predictor_ = model_config["variance_predictor"]["predictor"]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

    def get_pitch_embedding(self, x, target, mask, control, useGT, train_mode=None):

        prediction = self.pitch_predictor(x, mask)
        if useGT:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control, useGT, train_mode=None):
        
        prediction = self.energy_predictor(x, mask)
        if useGT:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )

        return prediction, embedding

    def forward(
            self,
            x,
            src_masks=None,
            visual_masks=None,
            pitch_target=None,
            energy_target=None,
            Feature_256=None,
            p_control=1.0,
            e_control=1.0,
            useGT=None,
            train_mode=None,
    ):  

        M = x
        valence = self.emo_fc_2_val(Feature_256)
        if self.scale_fusion:
            context_valence, _ = self.valence_attention(query=M.transpose(0, 1), 
                                                        key=valence.transpose(0, 1), 
                                                        value=valence.transpose(0, 1),
                                                        key_padding_mask=visual_masks)
            context_valence = context_valence.transpose(0, 1)
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            context_valence, pitch_target, src_masks, p_control, useGT
        )

        Arousal = self.emo_fc_2_aro(Feature_256)
        if self.scale_fusion:
            context_arousal, _ = self.arousal_attention(query=M.transpose(0, 1), 
                                                        key=Arousal.transpose(0, 1), 
                                                        value=Arousal.transpose(0, 1),
                                                        key_padding_mask=visual_masks)
            context_arousal = context_arousal.transpose(0, 1)
        energy_prediction, energy_embedding = self.get_energy_embedding(
            context_arousal, energy_target, src_masks, e_control, useGT
        )

        output = x + pitch_embedding + energy_embedding
    
        return (
            output,
            pitch_prediction,
            energy_prediction,
        )


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function.
    In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    """
    Gradient Reversal Layer
    Code from:
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
    """
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    
class AdversarialClassifier(nn.Module):
    """
    AdversarialClassifier
        - 1 gradident reversal layer
        - n hidden linear layers with ReLU activation
        - 1 output linear layer with Softmax activation
    """
    def __init__(self, in_dim, out_dim, hidden_dims=[256], rev_scale=1):
        """
        Args:
            in_dim: input dimension
            out_dim: number of units of output layer (number of classes)
            hidden_dims: number of units of hidden layers
            rev_scale: gradient reversal scale
        """
        super(AdversarialClassifier, self).__init__()

        self.gradient_rev = GradientReversal(rev_scale)

        in_sizes = [in_dim] + hidden_dims[:]
        out_sizes = hidden_dims[:] + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size, bias=True)
             for (in_size, out_size) in zip(in_sizes, out_sizes)])

        self.activations = [nn.ReLU()] * len(hidden_dims) + [nn.Softmax(dim=-1)]

    def forward(self, x, is_reversal=True):
        if is_reversal:
            x = self.gradient_rev(x)
        for (linear, activate) in zip(self.layers, self.activations):
            x = activate(linear(x))
        return x
    

class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x



