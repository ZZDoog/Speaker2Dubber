import torch
import torch.nn as nn
import numpy as np

import transformer.Constants as Constants
from .Layers import FFTBlock, FFTBlock_style
from text.symbols import symbols
# from style_models.Modules import Mish, LinearNorm

def get_sinusoid_encoding_table_512(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Lip_Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super(Lip_Encoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = 512
        n_layers = config["Lip_transformer"]["encoder_layer"]
        n_head = config["Lip_transformer"]["encoder_head"]
        d_k = d_v = (
            config["Lip_transformer"]["encoder_hidden"]
            // config["Lip_transformer"]["encoder_head"]
        )
        d_model = config["Lip_transformer"]["encoder_hidden"]
        d_inner = config["Lip_transformer"]["conv_filter_size"]
        kernel_size = config["Lip_transformer"]["conv_kernel_size"]
        dropout = config["Lip_transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table_512(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        # self.position_enc = nn.Parameter(
        #     get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
        #     requires_grad=False,
        # )
        self.fc_out = nn.Linear(self.d_model, 256)

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # # -- Forward
        # if not self.training and src_seq.shape[1] > self.max_seq_len:
        #     enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
        #         src_seq.shape[1], self.d_model
        #     )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
        #         src_seq.device
        #     )
        # else:
        #     enc_output = self.src_word_emb(src_seq) + self.position_enc[
        #         :, :max_len, :
        #     ].expand(batch_size, -1, -1)'

        enc_output = src_seq + self.position_enc[
                                                  :, :max_len, :
                                                  ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        enc_output = self.fc_out(enc_output)

        return enc_output


class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super(Encoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]
        style_dim = config["transformer"]["style_dim"]
        self.use_stylefft = config['transformer']['style_FFT']
        self.use_face_id = config['transformer']['face_id']

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.mid_layer = n_layers / 2

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        if self.use_stylefft == True:
            self.layer_stack = nn.ModuleList(
                [
                    FFTBlock_style(
                        d_model, n_head, d_k, d_v, d_inner, kernel_size, style_dim, dropout=dropout
                    )
                    for _ in range(n_layers)
                ]
            )

        else:
            self.layer_stack = nn.ModuleList(
                [
                    FFTBlock(
                        d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                    )
                    for _ in range(n_layers)
                ]
            )

    def forward(self, src_seq, style_vector, face_vector, mask, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        if self.use_stylefft == True:

            # Use the EmoFAN feature as the addition identity information
            if self.use_face_id == True:
                
                layer_idx = 0
                for enc_layer in self.layer_stack:
                    # Use face embedding in front half layer
                    if layer_idx < self.mid_layer:
                        enc_output, enc_slf_attn = enc_layer(
                            enc_output, face_vector, mask=mask, slf_attn_mask=slf_attn_mask
                        )
                        if return_attns:
                            enc_slf_attn_list += [enc_slf_attn]
                    
                    # Use voice embedding in back half layer
                    else:
                        enc_output, enc_slf_attn = enc_layer(
                        enc_output, style_vector, mask=mask, slf_attn_mask=slf_attn_mask
                        )
                    if return_attns:
                        enc_slf_attn_list += [enc_slf_attn]
                    layer_idx += 1

            # Use the Voice feature only as identity information
            else:
                for enc_layer in self.layer_stack:
                    enc_output, enc_slf_attn = enc_layer(
                        enc_output, style_vector, mask=mask, slf_attn_mask=slf_attn_mask
                    )
                    if return_attns:
                        enc_slf_attn_list += [enc_slf_attn]

        else:
            for enc_layer in self.layer_stack:
                enc_output, enc_slf_attn = enc_layer(
                    enc_output, mask=mask, slf_attn_mask=slf_attn_mask
                )
                if return_attns:
                    enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]
        style_dim = config["transformer"]["style_dim"]
        self.use_stylefft = config['transformer']['style_FFT']
        self.use_face_id = config['transformer']['face_id']

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.mid_layer = n_layers / 2

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        if self.use_stylefft == True:
                self.layer_stack = nn.ModuleList(
                    [
                        FFTBlock_style(
                            d_model, n_head, d_k, d_v, d_inner, kernel_size, style_dim, dropout=dropout
                        )
                        for _ in range(n_layers)
                    ]
                )

        else:
            self.layer_stack = nn.ModuleList(
                [
                    FFTBlock(
                        d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                    )
                    for _ in range(n_layers)
                ]
            )

    def forward(self, enc_seq, mask, style_vector, face_vector, return_attns=False):

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        if self.use_stylefft == True:

            if self.use_face_id:
                
                layer_idx = 0
                for dec_layer in self.layer_stack:
                    # Use face embedding in front half layer
                    if layer_idx < self.mid_layer:
                        dec_output, dec_slf_attn = dec_layer(
                            dec_output, face_vector, mask=mask, slf_attn_mask=slf_attn_mask
                        )
                        if return_attns:
                            dec_slf_attn_list += [dec_slf_attn]
                    
                    # Use voice embedding in back half layer
                    else:
                        dec_output, dec_slf_attn = dec_layer(
                        dec_output, style_vector, mask=mask, slf_attn_mask=slf_attn_mask
                        )
                    if return_attns:
                        dec_slf_attn_list += [dec_slf_attn]
                    layer_idx += 1

            else:   

                for dec_layer in self.layer_stack:
                    dec_output, enc_slf_attn = dec_layer(
                        dec_output, style_vector, mask=mask, slf_attn_mask=slf_attn_mask
                    )
                    if return_attns:
                        dec_slf_attn_list += [enc_slf_attn]

        else:
            for dec_layer in self.layer_stack:
                dec_output, enc_slf_attn = dec_layer(
                    dec_output, mask=mask, slf_attn_mask=slf_attn_mask
                )
                if return_attns:
                    enc_slf_attn_list += [enc_slf_attn]

        return dec_output, mask
