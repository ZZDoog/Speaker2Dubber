import torch
import torch.nn as nn
import torch.nn.functional as F


class Speaker2Dubber_Loss(nn.Module):
    """ Speaker2Dubber Loss """
    def __init__(self, preprocess_config, model_config):
        super(Speaker2Dubber_Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
        self.loss_model = model_config["loss_function"]["model"]
        self.use_speaker_loss = model_config["loss_function"]["speaker_loss"]
        self.mae_loss = nn.L1Loss()
        self.CTC_criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='sum').cuda()
        self.mse_loss = nn.MSELoss()
        self.model_config = model_config
        self.preprocess_config = preprocess_config

    def weights_nonzero_speech(self, target):
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)
    def mse_loss_v2c(self, decoder_output, target):
        assert decoder_output.shape == target.shape
        mse_loss = F.mse_loss(decoder_output, target, reduction='none')
        weights = self.weights_nonzero_speech(target)
        mse_loss = (mse_loss * weights).sum() / weights.sum()
        return mse_loss
    def forward(self, inputs, predictions):
        (
            texts, 
            src_lens,
            _,
            mel_targets,
            mel_lens,
            max_mel_len,
            pitch_targets,
            energy_targets,
            duration_targets,
            _,
            emo_class_target,
            _,
            _,
            lip_lens,
            _,
            _,
        ) = inputs[3:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            SIMILARITY_ALL,
            src_masks,
            mel_masks,
            _,
            # mel_lens,
            _,
            CTC_ALL,
            max_src_len,
        ) = predictions


        
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        
        # We directly use the ground truth duration in the pretrain.
        if not self.model_config['train_mode'] == 'pretrain':
            log_duration_targets = torch.log(duration_targets.float() + 1)  # log_duration_targets.requires_grad = False
        else:
            log_duration_targets = duration_targets.float()
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False
        log_duration_targets.requires_grad = False
        ctc_pred_MDA_video =  CTC_ALL[0]
        ctc_pred_mel =  CTC_ALL[1]

        if not self.model_config['train_mode'] == 'pretrain':
            similarity = SIMILARITY_ALL[0]
            gt_similarity = SIMILARITY_ALL[1]
            duration_predictions = SIMILARITY_ALL[2]
            # duration_loss = torch.sum(gt_similarity - gt_similarity*similarity)
            # duration_loss = torch.tensor([0]).cuda()
            duration_loss = torch.abs(gt_similarity-similarity).mean()
            
        else:
            duration_loss = torch.tensor([0]).cuda()

        if self.model_config['train_mode'] == 'pretrain':
            CTC_loss_MDA_video = torch.tensor([0]).cuda()
            CTC_loss_MEL = torch.tensor([0]).cuda()
        else:
            CTC_loss_MDA_video = self.CTC_criterion(ctc_pred_MDA_video.transpose(0, 1).log_softmax(2), texts, lip_lens, src_lens) / texts.shape[0]
            CTC_loss_MEL = self.CTC_criterion(ctc_pred_mel.transpose(0, 1).log_softmax(2), texts, mel_lens, src_lens) / texts.shape[0]
        mse_loss_v2c1 = self.mse_loss_v2c(mel_predictions, mel_targets)
        mse_loss_v2c2 = self.mse_loss_v2c(postnet_mel_predictions, mel_targets)        
        pitch_predictions = pitch_predictions.masked_select(src_masks)
        pitch_targets = pitch_targets.masked_select(src_masks)
        energy_predictions = energy_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)
        duration_predictions = duration_predictions.masked_select(src_masks)
        energy_targets = energy_targets.masked_select(src_masks)
        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))
        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss_mse = self.mse_loss(duration_predictions, log_duration_targets)
        pitch_loss_mae = self.mae_loss(pitch_predictions, pitch_targets)
        energy_loss_mae = self.mae_loss(energy_predictions, energy_targets)
        duration_loss_mae = self.mse_loss(duration_predictions, log_duration_targets)

        if self.loss_model == "Chem":
            total_loss = (
                    mel_loss + postnet_mel_loss + pitch_loss + energy_loss + duration_loss  +
                    pitch_loss_mae + energy_loss_mae + mse_loss_v2c1 + mse_loss_v2c2
                    + 0.01*CTC_loss_MDA_video + 0.01*CTC_loss_MEL
            )
            return (
                total_loss,
                mel_loss,
                postnet_mel_loss,
                pitch_loss,
                energy_loss,
                pitch_loss_mae,
                energy_loss_mae,
                mse_loss_v2c1,
                mse_loss_v2c2,
                0.01*CTC_loss_MDA_video,
                0.01*CTC_loss_MEL,
                duration_loss,
                torch.Tensor([0])
            )
        if self.loss_model == "V2C":

            emo_loss = torch.tensor([0])
            total_loss = (
                            mel_loss + postnet_mel_loss + pitch_loss + energy_loss + 10 * duration_loss + duration_loss_mse + duration_loss_mae +
                            pitch_loss_mae + energy_loss_mae + mse_loss_v2c1 + mse_loss_v2c2
                            + 0.01*CTC_loss_MDA_video + 0.01*CTC_loss_MEL
                        )
            return (
                total_loss,
                mel_loss,
                postnet_mel_loss,
                pitch_loss,
                energy_loss,
                pitch_loss_mae,
                energy_loss_mae,
                mse_loss_v2c1,
                mse_loss_v2c2,
                0.05*emo_loss,
                0.01*CTC_loss_MDA_video,
                0.01*CTC_loss_MEL,
                torch.Tensor([0]), # Speaker Loss for log and Tensorboard
                10* duration_loss,
                duration_loss_mae + duration_loss_mae
            )




