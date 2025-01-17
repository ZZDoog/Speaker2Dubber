import json
import math
import os
import random
import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False, diff_audio=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )

        if 'Grid' in self.preprocessed_path:
            self.is_grid = True
        else:
            self.is_grid = False


        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        # update the speaker id
        if preprocess_config["last_n_speaker"] != 0:
            for k, v in self.speaker_map.items():
                self.speaker_map[k] = v + preprocess_config["last_n_speaker"]
        # 
        if self.dataset_name == "MovieAnimation":
            if not self.is_grid:
                print("Reading emotions.json ...")
                with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
                    self.emotion_map = json.load(f)

        self.sort = sort
        self.drop_last = drop_last
        # 
        self.n_emotion = preprocess_config["preprocessing"]["emotion"]["n_emotion"]

        #
        self.padding = preprocess_config["Padding"]["preprocess"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]

        if "Denoise" in self.preprocessed_path:
            spk_path = os.path.join(
                self.preprocessed_path,
                "spk2", # "spk2", "spk2_vc" or "spk2_microsoft"
                "wav_22050_chenqi_clean_Denoise_version2_all-spk-{}.npy".format(basename),
                # "Micro-spk-{}.npy".format(basename),
            )
        else:
            spk_path = os.path.join(
                self.preprocessed_path,
                "spk2",
                "{}-spk-{}.npy".format(speaker, basename),
            )
        spk = np.load(spk_path)

        speaker_id = self.speaker_map[speaker]
        emo_path, emotion_id = None, None

        if self.dataset_name == "MovieAnimation":
            if self.is_grid:
                emotion_id = 0
                visual_embedding_path = os.path.join(
                        self.preprocessed_path,
                        "emos",
                        "{}-emo-{}.npy".format(speaker, basename.split('-')[-1]),
                    )
                if os.path.exists(visual_embedding_path):
                    emo = np.load(visual_embedding_path)
                else:
                    emo = np.zeros(256)

            else:
                emotion_id = self.emotion_map[basename]
                visual_embedding_path = os.path.join(
                    self.preprocessed_path,
                    "emos",
                    "{}-emo-{}.npy".format(speaker, basename),
                )
                emo = np.load(visual_embedding_path)

        elif self.dataset_name == "Chem":
            emotion_id = self.n_emotion
            emo_path = os.path.join(
                self.preprocessed_path,
                "spk2",
                "{}-spk-{}.npy".format(speaker, basename),
            )
            emo = np.load(emo_path)


        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        if not self.is_grid:
            feature_256_path = os.path.join(
                self.preprocessed_path,
                "VA_feature",
                "{}-feature-{}.npy".format(speaker, basename),
            )
        else:
            feature_256_path = os.path.join(
                self.preprocessed_path,
                "VA_feature",
                "{}-feature-{}.npy".format(speaker, basename.split('-')[-1]),
            )
        feature_256 = np.load(feature_256_path)

        # lip_embedding
        lip_embedding = None
        if self.dataset_name == "Chem":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_Chem_gray",
                "{}-face-{}.npy".format(speaker, basename),
            )
            lip_embedding = np.load(lip_embedding_path)

        elif self.dataset_name == "MovieAnimation":
            if not self.is_grid:
                lip_embedding_path = os.path.join(
                    self.preprocessed_path,
                    "extrated_embedding_V2C_gray",
                    "{}-face-{}.npy".format(speaker, basename),
                )
            else:
                lip_embedding_path = os.path.join(
                    self.preprocessed_path,
                    "extrated_embedding_Grid_152_gray",
                    "{}-face-{}.npy".format(speaker, basename.split('-')[-1]),
                )
            lip_embedding = np.load(lip_embedding_path)

        if not mel.shape[0] == duration.sum():
            print('{} duration mismatch, duration sum: {}, mel_len: {}'.format(basename, duration.sum(), mel.shape[0]))

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "spk": spk,
            "emotion": emotion_id,
            "emo": emo,
            "feature_256": feature_256,
            "lip_embedding": lip_embedding,
        }
        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        #
        spks = [data[idx]["spk"] for idx in idxs]
        emotions = [data[idx]["emotion"] for idx in idxs]
        emos = [data[idx]["emo"] for idx in idxs]
        feature_256 = [data[idx]["feature_256"] for idx in idxs]

        lip_embedding = [data[idx]["lip_embedding"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        lip_lens = np.array([lip_e.shape[0] for lip_e in lip_embedding])

        speakers = np.array(speakers) # 16
        texts = pad_1D(texts) # 16x273
        mels = pad_2D(mels) # 16x2379x80
        pitches = pad_1D(pitches) # 16x273
        energies = pad_1D(energies) # 16x273
        durations = pad_1D(durations) # 16x273 Beaca
        # Since we don't need to use Length Regulator, convert word length to mel-spectrum length
        # durations = np.array(durations)  # 16x273
        spks = np.array(spks) # 16x256
        emotions = np.array(emotions) # 16
        emos = np.array(emos) # 16x256
        feature_256 = pad_2D(feature_256)
        lip_embedding = pad_2D(lip_embedding)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
            spks,
            emotions,
            emos,
            feature_256,
            lip_lens,
            max(lip_lens),
            lip_embedding,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)
        # 
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.with_gt = preprocess_config["with_gt"]
        # 
        if self.dataset_name == "MovieAnimation":
            print("Reading emotions.json ...")
            with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
                self.emotion_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        if self.dataset_name == "MovieAnimation":
            emotion_id = self.emotion_map[basename]
        else:
            emotion_id = self.n_emotion
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        
        # ground-truth
        if self.with_gt:
            mel_path = os.path.join(
                self.preprocessed_path,
                "mel",
                "{}-mel-{}.npy".format(speaker, basename),
            )
            mel = np.load(mel_path)
            pitch_path = os.path.join(
                self.preprocessed_path,
                "pitch",
                "{}-pitch-{}.npy".format(speaker, basename),
            )
            pitch = np.load(pitch_path)
            energy_path = os.path.join(
                self.preprocessed_path,
                "energy",
                "{}-energy-{}.npy".format(speaker, basename),
            )
            energy = np.load(energy_path)
            duration_path = os.path.join(
                self.preprocessed_path,
                "duration",
                "{}-duration-{}.npy".format(speaker, basename),
            )
            duration = np.load(duration_path)
            return (basename, speaker_id, phone, raw_text, emotion_id, \
                mel, pitch, energy, duration)

        # return (basename, speaker_id, phone, raw_text)
        return (basename, speaker_id, phone, raw_text, emotion_id)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        emotions = np.array([d[4] for d in data])
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        if self.with_gt:
            mels = [d[5] for d in data]
            pitches = [d[6] for d in data]
            energies = [d[7] for d in data]
            durations = [d[8] for d in data]
            mel_lens = np.array([mel.shape[0] for mel in mels])
            # 
            mels = pad_2D(mels) # 16x2379x80
            pitches = pad_1D(pitches) # 16x273
            energies = pad_1D(energies) # 16x273
            durations = pad_1D(durations) # 16x273
            # 
            return ids, raw_texts, speakers, texts, text_lens, max(text_lens), emotions,\
            mels, pitches, energies, durations, mel_lens

        # return ids, raw_texts, speakers, texts, text_lens, max(text_lens)
        return ids, raw_texts, speakers, texts, text_lens, max(text_lens), emotions


class PretrainDataset(Dataset):
    
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False, diff_audio=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )

        if 'Grid' in self.preprocessed_path:
            self.is_grid = True
        else:
            self.is_grid = False


        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        # update the speaker id
        if preprocess_config["last_n_speaker"] != 0:
            for k, v in self.speaker_map.items():
                self.speaker_map[k] = v + preprocess_config["last_n_speaker"]
        # 
        if self.dataset_name == "MovieAnimation":
            if not self.is_grid:
                print("Reading emotions.json ...")
                with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
                    self.emotion_map = json.load(f)

        self.sort = sort
        self.drop_last = drop_last
        # 
        self.n_emotion = preprocess_config["preprocessing"]["emotion"]["n_emotion"]

        #
        self.padding = preprocess_config["Padding"]["preprocess"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]

        if "Denoise" in self.preprocessed_path:
            spk_path = os.path.join(
                self.preprocessed_path,
                "spk2",
                "wav_22050_chenqi_clean_Denoise_version2_all-spk-{}.npy".format(basename),
            )
        else:
            spk_path = os.path.join(
                self.preprocessed_path,
                "spk2",
                "{}-spk-{}.npy".format(speaker, basename),
            )
        spk = np.load(spk_path)

        speaker_id = self.speaker_map[speaker]
        emo_path, emotion_id = None, None
        emo = None

        # if self.dataset_name == "MovieAnimation":
        #     if self.is_grid:
        #         emotion_id = 0
        #         visual_embedding_path = os.path.join(
        #                 self.preprocessed_path,
        #                 "emos",
        #                 "{}-emo-{}.npy".format(speaker, basename.split('-')[-1]),
        #             )
        #         if os.path.exists(visual_embedding_path):
        #             emo = np.load(visual_embedding_path)
        #         else:
        #             emo = np.zeros(256)

        #     else:
        #         emotion_id = self.emotion_map[basename]
        #         visual_embedding_path = os.path.join(
        #             self.preprocessed_path,
        #             "emos",
        #             "{}-emo-{}.npy".format(speaker, basename),
        #         )
        #         emo = np.load(visual_embedding_path)

        # elif self.dataset_name == "Chem":
        #     emotion_id = self.n_emotion
        #     emo_path = os.path.join(
        #         self.preprocessed_path,
        #         "spk2",
        #         "{}-spk-{}.npy".format(speaker, basename),
        #     )
        #     emo = np.load(emo_path)


        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        if not mel.shape[0] == duration.sum():
            print('{} duration mismatch, duration sum: {}, mel_len: {}'.format(basename, duration.sum(), mel.shape[0]))

        feature_256 = None
        # if not self.is_grid:
        #     feature_256_path = os.path.join(
        #         self.preprocessed_path,
        #         "VA_feature",
        #         "{}-feature-{}.npy".format(speaker, basename),
        #     )
        # else:
        #     feature_256_path = os.path.join(
        #         self.preprocessed_path,
        #         "VA_feature",
        #         "{}-feature-{}.npy".format(speaker, basename.split('-')[-1]),
        #     )
        # feature_256 = np.load(feature_256_path)

        # lip_embedding
        lip_embedding = None
        # if self.dataset_name == "Chem":
        #     lip_embedding_path = os.path.join(
        #         self.preprocessed_path,
        #         "extrated_embedding_Chem_gray",
        #         "{}-face-{}.npy".format(speaker, basename),
        #     )
        #     lip_embedding = np.load(lip_embedding_path)

        # elif self.dataset_name == "MovieAnimation":
        #     if not self.is_grid:
        #         lip_embedding_path = os.path.join(
        #             self.preprocessed_path,
        #             "extrated_embedding_V2C_gray",
        #             "{}-face-{}.npy".format(speaker, basename),
        #         )
        #     else:
        #         lip_embedding_path = os.path.join(
        #             self.preprocessed_path,
        #             "extrated_embedding_Grid_152_gray",
        #             "{}-face-{}.npy".format(speaker, basename.split('-')[-1]),
        #         )
        #     lip_embedding = np.load(lip_embedding_path)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "spk": spk,
            "emotion": emotion_id, # None
            "emo": emo, # None
            "feature_256": feature_256, # None
            "lip_embedding": lip_embedding, # None
        }
        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        #
        spks = [data[idx]["spk"] for idx in idxs]
        # emotions = [data[idx]["emotion"] for idx in idxs]
        # emos = [data[idx]["emo"] for idx in idxs]
        # feature_256 = [data[idx]["feature_256"] for idx in idxs]

        # lip_embedding = [data[idx]["lip_embedding"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        # lip_lens = np.array([lip_e.shape[0] for lip_e in lip_embedding])

        speakers = np.array(speakers) # 16
        texts = pad_1D(texts) # 16x273
        mels = pad_2D(mels) # 16x2379x80
        pitches = pad_1D(pitches) # 16x273
        energies = pad_1D(energies) # 16x273
        durations = pad_1D(durations) # 16x273 Beaca
        # Since we don't need to use Length Regulator, convert word length to mel-spectrum length
        # durations = np.array(durations)  # 16x273
        spks = np.array(spks) # 16x256
        # emotions = np.array(emotions) # 16
        # emos = np.array(emos) # 16x256
        # feature_256 = pad_2D(feature_256)
        # lip_embedding = pad_2D(lip_embedding)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
            spks,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output
    


class Dataset_setting2(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False, diff_audio=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.text, self.speaker, self.ref_name, self.ref_text = self.process_meta(
            filename
        )

        if 'Grid' in self.preprocessed_path:
            self.is_grid = True
        else:
            self.is_grid = False


        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        # update the speaker id
        if preprocess_config["last_n_speaker"] != 0:
            for k, v in self.speaker_map.items():
                self.speaker_map[k] = v + preprocess_config["last_n_speaker"]
        # 
        if self.dataset_name == "MovieAnimation":
            if not self.is_grid:
                print("Reading emotions.json ...")
                with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
                    self.emotion_map = json.load(f)

        self.sort = sort
        self.drop_last = drop_last
        # 
        self.n_emotion = preprocess_config["preprocessing"]["emotion"]["n_emotion"]

        #
        self.padding = preprocess_config["Padding"]["preprocess"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        ref_basename = self.ref_name[idx]

        if "Denoise" in self.preprocessed_path:
            spk_path = os.path.join(
                self.preprocessed_path,
                "spk2", # "spk2", "spk2_vc" or "spk2_microsoft"
                "wav_22050_chenqi_clean_Denoise_version2_all-spk-{}.npy".format(ref_basename),
                # "Micro-spk-{}.npy".format(ref_basename),
            )
        else:
            spk_path = os.path.join(
                self.preprocessed_path,
                "spk2",
                "{}-spk-{}.npy".format(speaker, ref_basename),
            )
        spk = np.load(spk_path)

        speaker_id = self.speaker_map[speaker]
        emo_path, emotion_id = None, None

        if self.dataset_name == "MovieAnimation":
            if self.is_grid:
                emotion_id = 0
                visual_embedding_path = os.path.join(
                        self.preprocessed_path,
                        "emos",
                        "{}-emo-{}.npy".format(speaker, basename.split('-')[-1]),
                    )
                if os.path.exists(visual_embedding_path):
                    emo = np.load(visual_embedding_path)
                else:
                    emo = np.zeros(256)

            else:
                emotion_id = self.emotion_map[basename]
                visual_embedding_path = os.path.join(
                    self.preprocessed_path,
                    "emos",
                    "{}-emo-{}.npy".format(speaker, basename),
                )
                emo = np.load(visual_embedding_path)

        elif self.dataset_name == "Chem":
            emotion_id = self.n_emotion
            emo_path = os.path.join(
                self.preprocessed_path,
                "spk2",
                "{}-spk-{}.npy".format(speaker, basename),
            )
            emo = np.load(emo_path)


        raw_text = self.text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, ref_basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        if not self.is_grid:
            feature_256_path = os.path.join(
                self.preprocessed_path,
                "VA_feature",
                "{}-feature-{}.npy".format(speaker, basename),
            )
        else:
            feature_256_path = os.path.join(
                self.preprocessed_path,
                "VA_feature",
                "{}-feature-{}.npy".format(speaker, basename.split('-')[-1]),
            )
        feature_256 = np.load(feature_256_path)

        # lip_embedding
        lip_embedding = None
        if self.dataset_name == "Chem":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_Chem_gray",
                "{}-face-{}.npy".format(speaker, basename),
            )
            lip_embedding = np.load(lip_embedding_path)

        elif self.dataset_name == "MovieAnimation":
            if not self.is_grid:
                lip_embedding_path = os.path.join(
                    self.preprocessed_path,
                    "extrated_embedding_V2C_gray",
                    "{}-face-{}.npy".format(speaker, basename),
                )
            else:
                lip_embedding_path = os.path.join(
                    self.preprocessed_path,
                    "extrated_embedding_Grid_152_gray",
                    "{}-face-{}.npy".format(speaker, basename.split('-')[-1]),
                )
            lip_embedding = np.load(lip_embedding_path)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "spk": spk,
            "emotion": emotion_id,
            "emo": emo,
            "feature_256": feature_256,
            "lip_embedding": lip_embedding,
        }
        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            text = []
            speaker = []
            ref_name = []
            ref_text = []
            for line in f.readlines():
                n, t, s, rn, rt = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                ref_name.append(rn)
                ref_text.append(rt)
            return name, text, speaker, ref_name, ref_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        #
        spks = [data[idx]["spk"] for idx in idxs]
        emotions = [data[idx]["emotion"] for idx in idxs]
        emos = [data[idx]["emo"] for idx in idxs]
        feature_256 = [data[idx]["feature_256"] for idx in idxs]

        lip_embedding = [data[idx]["lip_embedding"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        lip_lens = np.array([lip_e.shape[0] for lip_e in lip_embedding])

        speakers = np.array(speakers) # 16
        texts = pad_1D(texts) # 16x273
        mels = pad_2D(mels) # 16x2379x80
        pitches = pad_1D(pitches) # 16x273
        energies = pad_1D(energies) # 16x273
        durations = pad_1D(durations) # 16x273 Beaca
        # Since we don't need to use Length Regulator, convert word length to mel-spectrum length
        # durations = np.array(durations)  # 16x273
        spks = np.array(spks) # 16x256
        emotions = np.array(emotions) # 16
        emos = np.array(emos) # 16x256
        feature_256 = pad_2D(feature_256)
        lip_embedding = pad_2D(lip_embedding)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
            spks,
            emotions,
            emos,
            feature_256,
            lip_lens,
            max(lip_lens),
            lip_embedding,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output
    

class Dataset_setting3(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False, diff_audio=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.text, self.speaker, self.ref_name, self.ref_text = self.process_meta(
            filename
        )

        if 'Grid' in self.preprocessed_path:
            self.is_grid = True
        else:
            self.is_grid = False


        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        # update the speaker id
        if preprocess_config["last_n_speaker"] != 0:
            for k, v in self.speaker_map.items():
                self.speaker_map[k] = v + preprocess_config["last_n_speaker"]
        # 
        if self.dataset_name == "MovieAnimation":
            if not self.is_grid:
                print("Reading emotions.json ...")
                with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
                    self.emotion_map = json.load(f)

        self.sort = sort
        self.drop_last = drop_last
        # 
        self.n_emotion = preprocess_config["preprocessing"]["emotion"]["n_emotion"]

        #
        self.padding = preprocess_config["Padding"]["preprocess"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        ref_basename = self.ref_name[idx]

        if "Denoise" in self.preprocessed_path:
            spk_path = os.path.join(
                "/data1/home/zhangzhedong/preprocessed_data/Grid_Wav_22050_Abs_Feature",
                "spk2",
                "{}-spk-{}.npy".format(ref_basename.split('-')[0], ref_basename),
            )
        else:
            spk_path = os.path.join(
                self.preprocessed_path,
                "spk2",
                "{}-spk-{}.npy".format(speaker, ref_basename),
            )
        spk = np.load(spk_path)

        speaker_id = self.speaker_map[speaker]
        emo_path, emotion_id = None, None

        if self.dataset_name == "MovieAnimation":
            if self.is_grid:
                emotion_id = 0
                visual_embedding_path = os.path.join(
                        self.preprocessed_path,
                        "emos",
                        "{}-emo-{}.npy".format(speaker, basename.split('-')[-1]),
                    )
                if os.path.exists(visual_embedding_path):
                    emo = np.load(visual_embedding_path)
                else:
                    emo = np.zeros(256)

            else:
                emotion_id = self.emotion_map[basename]
                visual_embedding_path = os.path.join(
                    self.preprocessed_path,
                    "emos",
                    "{}-emo-{}.npy".format(speaker, basename),
                )
                emo = np.load(visual_embedding_path)

        elif self.dataset_name == "Chem":
            emotion_id = self.n_emotion
            emo_path = os.path.join(
                self.preprocessed_path,
                "spk2",
                "{}-spk-{}.npy".format(speaker, basename),
            )
            emo = np.load(emo_path)


        raw_text = self.text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        if not self.is_grid:
            feature_256_path = os.path.join(
                self.preprocessed_path,
                "VA_feature",
                "{}-feature-{}.npy".format(speaker, basename),
            )
        else:
            feature_256_path = os.path.join(
                self.preprocessed_path,
                "VA_feature",
                "{}-feature-{}.npy".format(speaker, basename.split('-')[-1]),
            )
        feature_256 = np.load(feature_256_path)

        # lip_embedding
        lip_embedding = None
        if self.dataset_name == "Chem":
            lip_embedding_path = os.path.join(
                self.preprocessed_path,
                "extrated_embedding_Chem_gray",
                "{}-face-{}.npy".format(speaker, basename),
            )
            lip_embedding = np.load(lip_embedding_path)

        elif self.dataset_name == "MovieAnimation":
            if not self.is_grid:
                lip_embedding_path = os.path.join(
                    self.preprocessed_path,
                    "extrated_embedding_V2C_gray",
                    "{}-face-{}.npy".format(speaker, basename),
                )
            else:
                lip_embedding_path = os.path.join(
                    self.preprocessed_path,
                    "extrated_embedding_Grid_152_gray",
                    "{}-face-{}.npy".format(speaker, basename.split('-')[-1]),
                )
            lip_embedding = np.load(lip_embedding_path)

        basename = "{}_{}".format(basename, ref_basename)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "spk": spk,
            "emotion": emotion_id,
            "emo": emo,
            "feature_256": feature_256,
            "lip_embedding": lip_embedding,
        }
        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            text = []
            speaker = []
            ref_name = []
            ref_text = []
            for line in f.readlines():
                n, t, rn, rt = line.strip("\n").split("|")
                name.append(n)
                text.append(t)
                speaker.append(n.split("_")[0])
                ref_name.append(rn)
                ref_text.append(rt)
            return name, text, speaker, ref_name, ref_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        #
        spks = [data[idx]["spk"] for idx in idxs]
        emotions = [data[idx]["emotion"] for idx in idxs]
        emos = [data[idx]["emo"] for idx in idxs]
        feature_256 = [data[idx]["feature_256"] for idx in idxs]

        lip_embedding = [data[idx]["lip_embedding"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        lip_lens = np.array([lip_e.shape[0] for lip_e in lip_embedding])

        speakers = np.array(speakers) # 16
        texts = pad_1D(texts) # 16x273
        mels = pad_2D(mels) # 16x2379x80
        pitches = pad_1D(pitches) # 16x273
        energies = pad_1D(energies) # 16x273
        durations = pad_1D(durations) # 16x273 Beaca
        # Since we don't need to use Length Regulator, convert word length to mel-spectrum length
        # durations = np.array(durations)  # 16x273
        spks = np.array(spks) # 16x256
        emotions = np.array(emotions) # 16
        emos = np.array(emos) # 16x256
        feature_256 = pad_2D(feature_256)
        lip_embedding = pad_2D(lip_embedding)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
            spks,
            emotions,
            emos,
            feature_256,
            lip_lens,
            max(lip_lens),
            lip_embedding,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output