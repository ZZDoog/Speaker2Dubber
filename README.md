# Speaker2Dubber

[ACM MM24] Official implementation of paper "From Speaker to Dubber: Movie Dubbing with Prosody and Duration Consistency Learning"

<img width="550" alt="image" src="Figs/Intro.png">

<img width="1000" alt="image" src="Figs\Speaker2Dubber.png">

# News
- [Feb. 2025] Our new paper on Movie Dubbing task: *Prosody-Enhanced Acoustic Pre-training and Acoustic-Disentangled Prosody Adapting for Movie Dubbing* is accepted by **CVPR2025** with better performance and higher acoustic quality. The [paper](https://arxiv.org/abs/2503.12042), [code](https://github.com/ZZDoog/ProDubber) and [demo](https://zzdoog.github.io/ProDubber/) are all available. 

## 🗒 TODOs

- [x] Release Speaker2Dubber‘s demo at [here](https://speaker2dubber.github.io/).

- [x] Release the generated test set at [Google Drive](https://drive.google.com/file/d/1FJsGIVLqoQKqnzfnhKBlx9_dek0V7Iiu/view?usp=drive_link) or [Baidu Cloud Drive](https://pan.baidu.com/s/1nxKcBbyCnGPSyz9cpnjW9Q) (Password: mm24).

- [x] Release Speaker2Dubber's train and inference code.

- [x] Release Speaker2Dubber's model.

- [x] Update README.md (How to use).

- [x] Release the pre-trained checkpoints on [Baidu Cloud Drive](https://pan.baidu.com/s/1T0ndIerg1jzPYi6L_kWO8A) (Password: 4ud3) and [Google Drive](https://drive.google.com/drive/folders/1VvGxkSYQ4yQFl4rs33lTVb5MBZXtXJFi?usp=sharing).

## 🌼 Environment

Our python version is ```3.8.18``` and cuda version ```11.5```. It's possible to have another compatible version. 
Both training and inference are implemented with PyTorch on a
GeForce RTX 4090 GPU.

```
conda create -n speaker2dubber python=3.8.18
conda activate speaker2dubber
pip install -r requirements.txt
pip install git+https://github.com/resemble-ai/monotonic_align.git
```

## 🔧 Training

You need repalce tha path in ```preprocess_config``` to your preprocssed data path  (see "config/MovieAnimation/preprocess.yaml") to you own path and run:

```
python train.py -p config/MovieAnimation/preprocess.yaml -m config/MovieAnimation/model.yaml -t config/MovieAnimation/train.yaml
```

## ✍ Inference

There is three setting in V2C task.

```
python Synthesis.py --restore_step 50000 -s 1 -n 'YOUR_EXP_NAME'
```
```
python Synthesis.py --restore_step 50000 -s 2 -n 'YOUR_EXP_NAME'
```
```
python Synthesis.py --restore_step 50000 -s 3 -n 'YOUR_EXP_NAME'
```
The `s` denotes the inference setting (`1` for setting1 which use gt audio as reference audio, `2` for setting2 which use another audio from target speaker as reference audio, `3` for zero shot setting which use reference audio from unseen dataset as refernce audio.)

## 📊 Dataset

- GRID ([BaiduDrive](https://pan.baidu.com/s/1E4cPbDvw_Zfk3_F8qoM7JA) (code: GRID) /  [GoogleDrive](https://drive.google.com/drive/folders/1_z51hy6H3K4kyHy-MXtMfo2Py6edpscE?usp=drive_link))
- V2C-Animation dataset ([BaiduDrive]( https://pan.baidu.com/s/12hEFbXwpv4JscG3tUffjbA) (code: k9mb) / [GoogleDrive](https://drive.google.com/drive/folders/11WhRulJd23XzeuWmUVay5carpudGq3ig?usp=drive_link))



## 🙏 Acknowledgments
We would like to thank the authors of previous related projects for generously sharing their code and insights: [HPMDubbing](https://github.com/GalaxyCong/HPMDubbing), [Monotonic Align](https://github.com/resemble-ai/monotonic_align), [StyleSpeech](https://github.com/keonlee9420/StyleSpeech), [FastSpeech2](https://github.com/ming024/FastSpeech2), [V2C](https://github.com/chenqi008/V2C), [StyleDubber](https://github.com/GalaxyCong/StyleDubber), [PL-BERT](https://github.com/yl4579/PL-BERT), and [HiFi-GAN](https://github.com/jik876/hifi-gan).


## 🤝 Ciation
If you find our work useful, please consider citing:
```
@inproceedings{zhang-etal-2024-speaker2dubber,
  author       = {Zhedong Zhang and
                  Liang Li and
                  Gaoxiang Cong and
                  Haibing Yin and
                  Yuhan Gao and
                  Chenggang Yan and
                  Anton van den Hengel and
                  Yuankai Qi},
  title        = {From Speaker to Dubber: Movie Dubbing with Prosody and Duration Consistency
                  Learning},
  booktitle    = {Proceedings of the 32nd {ACM} International Conference on Multimedia,
                  {MM} 2024, Melbourne, VIC, Australia, 28 October 2024 - 1 November
                  2024},
  pages        = {7523--7532},
  publisher    = {{ACM}},
  year         = {2024},
}
```