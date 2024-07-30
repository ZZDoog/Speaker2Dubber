# Speaker2Dubber

[ACM MM24] Official implementation of paper "From Speaker to Dubber: Movie Dubbing with Prosody and Duration Consistency Learning"

<img width="550" alt="image" src="Figs/Intro.png">

<img width="1000" alt="image" src="Figs\Speaker2Dubber.png">

## 🗒 TODOs

- [x] Release Speaker2Dubber‘s demo at [here](https://speaker2dubber.github.io/).

- [x] Release the generated test set at [Google Drive](https://drive.google.com/file/d/1FJsGIVLqoQKqnzfnhKBlx9_dek0V7Iiu/view?usp=drive_link) or [Baidu Cloud Drive](https://pan.baidu.com/s/1nxKcBbyCnGPSyz9cpnjW9Q) (Password: mm24).

- [x] Release Speaker2Dubber's train and inference code.

- [ ] Release Speaker2Dubber's model and checkpoints.

- [ ] Update README.md (How to use).

## 🌼 Environment

Our python version is ```3.8.18``` and cuda version ```11.5```. It's possible to have other compatible version. 
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

- GRID ([BaiduDrive](https://pan.baidu.com/s/1E4cPbDvw_Zfk3_F8qoM7JA) (code: GRID) / GoogleDrive)
- V2C-Animation dataset (chenqi-Denoise2) 



## 🙏 Acknowledgments
We would like to thank the authors of previous related projects for generously sharing their code and insights: [HPMDubbing](https://github.com/GalaxyCong/HPMDubbing), [Monotonic Align](https://github.com/resemble-ai/monotonic_align), [StyleSpeech](https://github.com/keonlee9420/StyleSpeech), [FastSpeech2](https://github.com/ming024/FastSpeech2), [V2C](https://github.com/chenqi008/V2C), and [HiFi-GAN](https://github.com/jik876/hifi-gan).
