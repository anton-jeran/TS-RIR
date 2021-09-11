# Related Works

1) [**IR-GAN: Room Impulse Response Generator for Far-field Speech Recognition (INTERSPEECH2021)**](https://github.com/anton-jeran/IR-GAN)


# TS-RIR (Accepted to IEEE ASRU 2021)

This is the official implementation of **TS-RIRGAN**. We started our implementation from [**WaveGAN**](https://github.com/chrisdonahue/wavegan). TS-RIRGAN is a one-dimensional CycleGAN that takes synthetic RIRs as raw waveform audio and translates it into real RIRs. Our network architecture is shown below.



![Architecture-1.png](https://github.com/anton-jeran/TS-RIR/blob/main/images/Architecture-1.png)

You can find more details about our implementation from [**TS-RIR: Translated synthetic room impulse responses for speech augmentation**](https://arxiv.org/pdf/2103.16804v2.pdf).


## Requirements

```
tensorflow-gpu==1.12.0
scipy==1.0.0
matplotlib==3.0.2
librosa==0.6.2
ffmpeg ==4.2.1
cuda ==9.0.176
cudnn ==7.6.5
```

## Datasets

In order to train **TS-RIRGAN** to translate Synthetic RIRs to Real RIRs, download the RIRs from [**IRs_for_GAN**](https://drive.google.com/file/d/1ivj_UZ5j5inAZwsDTCQ6jEvI5JDtwH_2/view?usp=sharing). Unzip **IRs_for_GAN** directory inside **TS-RIR** folder.

This folder contains Synthetic RIRs generated using [**Geometric Acoustic Simulator**](https://github.com/RoyJames/pygsound) and Real RIRs from [**BUT ReverbDB**](https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database) dataset.et.

## Translate Synthetic RIRs to Real RIRs using the trained model

Download all the [**MODEL FILES**](https://drive.google.com/file/d/1fdAaIkvFbky-Xf7iuYCFa87nWpSaI1Ow/view?usp=sharing) and move all the files to the **generator** folder. Create a similar structure as the dataset inside the **generator** folder. You can convert **Synthetic RIRs** to **Real RIRs** by running the following command inside the **generator** folder.


```
export CUDA_VISIBLE_DEVICES=1
python3 generator.py --data1_dir ../IRs_for_GAN/Real_IRs/train --data1_first_slice --data1_pad_end --data1_fast_wav --data2_dir ../IRs_for_GAN/Synthetic_IRs/train --data2_first_slice --data2_pad_end --data2_fast_wav
```

## Training TS-RIRGAN

Run following command to train TS-RIRGAN.

```
export CUDA_VISIBLE_DEVICES=0
python3 train_TSRIRgan.py train ./train --data1_dir ./IRs_for_GAN/Real_IRs/train --data1_first_slice --data1_pad_end --data1_fast_wav --data2_dir ./IRs_for_GAN/Synthetic_IRs/train --data2_first_slice --data2_pad_end --data2_fast_wav
```

To backup the mode for every 1 hour, run the follwing command


```
export CUDA_VISIBLE_DEVICES=1
python3 backup.py ./train 60
```

To monitor the training using tensorboard, run the followind command

```
tensorboard --logdir=./train
```

## Results
The figure below shows Synthetic RIR generated using [**Geometric Acoustic Simulator**](https://github.com/RoyJames/pygsound), Synthetic RIR translated to Real RIR using our [**TS-RIRGAN**](https://arxiv.org/pdf/2103.16804v2.pdf) and a Real RIR from [**BUT ReverbDB**](https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database) dataset. Please note that there is no one-to-one relationship between Synthetic RIR and Real RIR from **BUT ReverbDB**. We show an example of Real RIR to compare the energy distribution of our translated RIR with the energy distribution of Real RIR.
![spectrogram.png](https://github.com/anton-jeran/TS-RIR/blob/main/images/spectrogram.png)

## Output

You can download RIRs generated for our Kaldi Far-field Automatic Speech Recognition Exepriments.

- RIR generated using Geomteric Acoustic Simulator ([**GAS**](https://github.com/RoyJames/pygsound)). -- [**Output**](https://drive.google.com/file/d/175g-lZSJpU1yrjm8LB5c3tOtdJxi-RAc/view?usp=sharing)
- Perform **room equalization** on Synthetic RIRs from **GAS**. -- [**Output**](https://drive.google.com/file/d/1Xo8eX3vlZMvLyKBYAGovO9XgpeFjpsMk/view?usp=sharing)
- First, perform **room equalization**, then **translate the equalized synthetic RIR to a real RIR**. -- [**Output**](https://drive.google.com/file/d/1zKGS2ENYF_YTnhaifXKQyh3Q-g9wFeLK/view?usp=sharing)
- Only **translate synthetic RIR to real RIR**. -- [**Output**](https://drive.google.com/file/d/1ZISePenNQ37_0xazlr1TSMjnVxktkrjD/view?usp=sharing)
- First, **translate a synthetic RIR to a real RIR**, then **perform room equalization to the translated RIR**. -- [**Output**](https://drive.google.com/file/d/1PrLvFOl10qqztNob8whZFdErHg6yYrSt/view?usp=sharing)


### Attribution

If you use this code in your research, please consider citing

```
@article{DBLP:journals/corr/abs-2103-16804,
  author    = {Anton Ratnarajah and
               Zhenyu Tang and
               Dinesh Manocha},
  title     = {{TS-RIR:} Translated synthetic room impulse responses for speech augmentation},
  journal   = {CoRR},
  volume    = {abs/2103.16804},
  year      = {2021}
}
```

```
@inproceedings{donahue2019wavegan,
  title={Adversarial Audio Synthesis},
  author={Donahue, Chris and McAuley, Julian and Puckette, Miller},
  booktitle={ICLR},
  year={2019}
}
```

If you use **Sub-band Room Equalization** please consider citing
```
@inproceedings{9054454,  
  author={Z. {Tang} and H. {Meng} and D. {Manocha}},  
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},  
  title={Low-Frequency Compensated Synthetic Impulse Responses For Improved Far-Field Speech Recognition},   
  year={2020},  
  volume={},  
  number={},  
  pages={6974-6978},
}

```
If you use **Real RIRs** from our dataset folder([**IRs_for_GAN**](https://drive.google.com/file/d/1ivj_UZ5j5inAZwsDTCQ6jEvI5JDtwH_2/view?usp=sharing)), please consider citing

```
@article{DBLP:journals/jstsp/SzokeSMPC19,
  author    = {Igor Sz{\"{o}}ke and
               Miroslav Sk{\'{a}}cel and
               Ladislav Mosner and
               Jakub Paliesek and
               Jan Honza Cernock{\'{y}}},
  title     = {Building and Evaluation of a Real Room Impulse Response Dataset},
  journal   = {{IEEE} J. Sel. Top. Signal Process.},
  volume    = {13},
  number    = {4},
  pages     = {863--876},
  year      = {2019}
}
```
If you use **Synthetic RIRs** from our dataset folder([**IRs_for_GAN**](https://drive.google.com/file/d/1ivj_UZ5j5inAZwsDTCQ6jEvI5JDtwH_2/view?usp=sharing)), please consider citing

```
@inproceedings{9052932,
  author={Z. {Tang} and L. {Chen} and B. {Wu} and D. {Yu} and D. {Manocha}},  
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},  
  title={Improving Reverberant Speech Training Using Diffuse Acoustic Simulation},   
  year={2020},  
  volume={},  
  number={},  
  pages={6969-6973},
}
```


