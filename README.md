# TS-RIR

This is the official implementation of **TS-RIRGAN**. We started our implementation from [**WaveGAN**](https://github.com/chrisdonahue/wavegan). TS-RIRGAN is one dimensional CycleGAN that takes synthetic RIRs as raw waveform audio and translate it into real RIRs. Our network architecture is shown below.



![Architecture-1.png](https://github.com/anton-jeran/TS-RIR/blob/main/images/Architecture-1.png)


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

