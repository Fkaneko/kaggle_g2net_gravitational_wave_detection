# 15th place solution for g2net_gravitational_wave_detection at kaggle.
* For the details of the competition, please check this page -> [G2Net Gravitational Wave Detection](https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion).

# Overview
![model](https://user-images.githubusercontent.com/61892693/135449819-bcc73b16-eb7c-41e1-ba1a-6b870a535dfa.png)

* Whitening: Using average PSD. Averaging over all noise samples for each site.
* CQT Scaling with `filter_scale = 8/bins_per_octave` and (fmin, fmax)=(20, 1024).  Both abs and angle part were used.
* Augmentation
    * Horizontal/time shift
        * Pad both side and then horizontal random crop to get time shift image. -> ROC +0.002.
    * Mixup, prevent from overfitting
* GeM Fixed power 3 was better than the trainable case. -> ROC +0.001

* Scores

| net       | spec     | height | width | PB score |
| ---       | ---      | ---    | ---   | ---      |
| effnet b0 | Log STFT | 256    | 513   | 0.8760   |
| effnet b0 | CQT      | 181    | 513   | 0.8768   |
| effnet b3 | CQT      | 181    | 1024  | 0.8797   |
| effnet b3 | CQT      | 273    | 1024  | 0.8802  |

My final score is ensemble of Log STFT/CQT models.

# How to run
## environment
* Ubuntu 18.04
* Python with Anaconda/Mamba
* NVIDIA GPUx1

## Data Preparation
First, download the data, [here](https://www.kaggle.com/c/g2net-gravitational-wave-detection/data),
and then place it like below.
```bash
../input/
    └ g2net-gravitational-wave-detection/
```
Outputs will be stored under `../working/` through [hydra](https://hydra.cc/).

## Code&Package Installation
```bash
# clone project
$PROJECT=kaggle_g2net_gravitational_wave_detection
git clone https://github.com/Fkaneko/$PROJECT

# install project
cd $PROJECT
conda create -n g2_net python==3.8.10
bash install.sh
 ```
* This code was for the competition, so some parts of code are not so clean or clear. Please be careful.

## Training & Testing
Simply run followings
```python
python train.py
```
Please check the `src/config/config.yaml` for the default training configuration.
After training, testing will be automatically started with the best validation score checkpoint.

# License
### Code
Apache 2.0

### Dataset
Please check the kaggle page -> https://www.kaggle.com/c/g2net-gravitational-wave-detection/rules
