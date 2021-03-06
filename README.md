# Arabic Text-to-Speech using transfer learning from Tacotron 2 

PyTorch implementation of [Natural TTS Synthesis By Conditioning
Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf). 

This work is based on the work described [A Transfer Learning End-to-End Arabic Text-To-Speech (TTS) Deep Architecture](https://link.springer.com/chapter/10.1007/978-3-030-58309-5_22)

This word uses almost the same code as [Nivedia Tacotron 2](https://github.com/NVIDIA/tacotron2)[Keith
Ito](https://github.com/keithito/tacotron/)
This implementation includes **distributed** and **automatic mixed precision** support
and uses [Nawar Halabi's dataset](http://en.arabicspeechcorpus.com/).

Distributed and Automatic Mixed Precision support relies on NVIDIA's [Apex] and [AMP].

![Alignment, Predicted Mel Spectrogram, Target Mel Spectrogram](tensorboard.png)

## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup
1. Clone this repo: `https://github.com/FadyKhalaf/tacotron2.git`
2. CD into this repo: `cd tacotron2`
3. Download and extract [Nawar Halabi's dataset](http://en.arabicspeechcorpus.com/)
4. place the data set in a folder called `arabic_dataset_folder`
5. Run the preprocessing script `python preprocess_data.py`. 
6. Install [PyTorch]
7. Install [Apex]
8. Install python requirements or build docker image 
    - Install python requirements: `pip install -r requirements.txt`

## Training from the English model [As published by NVIDIA/tacotron2](https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view) 
1. `python train.py --output_directory=outdir --log_directory=logdir -c tacotron2_statedict.pt --warm_start`
2. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  
By default, the dataset dependent text embedding layers are [ignored]

1. Download my pretrained [Tacotron 2] model
2. `python train.py --output_directory=outdir --log_directory=logdir -c checkpoint_55000 --warm_start`

## Multi-GPU (distributed) and Automatic Mixed Precision Training
1. `python -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True`

## Inference demo
1. Download our published [Tacotron 2] model
2. Download our published [WaveGlow] model
3. `jupyter notebook --ip=127.0.0.1 --port=31337`
4. Load inference.ipynb 

N.b.  When performing Mel-Spectrogram to Audio synthesis, make sure Tacotron 2
and the Mel decoder were trained on the same mel-spectrogram representation. 


## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based
Generative Network for Speech Synthesis

[nv-wavenet](https://github.com/NVIDIA/nv-wavenet/) Faster than real time
WaveNet.

## Acknowledgements
This implementation uses code from the following repos: [Nivedia Tacotron 2](https://github.com/NVIDIA/tacotron2)[Keith
Ito](https://github.com/keithito/tacotron/), [Prem
Seetharaman](https://github.com/pseeth/pytorch-stft) as described in my code.

We are inspired by [Ryuchi Yamamoto's](https://github.com/r9y9/tacotron_pytorch)
Tacotron PyTorch implementation.

We are thankful to the Tacotron 2 paper authors, specially Jonathan Shen, Yuxuan
Wang and Zongheng Yang.


[WaveGlow]: https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF
[Tacotron 2]: https://drive.google.com/file/d/1mu7JHihfe98Syww1UsIWF1XOEoZ95IKQ/view?usp=sharing
[pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/WaveGlow
[ignored]: https://github.com/NVIDIA/tacotron2/blob/master/hparams.py#L22
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp
