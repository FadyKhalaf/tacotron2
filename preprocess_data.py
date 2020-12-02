import librosa
from os import listdir
from scipy.io.wavfile import read
import layers
import torch
import numpy as np
from hparams import create_hparams

if __name__ == '__main__':
	hparams = create_hparams()
	stft = layers.TacotronSTFT(
	hparams.filter_length, hparams.hop_length, hparams.win_length,
	hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
	hparams.mel_fmax)

	files = ['filelists/arabic_audio_text_train_filelist.txt' , 'filelists/arabic_audio_text_val_filelist.txt']
	for file in files:
		with open(file) as f:
			for line in f:
				fields = line.split('|')
				filename = fields[0].split('/')[-1]
				melname = filename.replace('.wav', '.mel')
				_data, sampling_rate = librosa.core.load(fields[0])
				data, _ = librosa.effects.trim(_data, frame_length=1024,hop_length=256)
				audio = torch.FloatTensor(data.astype(np.float32))
				audio_norm = audio.unsqueeze(0)
				audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
				melspec = stft.mel_spectrogram(audio_norm)
				melspec = torch.squeeze(melspec, 0)
				np.save(f'arabic_dataset_folder/mels/{melname}', melspec.numpy())
				print(melname)
