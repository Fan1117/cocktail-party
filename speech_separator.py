import argparse
import os
import shutil
import glob
import math
from datetime import datetime

import numpy as np

from mediaio.audio_io import AudioSignal, AudioMixer
from dsp.spectrogram import MelConverter
from audio_separator import list_audio_source_file_pairs


def softmax(x):
	x_exp = [math.exp(i) for i in x]
	x_exp_sum = sum(x_exp)

	return [i / x_exp_sum for i in x_exp]


def separate(dataset_dir, speech_prediction_dir, separation_output_dir, speakers):
	source_file_pairs = list_audio_source_file_pairs(dataset_dir, speakers, max_pairs=10)

	separation_output_dir = os.path.join(separation_output_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
	os.mkdir(separation_output_dir)

	for source_file_path1, source_file_path2 in source_file_pairs:
		print("predicting mix of %s, %s" % (source_file_path1, source_file_path2))
		source_signal1 = AudioSignal.from_wav_file(source_file_path1)
		source_signal2 = AudioSignal.from_wav_file(source_file_path2)
		mixed_signal = AudioMixer.mix([source_signal1, source_signal2])

		mel_converter = MelConverter(mixed_signal.get_sample_rate())
		mixed_spectrogram = mel_converter.signal_to_mel_spectrogram(mixed_signal)

		source_name1 = os.path.splitext(os.path.basename(source_file_path1))[0]
		source_name2 = os.path.splitext(os.path.basename(source_file_path2))[0]

		speech_prediction_path1 = glob.glob(os.path.join(speech_prediction_dir, speakers[0], source_name1 + ".wav"))
		speech_prediction_path2 = glob.glob(os.path.join(speech_prediction_dir, speakers[1], source_name2 + ".wav"))

		if len(speech_prediction_path1) != 1 or len(speech_prediction_path2) != 1:
			continue

		speech_signal1 = AudioSignal.from_wav_file(speech_prediction_path1[0])
		speech_signal2 = AudioSignal.from_wav_file(speech_prediction_path2[0])

		spectrogram1 = mel_converter.signal_to_mel_spectrogram(speech_signal1)
		spectrogram2 = mel_converter.signal_to_mel_spectrogram(speech_signal2)

		if mixed_spectrogram.shape[1] > spectrogram1.shape[1]:
			mixed_spectrogram = mixed_spectrogram[:, :spectrogram1.shape[1]]
		else:
			spectrogram1 = spectrogram1[:, :mixed_spectrogram.shape[1]]
			spectrogram2 = spectrogram2[:, :mixed_spectrogram.shape[1]]

		mask1 = np.zeros(shape=mixed_spectrogram.shape)
		mask2 = np.zeros(shape=mixed_spectrogram.shape)

		for i in range(mixed_spectrogram.shape[0]):
			for j in range(mixed_spectrogram.shape[1]):
				magnitudes = [spectrogram1[i, j], spectrogram2[i, j]]
				weights = softmax(magnitudes)

				mask1[i, j] = weights[0]
				mask2[i, j] = weights[1]

		separated_spectrogram1 = mixed_spectrogram * mask1
		separated_spectrogram2 = mixed_spectrogram * mask2

		reconstructed_signal1 = mel_converter.reconstruct_signal_from_mel_spectrogram(separated_spectrogram1)
		reconstructed_signal2 = mel_converter.reconstruct_signal_from_mel_spectrogram(separated_spectrogram2)

		source_separation_dir_path = os.path.join(separation_output_dir, source_name1 + "_" + source_name2)
		os.mkdir(source_separation_dir_path)

		reconstructed_signal1.save_to_wav_file(os.path.join(source_separation_dir_path, "predicted1.wav"))
		reconstructed_signal2.save_to_wav_file(os.path.join(source_separation_dir_path, "predicted2.wav"))

		mixed_signal.save_to_wav_file(os.path.join(source_separation_dir_path, "mix.wav"))

		shutil.copy(source_file_path1, source_separation_dir_path)
		shutil.copy(source_file_path2, source_separation_dir_path)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_dir", type=str)
	parser.add_argument("speech_prediction_dir", type=str)
	parser.add_argument("separation_output_dir", type=str)
	parser.add_argument("speakers", nargs=2, type=str)
	args = parser.parse_args()

	separate(args.dataset_dir, args.speech_prediction_dir, args.separation_output_dir, args.speakers)


if __name__ == "__main__":
	main()
