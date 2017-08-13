import argparse
import os
import shutil
import glob
import math
from datetime import datetime

import numpy as np

from mediaio.audio_io import AudioSignal, AudioMixer
from dsp.spectrogram import MelConverter
from dataset import AudioVisualDataset


def softmax(x):
	x_exp = [math.exp(i) for i in x]
	x_exp_sum = sum(x_exp)

	return [i / x_exp_sum for i in x_exp]


def separate(dataset_dir, speech_prediction_dir, separation_output_dir, speakers, softmax_masking=True):
	dataset = AudioVisualDataset(dataset_dir)

	subsets = [dataset.subset([speaker_id], max_files=20, shuffle=True) for speaker_id in speakers]
	paired_sources = zip(*[subset.audio_paths() for subset in subsets])

	separation_output_dir = os.path.join(separation_output_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
	os.mkdir(separation_output_dir)

	for source_file_paths in paired_sources:
		try:
			print("separating mixture of %s" % str(source_file_paths))

			source_signals = [AudioSignal.from_wav_file(f) for f in source_file_paths]
			mixed_signal = AudioMixer.mix(source_signals)

			mel_converter = MelConverter(mixed_signal.get_sample_rate(), n_mel_freqs=128, freq_min_hz=300, freq_max_hz=3400)
			mixed_spectrogram = mel_converter.signal_to_mel_spectrogram(mixed_signal)

			source_names = [os.path.splitext(os.path.basename(f))[0] for f in source_file_paths]
			speech_prediction_paths = [
				glob.glob(os.path.join(speech_prediction_dir, speakers[i], source_name + ".wav"))[0]
				for i, source_name in enumerate(source_names)
			]

			speech_signals = [AudioSignal.from_wav_file(f) for f in speech_prediction_paths]
			speech_spectrograms = [mel_converter.signal_to_mel_spectrogram(signal) for signal in speech_signals]

			if mixed_spectrogram.shape[1] > speech_spectrograms[0].shape[1]:
				mixed_spectrogram = mixed_spectrogram[:, :speech_spectrograms[0].shape[1]]
			else:
				speech_spectrograms = [spectrogram[:, :mixed_spectrogram.shape[1]] for spectrogram in speech_spectrograms]

			masks = [np.zeros(shape=mixed_spectrogram.shape) for _ in range(len(speakers))]

			for i in range(mixed_spectrogram.shape[0]):
				for j in range(mixed_spectrogram.shape[1]):
					magnitudes = [spectrogram[i, j] for spectrogram in speech_spectrograms]

					if softmax_masking:
						weights = softmax(magnitudes)

						for s in range(len(speakers)):
							masks[s][i, j] = weights[s]

					else:
						sort_indices = np.argsort(magnitudes)
						dominant_speaker = sort_indices[-1]
						sub_dominant_speaker = sort_indices[-2]

						if magnitudes[dominant_speaker] - magnitudes[sub_dominant_speaker] > 1:
							masks[dominant_speaker][i, j] = 1

			separated_spectrograms = [mixed_spectrogram * mask for mask in masks]
			reconstructed_signals = [mel_converter.reconstruct_signal_from_mel_spectrogram(s) for s in separated_spectrograms]

			source_separation_dir_path = os.path.join(separation_output_dir, '_'.join(source_names))
			os.mkdir(source_separation_dir_path)

			for i in range(len(source_file_paths)):
				shutil.copy(source_file_paths[i], os.path.join(source_separation_dir_path, "source-%d.wav" % i))
				reconstructed_signals[i].save_to_wav_file(os.path.join(source_separation_dir_path, "estimated-%d.wav" % i))

			mixed_signal.save_to_wav_file(os.path.join(source_separation_dir_path, "mixture.wav"))

		except Exception as e:
			print("failed to separate mixture (%s). skipping" % e)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_dir", type=str)
	parser.add_argument("speech_prediction_dir", type=str)
	parser.add_argument("separation_output_dir", type=str)
	parser.add_argument("speakers", nargs='+', type=str)
	args = parser.parse_args()

	separate(args.dataset_dir, args.speech_prediction_dir, args.separation_output_dir, args.speakers)


if __name__ == "__main__":
	main()
