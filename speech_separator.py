import argparse
import os
import shutil
from datetime import datetime

import numpy as np

from mediaio.audio_io import AudioSignal, AudioMixer
from spectrogram import MelConverter
from dataset import AudioVisualDataset


def generate_separation_masks(mixed_spectrogram, prediction_spectrograms, separation_function):
	masks = [np.zeros(shape=mixed_spectrogram.shape) for _ in range(len(prediction_spectrograms))]

	for f in range(mixed_spectrogram.shape[0]):
		for t in range(mixed_spectrogram.shape[1]):
			magnitudes = [spectrogram[f, t] for spectrogram in prediction_spectrograms]
			weights = separation_function(magnitudes)

			for s in range(len(weights)):
				masks[s][f, t] = weights[s]

	return masks


def separate_sources(source_file_paths, prediction_file_paths, separation_function):
	print("separating mixture of %s" % str(source_file_paths))

	source_signals = [AudioSignal.from_wav_file(f) for f in source_file_paths]
	prediction_signals = [AudioSignal.from_wav_file(f) for f in prediction_file_paths]

	signals = source_signals + prediction_signals
	max_length = max([signal.get_number_of_samples() for signal in signals])
	for signal in signals:
		signal.pad_with_zeros(max_length)

	mixed_signal = AudioMixer.mix(source_signals)

	mel_converter = MelConverter(mixed_signal.get_sample_rate(), n_mel_freqs=128, freq_min_hz=0, freq_max_hz=None)
	mixed_spectrogram, original_phase = mel_converter.signal_to_mel_spectrogram(mixed_signal, log=False, get_phase=True)
	prediction_spectrograms = [mel_converter.signal_to_mel_spectrogram(signal, log=False) for signal in prediction_signals]

	masks = generate_separation_masks(mixed_spectrogram, prediction_spectrograms, separation_function)
	separated_spectrograms = [mixed_spectrogram * mask for mask in masks]
	separated_signals = [
		mel_converter.reconstruct_signal_from_mel_spectrogram(s, log=False, phase=original_phase) for s in separated_spectrograms
	]

	return mixed_signal, separated_signals


def apply_source_separation(dataset_dir, prediction_input_dir, separation_output_dir, speakers, separation_function):
	separation_output_dir = os.path.join(separation_output_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
	os.mkdir(separation_output_dir)

	for source_file_paths in list_source_pairs(dataset_dir, speakers):
		try:
			source_names = [os.path.splitext(os.path.basename(f))[0] for f in source_file_paths]
			source_separation_dir_path = os.path.join(separation_output_dir, '_'.join(source_names))
			os.mkdir(source_separation_dir_path)

			prediction_file_paths = list_prediction_files(source_file_paths, prediction_input_dir, speakers)
			mixed_signal, separated_signals = separate_sources(source_file_paths, prediction_file_paths, separation_function)

			for i in range(len(source_file_paths)):
				shutil.copy(source_file_paths[i], os.path.join(source_separation_dir_path, "source-%d.wav" % i))
				shutil.copy(prediction_file_paths[i], os.path.join(source_separation_dir_path, "prediction-%d.wav" % i))
				separated_signals[i].save_to_wav_file(os.path.join(source_separation_dir_path, "estimated-%d.wav" % i))

			mixed_signal.save_to_wav_file(os.path.join(source_separation_dir_path, "mixture.wav"))

		except Exception as e:
			print("failed to separate mixture (%s). skipping" % e)


def list_prediction_files(source_file_paths, prediction_dir, speakers):
	source_names = [os.path.basename(f) for f in source_file_paths]
	prediction_paths = [
		os.path.join(prediction_dir, speakers[i], source_name)
		for i, source_name in enumerate(source_names)
	]

	return prediction_paths


def list_source_pairs(dataset_dir, speakers):
	dataset = AudioVisualDataset(dataset_dir)
	subsets = [dataset.subset([speaker_id], max_files=20, shuffle=True) for speaker_id in speakers]

	return zip(*[subset.audio_paths() for subset in subsets])


def ratio_separator(magnitudes):
	if np.sum(magnitudes) == 0:
		return np.zeros_like(magnitudes)

	return [m / np.sum(magnitudes) for m in magnitudes]


def binary_separator(magnitudes):
	weights = [0] * len(magnitudes)

	sort_indices = np.argsort(magnitudes)
	dominant_source = sort_indices[-1]

	weights[dominant_source] = 1

	return weights


def get_separation_function(separation_type):
	if separation_type == 'ratio':
		return ratio_separator

	elif separation_type == 'binary':
		return binary_separator

	else:
		raise Exception("unsupported separation type: %s" % separation_type)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_dir", type=str)
	parser.add_argument("prediction_input_dir", type=str)
	parser.add_argument("separation_output_dir", type=str)
	parser.add_argument("separation_type", type=str, choices=["ratio", "binary"])
	parser.add_argument("speakers", nargs='+', type=str)
	args = parser.parse_args()

	separation_function = get_separation_function(args.separation_type)

	apply_source_separation(
		args.dataset_dir, args.prediction_input_dir, args.separation_output_dir, args.speakers, separation_function
	)


if __name__ == "__main__":
	main()
