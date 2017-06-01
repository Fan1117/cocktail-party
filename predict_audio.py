import argparse
import os
import shutil

import numpy as np

from mediaio.audio_io import AudioSignal, AudioMixer
from source_separator.audio import SpeechSeparator
from source_separator.speech import Speech


def read_audio_data(source_dir_path1, source_dir_path2):
	print("reading dataset...")

	source_file_paths1 = [os.path.join(source_dir_path1, f) for f in os.listdir(source_dir_path1)]
	source_file_paths2 = [os.path.join(source_dir_path2, f) for f in os.listdir(source_dir_path2)]

	n_samples = min(len(source_file_paths1), len(source_file_paths2))

	x = np.ndarray(shape=(n_samples, 64 * 100))
	mixed_signals = []

	for i in range(n_samples):
		signal1 = AudioSignal.from_wav_file(source_file_paths1[i])
		signal2 = AudioSignal.from_wav_file(source_file_paths2[i])

		mixed_signal = AudioMixer.mix([signal1, signal2], mixing_weights=[1, 1])

		x[i, :] = Speech.signal_to_mel_spectogram(mixed_signal).flatten()
		mixed_signals.append(mixed_signal)

	return x, mixed_signals, source_file_paths1[:n_samples], source_file_paths2[:n_samples]


def reconstruct_signals(y, sample_rate):
	signals = []

	for i in range(y.shape[0]):
		mel_spectogram = y[i, :].reshape((64, 100))
		signal = Speech.reconstruct_signal_from_mel_spectogram(mel_spectogram, sample_rate)

		signals.append(signal)

	return signals


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("source_dir1", type=str)
	parser.add_argument("source_dir2", type=str)
	parser.add_argument("prediction_output_dir", type=str)
	parser.add_argument("train_cache", type=str)
	args = parser.parse_args()

	x, mixed_signals, source_file_paths1, source_file_paths2 = read_audio_data(args.source_dir1, args.source_dir2)

	separator = SpeechSeparator(train_cache_path=args.train_cache)
	y_predicted = separator.predict(x)

	source_signals = reconstruct_signals(y_predicted, sample_rate=44100)

	if os.path.exists(args.prediction_output_dir):
		shutil.rmtree(args.prediction_output_dir)

	os.mkdir(args.prediction_output_dir)

	for i in range(x.shape[0]):
		source_name1 = os.path.splitext(os.path.basename(source_file_paths1[i]))[0]
		source_name2 = os.path.splitext(os.path.basename(source_file_paths2[i]))[0]

		source_prediction_dir_path = os.path.join(args.prediction_output_dir, source_name1 + "_" + source_name2)
		if not os.path.exists(source_prediction_dir_path):
			os.mkdir(source_prediction_dir_path)

		mixed_signals[i].save_to_wav_file(os.path.join(source_prediction_dir_path, "mix.wav"))
		source_signals[i].save_to_wav_file(os.path.join(source_prediction_dir_path, "predicted.wav"))

		shutil.copy(source_file_paths1[i], source_prediction_dir_path)
		shutil.copy(source_file_paths2[i], source_prediction_dir_path)


if __name__ == "__main__":
	main()
