import argparse
import os

import numpy as np

from mediaio.audio_io import AudioSignal, AudioMixer
from source_separator.audio import SpeechSeparator
from source_separator.speech import Speech


def read_audio_data(source_dir_path1, source_dir_path2, max_samples):
	print("reading dataset...")

	source_file_paths1 = [os.path.join(source_dir_path1, f) for f in os.listdir(source_dir_path1)]
	source_file_paths2 = [os.path.join(source_dir_path2, f) for f in os.listdir(source_dir_path2)]

	n_samples = min(len(source_file_paths1), len(source_file_paths2), max_samples)

	x = np.ndarray(shape=(n_samples, 64 * 100))
	y = np.ndarray(shape=(n_samples, 64 * 100))

	for i in range(n_samples):
		signal1 = AudioSignal.from_wav_file(source_file_paths1[i])
		signal2 = AudioSignal.from_wav_file(source_file_paths2[i])

		mixed_signal = AudioMixer.mix([signal1, signal2], mixing_weights=[1, 1])

		x[i, :] = Speech.signal_to_mel_spectogram(mixed_signal).flatten()
		y[i, :] = Speech.signal_to_mel_spectogram(signal1).flatten()

	return x, y


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("train_source_dir1", type=str)
	parser.add_argument("train_source_dir2", type=str)
	parser.add_argument("test_source_dir1", type=str)
	parser.add_argument("test_source_dir2", type=str)
	parser.add_argument("train_cache", type=str)
	args = parser.parse_args()

	x_train, y_train = read_audio_data(args.train_source_dir1, args.train_source_dir2, max_samples=500)
	x_test, y_test = read_audio_data(args.test_source_dir1, args.test_source_dir2, max_samples=10)

	separator = SpeechSeparator(train_cache_path=args.train_cache)
	separator.train(x_train, y_train)
	score = separator.evaluate(x_test, y_test)
	print("score: %.2f" % score)


if __name__ == "__main__":
	main()
