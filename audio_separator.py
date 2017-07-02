import argparse
import os
import shutil
import copy
import math
from datetime import datetime
import glob
import random

from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras.models import Sequential, model_from_json
from keras import optimizers

import numpy as np

from mediaio.audio_io import AudioSignal, AudioMixer
from dsp.spectrogram import MelConverter


class AudioSourceSeparator:

	def __init__(self):
		pass

	@staticmethod
	def load(model_cache_path, weights_cache_path):
		separator = AudioSourceSeparator()

		with open(model_cache_path, "r") as model_fd:
			separator._model = model_from_json(model_fd.read())

		separator._model.load_weights(weights_cache_path)

		return separator

	def init_model(self, spectrogram_size):
		self._model = Sequential()

		self._model.add(Dense(units=512, input_dim=spectrogram_size))
		self._model.add(BatchNormalization())
		self._model.add(Activation("relu"))
		self._model.add(Dropout(0.25))

		self._model.add(Dense(units=512))
		self._model.add(BatchNormalization())
		self._model.add(Activation("relu"))
		self._model.add(Dropout(0.25))

		self._model.add(Dense(units=spectrogram_size, activation="sigmoid"))

		optimizer = optimizers.adam(lr=0.01, decay=1e-6)
		self._model.compile(loss='mean_squared_error', optimizer=optimizer)

	def train(self, x, y):
		self._model.fit(x, y, batch_size=32, epochs=100, verbose=1)

	def evaluate(self, x, y):
		score = self._model.evaluate(x, y, verbose=1)
		return score

	def predict(self, x):
		y = self._model.predict(x)
		return y

	def dump(self, model_cache_path, weights_cache_path):
		with open(model_cache_path, "w") as model_fd:
			model_fd.write(self._model.to_json())

		self._model.save_weights(weights_cache_path)


def get_mel_spectrogram_slices(audio_signal, slice_duration_ms=100):
	audio_signal = copy.deepcopy(audio_signal)

	new_signal_length = int(math.ceil(
		float(audio_signal.get_number_of_samples()) / MelConverter.HOP_LENGTH
	)) * MelConverter.HOP_LENGTH

	audio_signal.pad_with_zeros(new_signal_length)

	mel_converter = MelConverter(audio_signal.get_sample_rate())
	mel_spectrogram = mel_converter.signal_to_mel_spectrogram(audio_signal)

	samples_per_slice = int((float(slice_duration_ms) / 1000) * audio_signal.get_sample_rate())
	spectrogram_samples_per_slice = int(samples_per_slice / MelConverter.HOP_LENGTH)

	n_slices = int(mel_spectrogram.shape[1] / spectrogram_samples_per_slice)

	sample = np.ndarray(shape=(n_slices, MelConverter.N_MEL_FREQS * spectrogram_samples_per_slice))

	for i in range(n_slices):
		sample[i, :] = mel_spectrogram[:, (i * spectrogram_samples_per_slice):((i + 1) * spectrogram_samples_per_slice)].flatten()

	return sample


def preprocess_audio_signal_pair(source_file_path1, source_file_path2):
	signal1 = AudioSignal.from_wav_file(source_file_path1)
	signal2 = AudioSignal.from_wav_file(source_file_path2)
	mixed_signal = AudioMixer.mix([signal1, signal2])

	x = get_mel_spectrogram_slices(mixed_signal)

	mel_spectrogram_slices1 = get_mel_spectrogram_slices(signal1)
	mel_spectrogram_slices2 = get_mel_spectrogram_slices(signal2)

	y = np.zeros(shape=x.shape)
	y[mel_spectrogram_slices1 > mel_spectrogram_slices2] = 1

	return x, y, mixed_signal


def reconstruct_audio_signal(y, mixed_signal, separation_mask_threshold=0.5):
	source_separation_mask = np.zeros(shape=y.shape)
	source_separation_mask[y > separation_mask_threshold] = 1

	source_mel_spectrogram_slices = source_separation_mask * get_mel_spectrogram_slices(mixed_signal)
	slice_mel_spectrograms = [
		source_mel_spectrogram_slices[i, :].reshape((MelConverter.N_MEL_FREQS, -1))
		for i in range(source_mel_spectrogram_slices.shape[0])
	]

	source_mel_spectrogram = np.concatenate(slice_mel_spectrograms, axis=1)

	# TODO: maybe use the original phase of each frequency instead of using griffin-lim
	mel_converter = MelConverter(mixed_signal.get_sample_rate())
	return mel_converter.reconstruct_signal_from_mel_spectrogram(source_mel_spectrogram)


def preprocess_audio_data(source_file_pairs):
	print("reading audio dataset...")

	x = []
	y = []

	for source_file_path1, source_file_path2 in source_file_pairs:
		x_i, y_i, _ = preprocess_audio_signal_pair(source_file_path1, source_file_path2)

		x.append(x_i)
		y.append(y_i)

	return np.concatenate(x), np.concatenate(y)


def list_audio_source_file_pairs(dataset_dir, speaker_ids=None, max_pairs=None):
	if speaker_ids is None:
		speaker_ids = os.listdir(dataset_dir)

	speaker_data1 = glob.glob(os.path.join(dataset_dir, speaker_ids[0], "audio", "*.wav"))
	speaker_data2 = glob.glob(os.path.join(dataset_dir, speaker_ids[1], "audio", "*.wav"))

	random.shuffle(speaker_data1)
	random.shuffle(speaker_data2)

	return zip(speaker_data1, speaker_data2)[:max_pairs]


def train(args):
	source_file_pairs = list_audio_source_file_pairs(args.dataset_dir, speaker_ids=args.speakers, max_pairs=500)
	x_train, y_train = preprocess_audio_data(source_file_pairs)

	separator = AudioSourceSeparator()
	separator.init_model(spectrogram_size=x_train.shape[1])
	separator.train(x_train, y_train)
	separator.dump(args.model_cache, args.weights_cache)


def evaluate(args):
	source_file_pairs = list_audio_source_file_pairs(args.dataset_dir)
	x_test, y_test = preprocess_audio_data(source_file_pairs)

	separator = AudioSourceSeparator.load(args.model_cache, args.weights_cache)
	score = separator.evaluate(x_test, y_test)
	print("score: %.2f" % score)


def predict(args):
	separator = AudioSourceSeparator.load(args.model_cache, args.weights_cache)

	prediction_output_dir = os.path.join(args.prediction_output_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
	os.mkdir(prediction_output_dir)

	source_file_pairs = list_audio_source_file_pairs(args.dataset_dir, speaker_ids=args.speakers)

	for source_file_path1, source_file_path2 in source_file_pairs:
		x, _, mixed_signal = preprocess_audio_signal_pair(source_file_path1, source_file_path2)
		y_predicted = separator.predict(x)

		reconstructed_signal = reconstruct_audio_signal(y_predicted, mixed_signal)

		source_name1 = os.path.splitext(os.path.basename(source_file_path1))[0]
		source_name2 = os.path.splitext(os.path.basename(source_file_path2))[0]

		source_prediction_dir_path = os.path.join(prediction_output_dir, source_name1 + "_" + source_name2)
		os.mkdir(source_prediction_dir_path)

		reconstructed_signal.save_to_wav_file(os.path.join(source_prediction_dir_path, "predicted.wav"))
		mixed_signal.save_to_wav_file(os.path.join(source_prediction_dir_path, "mix.wav"))

		shutil.copy(source_file_path1, source_prediction_dir_path)
		shutil.copy(source_file_path2, source_prediction_dir_path)


def main():
	parser = argparse.ArgumentParser(add_help=False)
	action_parsers = parser.add_subparsers()

	train_parser = action_parsers.add_parser("train")
	train_parser.add_argument("dataset_dir", type=str)
	train_parser.add_argument("model_cache", type=str)
	train_parser.add_argument("weights_cache", type=str)
	train_parser.add_argument("--speakers", nargs="+", type=str)
	train_parser.set_defaults(func=train)

	predict_parser = action_parsers.add_parser("predict")
	predict_parser.add_argument("dataset_dir", type=str)
	predict_parser.add_argument("model_cache", type=str)
	predict_parser.add_argument("weights_cache", type=str)
	predict_parser.add_argument("prediction_output_dir", type=str)
	predict_parser.add_argument("--speakers", nargs="+", type=str)
	predict_parser.set_defaults(func=predict)

	args = parser.parse_args()
	args.func(args)

if __name__ == "__main__":
	main()
