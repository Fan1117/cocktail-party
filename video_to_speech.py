import argparse
import os
import glob
import random
import shutil
import math
import multiprocessing
from datetime import datetime

from keras.layers import Dense, Convolution3D, MaxPooling3D, ZeroPadding3D, Dropout, Flatten, BatchNormalization, LeakyReLU
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, model_from_json
from keras import optimizers

import numpy as np
import h5py

from mediaio.video_io import VideoFileReader
from mediaio.audio_io import AudioSignal
from dsp.spectrogram import MelConverter
from face_detection import FaceDetector


class VideoToSpeechNet:

	def __init__(self, model):
		self._model = model

	@staticmethod
	def build(video_shape, audio_spectrogram_size):
		model = Sequential()

		model.add(ZeroPadding3D(padding=(1, 2, 2), name='zero1', input_shape=video_shape))
		model.add(Convolution3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv1'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1'))
		model.add(Dropout(0.5))

		model.add(ZeroPadding3D(padding=(1, 2, 2), name='zero2'))
		model.add(Convolution3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv2'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2'))
		model.add(Dropout(0.5))

		model.add(ZeroPadding3D(padding=(1, 1, 1), name='zero3'))
		model.add(Convolution3D(96, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv3'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3'))
		model.add(Dropout(0.5))

		model.add(TimeDistributed(Flatten()))

		model.add(Dense(256, kernel_initializer='he_normal', name='dense0'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(Dropout(0.2))

		model.add(Dense(256, kernel_initializer='he_normal', name='dense1'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(Dropout(0.2))

		model.add(Flatten())

		model.add(Dense(512, kernel_initializer='he_normal', name='dense2'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(Dropout(0.2))

		model.add(Dense(512, kernel_initializer='he_normal', name='dense3'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(Dropout(0.2))

		model.add(Dense(audio_spectrogram_size, name='output'))

		model.summary()

		optimizer = optimizers.adam(lr=0.01, decay=1e-6)
		model.compile(loss="mean_squared_error", optimizer=optimizer)

		return VideoToSpeechNet(model)

	@staticmethod
	def load(model_cache_path, weights_cache_path):
		with open(model_cache_path, "r") as model_fd:
			model = model_from_json(model_fd.read())

		model.load_weights(weights_cache_path)

		return VideoToSpeechNet(model)

	def train(self, x, y):
		self._model.fit(x, y, batch_size=32, validation_split=0.05, epochs=120, verbose=1)

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


def preprocess_video_sample(video_file_path, slice_duration_ms=330, mouth_height=50, mouth_width=100):
	print("preprocessing %s" % video_file_path)

	face_detector = FaceDetector()

	with VideoFileReader(video_file_path) as reader:
		frames = reader.read_all_frames()

		mouth_cropped_frames = np.zeros(shape=(reader.get_frame_count(), mouth_height, mouth_width, 3), dtype=np.float32)
		for i in range(reader.get_frame_count()):
			mouth_cropped_frames[i, :] = face_detector.crop_mouth(frames[i, :], bounding_box_shape=(mouth_width, mouth_height))

		frames_per_slice = (float(slice_duration_ms) / 1000) * reader.get_frame_rate()
		n_slices = int(float(reader.get_frame_count()) / frames_per_slice)

		slices = [
			mouth_cropped_frames[int(i * frames_per_slice) : int(math.ceil((i + 1) * frames_per_slice))]
			for i in range(n_slices)
		]

		return np.stack(slices)


def try_preprocess_video_sample(video_file_path):
	try:
		return preprocess_video_sample(video_file_path)

	except Exception as e:
		print("failed to preprocess %s (%s)" % (video_file_path, e))
		return None


def preprocess_audio_sample(audio_file_path, slice_duration_ms=330):
	print("preprocessing %s" % audio_file_path)

	audio_signal = AudioSignal.from_wav_file(audio_file_path)

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


def reconstruct_audio_signal(y, sample_rate):
	slice_mel_spectrograms = [y[i, :].reshape((MelConverter.N_MEL_FREQS, -1)) for i in range(y.shape[0])]
	full_mel_spectrogram = np.concatenate(slice_mel_spectrograms, axis=1)

	mel_converter = MelConverter(sample_rate)
	return mel_converter.reconstruct_signal_from_mel_spectrogram(full_mel_spectrogram)


def video_to_audio_path(video_file_path):
	return video_file_path.replace("video", "audio").replace(".mpg", ".wav")


def preprocess_data(video_file_paths):
	print("reading dataset...")

	audio_file_paths = [video_to_audio_path(f) for f in video_file_paths]

	thread_pool = multiprocessing.Pool(8)
	x = thread_pool.map(try_preprocess_video_sample, video_file_paths)
	y = thread_pool.map(preprocess_audio_sample, audio_file_paths)

	invalid_sample_ids = [i for i, sample in enumerate(x) if sample is None]
	x = [sample for i, sample in enumerate(x) if i not in invalid_sample_ids]
	y = [sample for i, sample in enumerate(y) if i not in invalid_sample_ids]

	return np.concatenate(x), np.concatenate(y)


def list_video_files(dataset_dir, speaker_ids, max_files=None):
	video_file_paths = []

	for speaker_id in speaker_ids:
		video_file_paths.extend(glob.glob(os.path.join(dataset_dir, speaker_id, "video", "*.mpg")))

	random.shuffle(video_file_paths)
	return video_file_paths[:max_files]


def list_speakers(args):
	if args.speakers is None:
		speaker_ids = os.listdir(args.dataset_dir)
	else:
		speaker_ids = args.speakers

	if args.ignored_speakers is not None:
		for speaker_id in args.ignored_speakers:
			speaker_ids.remove(speaker_id)

	return speaker_ids


def train(args):
	speaker_ids = list_speakers(args)
	video_file_paths = list_video_files(args.dataset_dir, speaker_ids, max_files=15000)

	x, y = preprocess_data(video_file_paths)

	x /= 255
	x -= np.mean(x)

	network = VideoToSpeechNet.build(video_shape=x.shape[1:], audio_spectrogram_size=y.shape[1])
	network.train(x, y)
	network.dump(args.model_cache, args.weights_cache)


def predict(args):
	speaker_ids = list_speakers(args)

	prediction_output_dir = os.path.join(args.prediction_output_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
	os.mkdir(prediction_output_dir)

	network = VideoToSpeechNet.load(args.model_cache, args.weights_cache)

	for speaker_id in speaker_ids:
		speaker_prediction_dir = os.path.join(prediction_output_dir, speaker_id)
		os.mkdir(speaker_prediction_dir)

		test_video_file_paths = list_video_files(args.dataset_dir, [speaker_id], max_files=10)
		for video_file_path in test_video_file_paths:
			try:
				x = preprocess_video_sample(video_file_path)

				x /= 255
				x -= np.mean(x)

				y_predicted = network.predict(x)

				sample_name = os.path.splitext(os.path.basename(video_file_path))[0]

				reconstructed_signal = reconstruct_audio_signal(y_predicted, sample_rate=44100)
				reconstructed_signal.save_to_wav_file(os.path.join(speaker_prediction_dir, "%s.wav" % sample_name))

				shutil.copy(video_file_path, speaker_prediction_dir)

			except Exception as e:
				print("failed to preprocess %s (%s). skipping" % (video_file_path, e))


def main():
	parser = argparse.ArgumentParser(add_help=False)
	action_parsers = parser.add_subparsers()

	train_parser = action_parsers.add_parser("train")
	train_parser.add_argument("dataset_dir", type=str)
	train_parser.add_argument("model_cache", type=str)
	train_parser.add_argument("weights_cache", type=str)
	train_parser.add_argument("--speakers", nargs="+", type=str)
	train_parser.add_argument("--ignored_speakers", nargs="+", type=str)
	train_parser.set_defaults(func=train)

	predict_parser = action_parsers.add_parser("predict")
	predict_parser.add_argument("dataset_dir", type=str)
	predict_parser.add_argument("model_cache", type=str)
	predict_parser.add_argument("weights_cache", type=str)
	predict_parser.add_argument("prediction_output_dir", type=str)
	predict_parser.add_argument("--speakers", nargs="+", type=str)
	predict_parser.add_argument("--ignored_speakers", nargs="+", type=str)
	predict_parser.set_defaults(func=predict)

	args = parser.parse_args()
	args.func(args)

if __name__ == "__main__":
	main()
