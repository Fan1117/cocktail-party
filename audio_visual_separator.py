import argparse
import os
import shutil
import math
from datetime import datetime

from keras.layers import Dense, Convolution2D, MaxPooling2D, BatchNormalization, Activation, Dropout, LeakyReLU, Flatten, Merge
from keras.models import Sequential, model_from_json
from keras import optimizers

import numpy as np
import h5py
import cv2

from mediaio.video_io import VideoFileReader
from utils import fs

from audio_separator import preprocess_audio_data, preprocess_audio_signal_pair, reconstruct_audio_signal


class AudioVisualSourceSeparator:

	def __init__(self):
		pass

	@staticmethod
	def load(model_cache_path, weights_cache_path):
		separator = AudioVisualSourceSeparator()

		with open(model_cache_path, "r") as model_fd:
			separator._model = model_from_json(model_fd.read())

		separator._model.load_weights(weights_cache_path)

		return separator

	def init_model(self, audio_spectogram_size, video_shape):
		audio_model = self._init_audio_model(audio_spectogram_size)
		video_model = self._init_video_model(video_shape)

		self._model = Sequential()
		self._model.add(Merge([audio_model, video_model], mode='concat'))

		self._model.add(Dense(units=256))
		self._model.add(BatchNormalization())
		self._model.add(Activation("relu"))
		self._model.add(Dropout(0.25))

		self._model.add(Dense(units=256))
		self._model.add(BatchNormalization())
		self._model.add(Activation("relu"))
		self._model.add(Dropout(0.25))

		self._model.add(Dense(units=audio_spectogram_size))

		optimizer = optimizers.adam(lr=0.01, decay=1e-6)
		self._model.compile(loss='mean_squared_error', optimizer=optimizer)

	def _init_audio_model(self, audio_spectogram_size):
		model = Sequential()

		model.add(Dense(units=512, input_dim=audio_spectogram_size))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(Dropout(0.25))

		model.add(Dense(units=512))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(Dropout(0.25))

		return model

	def _init_video_model(self, video_shape):
		model = Sequential()
		model.add(Convolution2D(32, (3, 3), padding="same", kernel_initializer="he_normal", input_shape=video_shape))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Convolution2D(32, (3, 3), padding="same", kernel_initializer="he_normal"))
		model.add(BatchNormalization())
		model.add(LeakyReLU())

		model.add(Convolution2D(32, (3, 3), padding="same", kernel_initializer="he_normal"))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))

		model.add(Convolution2D(64, (3, 3), padding="same", kernel_initializer="he_normal"))
		model.add(BatchNormalization())
		model.add(LeakyReLU())

		model.add(Convolution2D(64, (3, 3), padding="same", kernel_initializer="he_normal"))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))

		model.add(Convolution2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
		model.add(BatchNormalization())
		model.add(LeakyReLU())

		model.add(Convolution2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))

		# model.add(Convolution2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
		# model.add(BatchNormalization())
		# model.add(LeakyReLU())

		# model.add(Convolution2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
		# model.add(BatchNormalization())
		# model.add(Activation("tanh"))
		# model.add(MaxPooling2D(pool_size=(2, 2)))
		# model.add(Dropout(0.2))
		model.add(Flatten())

		return model

	def train(self, x_audio, x_video, y):
		self._model.fit([x_audio, x_video], y, batch_size=16, epochs=50, verbose=1)

	def evaluate(self, x_audio, x_video, y):
		score = self._model.evaluate([x_audio, x_video], y, verbose=1)
		return score

	def predict(self, x_audio, x_video):
		y = self._model.predict([x_audio, x_video])
		return y

	def dump(self, model_cache_path, weights_cache_path):
		with open(model_cache_path, "w") as model_fd:
			model_fd.write(self._model.to_json())

		self._model.save_weights(weights_cache_path)


def preprocess_video_sample(video_source_file_path1, video_source_file_path2, slice_duration_ms=100):
	face_detector = cv2.CascadeClassifier(
		os.path.join(os.path.dirname(__file__), "res", "haarcascade_frontalface_alt.xml")
	)

	with VideoFileReader(video_source_file_path1) as reader:
		frames = reader.read_all_frames(convert_to_gray_scale=True)
		frames_per_slice = (float(slice_duration_ms) / 1000) * reader.get_frame_rate()
		n_slices = int(float(reader.get_frame_count()) / frames_per_slice)

	face_cropped_frames = np.zeros(shape=(75, 128, 128))
	for j in range(frames.shape[0]):
		faces = face_detector.detectMultiScale(frames[j, :])
		(face_x, face_y, face_width, face_height) = faces[0]
		face = frames[j, face_y: (face_y + face_height), face_x: (face_x + face_width)]

		face_cropped_frames[j, :] = cv2.resize(face, (128, 128))

	# fit to tensorflow channel_last data format
	face_cropped_frames = face_cropped_frames.transpose((1, 2, 0))

	slices = [
		face_cropped_frames[:, :, int(i * frames_per_slice) : int(math.ceil((i + 1) * frames_per_slice))]
		for i in range(n_slices)
	]

	return np.stack(slices)


def preprocess_video_data(source_dir_path1, source_dir_path2, max_pairs):
	print("reading video dataset...")

	source_file_paths1 = fs.list_dir_by_name(source_dir_path1)
	source_file_paths2 = fs.list_dir_by_name(source_dir_path2)

	x = []

	n_pairs = min(len(source_file_paths1), len(source_file_paths2), max_pairs)
	for i in range(n_pairs):
		x_i = preprocess_video_sample(source_file_paths1[i], source_file_paths2[i])

		x.append(x_i)

	return np.concatenate(x)


def read_data(source_dir_path1, source_dir_path2, max_pairs):
	audio_dir_path1 = os.path.join(source_dir_path1, "audio")
	audio_dir_path2 = os.path.join(source_dir_path2, "audio")

	x_audio, y = preprocess_audio_data(audio_dir_path1, audio_dir_path2, max_pairs)

	video_dir_path1 = os.path.join(source_dir_path1, "video")
	video_dir_path2 = os.path.join(source_dir_path2, "video")

	x_video = preprocess_video_data(video_dir_path1, video_dir_path2, max_pairs)

	return x_audio, x_video, y


def train(args):
	if os.path.exists("xy.h5"):
		with h5py.File("xy.h5", 'r') as cached_dataset_fd:
			x_audio_train = cached_dataset_fd["x_audio"][:]
			x_video_train = cached_dataset_fd["x_video"][:]
			y_train = cached_dataset_fd["y"][:]

	else:
		x_audio_train, x_video_train, y_train = read_data(args.train_source_dir1, args.train_source_dir2, max_pairs=500)

		with h5py.File("xy.h5", 'w') as cached_dataset_fd:
			cached_dataset_fd.create_dataset('x_audio', data=x_audio_train)
			cached_dataset_fd.create_dataset('x_video', data=x_video_train)
			cached_dataset_fd.create_dataset('y', data=y_train)

	separator = AudioVisualSourceSeparator()
	separator.init_model(audio_spectogram_size=x_audio_train.shape[1], video_shape=x_video_train.shape[1:])
	separator.train(x_audio_train, x_video_train, y_train)
	separator.dump(args.model_cache, args.weights_cache)


def predict(args):
	separator = AudioVisualSourceSeparator.load(args.model_cache, args.weights_cache)

	prediction_output_dir = os.path.join(args.prediction_output_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
	os.mkdir(prediction_output_dir)

	audio_source_file_paths1 = fs.list_dir_by_name(os.path.join(args.test_source_dir1, "audio"))
	audio_source_file_paths2 = fs.list_dir_by_name(os.path.join(args.test_source_dir2, "audio"))

	video_source_file_paths1 = fs.list_dir_by_name(os.path.join(args.test_source_dir1, "video"))
	video_source_file_paths2 = fs.list_dir_by_name(os.path.join(args.test_source_dir2, "video"))

	n_pairs = min(len(audio_source_file_paths1), len(audio_source_file_paths2))

	for i in range(n_pairs):
		x_audio, _, mixed_signal = preprocess_audio_signal_pair(audio_source_file_paths1[i], audio_source_file_paths2[i])
		x_video = preprocess_video_sample(video_source_file_paths1[i], video_source_file_paths2[i])

		y_predicted = separator.predict(x_audio, x_video)

		reconstructed_signal = reconstruct_audio_signal(y_predicted, mixed_signal.get_sample_rate())

		source_name1 = os.path.splitext(os.path.basename(audio_source_file_paths1[i]))[0]
		source_name2 = os.path.splitext(os.path.basename(audio_source_file_paths2[i]))[0]

		source_prediction_dir_path = os.path.join(prediction_output_dir, source_name1 + "_" + source_name2)
		os.mkdir(source_prediction_dir_path)

		reconstructed_signal.save_to_wav_file(os.path.join(source_prediction_dir_path, "predicted.wav"))
		mixed_signal.save_to_wav_file(os.path.join(source_prediction_dir_path, "mix.wav"))

		shutil.copy(audio_source_file_paths1[i], source_prediction_dir_path)
		shutil.copy(audio_source_file_paths2[i], source_prediction_dir_path)

		shutil.copy(video_source_file_paths1[i], source_prediction_dir_path)


def main():
	parser = argparse.ArgumentParser(add_help=False)
	action_parsers = parser.add_subparsers()

	train_parser = action_parsers.add_parser("train")
	train_parser.add_argument("train_source_dir1", type=str)
	train_parser.add_argument("train_source_dir2", type=str)
	train_parser.add_argument("model_cache", type=str)
	train_parser.add_argument("weights_cache", type=str)
	train_parser.set_defaults(func=train)

	predict_parser = action_parsers.add_parser("predict")
	predict_parser.add_argument("test_source_dir1", type=str)
	predict_parser.add_argument("test_source_dir2", type=str)
	predict_parser.add_argument("model_cache", type=str)
	predict_parser.add_argument("weights_cache", type=str)
	predict_parser.add_argument("prediction_output_dir", type=str)
	predict_parser.set_defaults(func=predict)

	args = parser.parse_args()
	args.func(args)

if __name__ == "__main__":
	main()
