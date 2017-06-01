import argparse
import os
import shutil

import cv2
import numpy as np

from mediaio.audio_io import AudioSignal, AudioMixer
from mediaio.video_io import VideoFileReader
from source_separator.audio_visual import SpeechSeparator
from source_separator.speech import Speech


def read_data(source_dir_path1, source_dir_path2):
	print("reading dataset...")

	audio_dir_path1 = os.path.join(source_dir_path1, "audio")
	audio_dir_path2 = os.path.join(source_dir_path2, "audio")

	file_names1 = [os.path.splitext(f)[0] for f in os.listdir(audio_dir_path1)]
	file_names2 = [os.path.splitext(f)[0] for f in os.listdir(audio_dir_path2)]

	n_samples = min(len(file_names1), len(file_names2))

	x_audio, mixed_signals, source_file_paths1, source_file_paths2 = read_audio_data(
		source_dir_path1,
		source_dir_path2,
		file_names1,
		file_names2,
		n_samples
	)

	x_video = read_video_data(
		source_dir_path1,
		source_dir_path2,
		file_names1,
		file_names2,
		n_samples
	)

	return x_audio, x_video, mixed_signals, source_file_paths1, source_file_paths2


def read_audio_data(source_dir_path1, source_dir_path2, source_file_names1, source_file_names2, n_samples):
	source_file_paths1 = [os.path.join(source_dir_path1, "audio", f + ".wav") for f in source_file_names1]
	source_file_paths2 = [os.path.join(source_dir_path2, "audio", f + ".wav") for f in source_file_names2]

	x = np.ndarray(shape=(n_samples, 64 * 100))
	mixed_signals = []

	for i in range(n_samples):
		signal1 = AudioSignal.from_wav_file(source_file_paths1[i])
		signal2 = AudioSignal.from_wav_file(source_file_paths2[i])

		mixed_signal = AudioMixer.mix([signal1, signal2], mixing_weights=[1, 1])

		x[i, :] = Speech.signal_to_mel_spectogram(mixed_signal).flatten()
		mixed_signals.append(mixed_signal)

	return x, mixed_signals, source_file_paths1[:n_samples], source_file_paths2[:n_samples]


def read_video_data(source_dir_path1, source_dir_path2, source_file_names1, source_file_names2, n_samples):
	source_file_paths1 = [os.path.join(source_dir_path1, "video", f + ".mpg") for f in source_file_names1]
	source_file_paths2 = [os.path.join(source_dir_path2, "video", f + ".mpg") for f in source_file_names2]

	face_detector = cv2.CascadeClassifier(os.path.join("res", "haarcascade_frontalface_alt.xml"))

	x = np.zeros(shape=(n_samples, 75, 128, 128), dtype=np.uint8)

	for i in range(n_samples):
		with VideoFileReader(source_file_paths1[i]) as reader:
			frames = reader.read_all_frames(convert_to_gray_scale=True)

		face_cropped_frames = np.zeros(shape=(75, 128, 128))
		for j in range(frames.shape[0]):
			faces = face_detector.detectMultiScale(frames[j, ])
			(face_x, face_y, face_width, face_height) = faces[0]
			face = frames[j, face_y: (face_y + face_height), face_x: (face_x + face_width)]

			face_cropped_frames[j, ] = cv2.resize(face, (128, 128))

		x[i, ] = face_cropped_frames

	return x


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

	x_audio, x_video, mixed_signals, source_file_paths1, source_file_paths2 = read_data(args.source_dir1, args.source_dir2)

	separator = SpeechSeparator(train_cache_path=args.train_cache)
	y_predicted = separator.predict(x_audio, x_video)

	source_signals = reconstruct_signals(y_predicted, sample_rate=44100)

	if os.path.exists(args.prediction_output_dir):
		shutil.rmtree(args.prediction_output_dir)

	os.mkdir(args.prediction_output_dir)

	for i in range(x_audio.shape[0]):
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
