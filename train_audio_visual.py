import argparse
import os

import cv2
import numpy as np

from mediaio.audio_io import AudioSignal, AudioMixer
from mediaio.video_io import VideoFileReader
from source_separator.audio_visual import SpeechSeparator
from source_separator.speech import Speech


def read_data(source_dir_path1, source_dir_path2, max_samples):
	print("reading dataset...")

	audio_dir_path1 = os.path.join(source_dir_path1, "audio")
	audio_dir_path2 = os.path.join(source_dir_path2, "audio")

	file_names1 = [os.path.splitext(f)[0] for f in os.listdir(audio_dir_path1)]
	file_names2 = [os.path.splitext(f)[0] for f in os.listdir(audio_dir_path2)]

	n_samples = min(len(file_names1), len(file_names2), max_samples)

	x_audio, y = read_audio_data(
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

	return x_audio, x_video, y


def read_audio_data(source_dir_path1, source_dir_path2, source_file_names1, source_file_names2, n_samples):
	source_file_paths1 = [os.path.join(source_dir_path1, "audio", f + ".wav") for f in source_file_names1]
	source_file_paths2 = [os.path.join(source_dir_path2, "audio", f + ".wav") for f in source_file_names2]

	x = np.ndarray(shape=(n_samples, 64 * 100))
	y = np.ndarray(shape=(n_samples, 64 * 100))

	for i in range(n_samples):
		signal1 = AudioSignal.from_wav_file(source_file_paths1[i])
		signal2 = AudioSignal.from_wav_file(source_file_paths2[i])

		mixed_signal = AudioMixer.mix([signal1, signal2], mixing_weights=[1, 1])

		x[i, :] = Speech.signal_to_mel_spectogram(mixed_signal).flatten()
		y[i, :] = Speech.signal_to_mel_spectogram(signal1).flatten()

	return x, y


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


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("train_source_dir1", type=str)
	parser.add_argument("train_source_dir2", type=str)
	parser.add_argument("test_source_dir1", type=str)
	parser.add_argument("test_source_dir2", type=str)
	parser.add_argument("train_cache", type=str)
	args = parser.parse_args()

	x_audio_train, x_video_train, y_train = read_data(args.train_source_dir1, args.train_source_dir2, max_samples=500)
	# np.savez_compressed("xy.npz", x_audio_train=x_audio_train, x_video_train=x_video_train, y_train=y_train)

	x_audio_test, x_video_test, y_test = read_data(args.test_source_dir1, args.test_source_dir2, max_samples=10)

	separator = SpeechSeparator(train_cache_path=args.train_cache)
	separator.train(x_audio_train, x_video_train, y_train)
	score = separator.evaluate(x_audio_test, x_video_test, y_test)
	print("score: %.2f" % score)

if __name__ == "__main__":
	main()
