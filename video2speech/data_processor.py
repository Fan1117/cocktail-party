import math
import multiprocessing

import numpy as np
import pickle

from facedetection.face_detection import FaceDetector
from mediaio.audio_io import AudioSignal
from mediaio.video_io import VideoFileReader
from spectrogram import MelConverter


def preprocess_video_sample(video_file_path, slice_duration_ms=330, mouth_height=50, mouth_width=100):
	print("preprocessing %s" % video_file_path)

	face_detector = FaceDetector()

	with VideoFileReader(video_file_path) as reader:
		cropped_frames = np.zeros(shape=(reader.get_frame_count(), mouth_height, mouth_width, 3), dtype=np.float32)

		for i in range(reader.get_frame_count()):
			frame = reader.read_next_frame()
			cropped_frames[i, :] = face_detector.crop_mouth(frame, bounding_box_shape=(mouth_width, mouth_height))

		frames_per_slice = int(math.ceil((float(slice_duration_ms) / 1000) * reader.get_frame_rate()))
		n_slices = int(float(reader.get_frame_count()) / frames_per_slice)

		slices = [
			cropped_frames[(i * frames_per_slice):((i + 1) * frames_per_slice)]
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

	mel_converter = MelConverter(audio_signal.get_sample_rate(), n_mel_freqs=128, freq_min_hz=0, freq_max_hz=4000)

	new_signal_length = int(math.ceil(
		float(audio_signal.get_number_of_samples()) / mel_converter.get_hop_length()
	)) * mel_converter.get_hop_length()

	audio_signal.pad_with_zeros(new_signal_length)

	mel_spectrogram = mel_converter.signal_to_mel_spectrogram(audio_signal)

	samples_per_slice = int((float(slice_duration_ms) / 1000) * audio_signal.get_sample_rate())
	spectrogram_samples_per_slice = int(samples_per_slice / mel_converter.get_hop_length())

	n_slices = int(mel_spectrogram.shape[1] / spectrogram_samples_per_slice)

	slices = [
		mel_spectrogram[:, (i * spectrogram_samples_per_slice):((i + 1) * spectrogram_samples_per_slice)].flatten()
		for i in range(n_slices)
	]

	return np.stack(slices)


def reconstruct_audio_signal(audio_sample, sample_rate):
	mel_converter = MelConverter(sample_rate, n_mel_freqs=128, freq_min_hz=0, freq_max_hz=4000)

	slice_mel_spectrograms = [audio_sample[i, :].reshape((mel_converter.get_n_mel_freqs(), -1)) for i in range(audio_sample.shape[0])]
	full_mel_spectrogram = np.concatenate(slice_mel_spectrograms, axis=1)

	return mel_converter.reconstruct_signal_from_mel_spectrogram(full_mel_spectrogram)


def preprocess_data(data):
	print("reading dataset...")

	thread_pool = multiprocessing.Pool(8)
	video_samples = thread_pool.map(try_preprocess_video_sample, data.video_paths())
	audio_samples = thread_pool.map(preprocess_audio_sample, data.audio_paths())

	invalid_sample_ids = [i for i, sample in enumerate(video_samples) if sample is None]
	video_samples = [sample for i, sample in enumerate(video_samples) if i not in invalid_sample_ids]
	audio_samples = [sample for i, sample in enumerate(audio_samples) if i not in invalid_sample_ids]

	return np.concatenate(video_samples), np.concatenate(audio_samples)


def normalize(video_samples, normalization_cache):
	normalization_data = __init_normalization_data(video_samples)

	with open(normalization_cache, 'wb') as normalization_cache_fd:
		pickle.dump(normalization_data, normalization_cache_fd)

	apply_normalization(video_samples, normalization_cache)


def apply_normalization(video_samples, normalization_cache):
	with open(normalization_cache, 'rb') as normalization_cache_fd:
		normalization_data = pickle.load(normalization_cache_fd)

	video_samples /= 255

	for channel in range(3):
		video_samples[:, :, :, :, channel] -= normalization_data.channel_means[channel]


def __init_normalization_data(video_samples):
	# video_samples: slices x frames_per_slice x height x width x channels
	channel_means = [video_samples[:, :, :, :, channel].mean() / 255 for channel in range(3)]

	return VideoNormalizationData(channel_means)


class VideoNormalizationData:

	def __init__(self, channel_means):
		self.channel_means = channel_means
