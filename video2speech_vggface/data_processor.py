import math
import multiprocessing

import numpy as np
import cv2

from keras_vggface.vggface import VGGFace

from facedetection.face_detection import FaceDetector
from mediaio.audio_io import AudioSignal
from mediaio.video_io import VideoFileReader
from mediaio.dsp.spectrogram import MelConverter


vgg_model = VGGFace(weights="vggface", include_top=False, pooling="avg")


def preprocess_video_sample(video_file_path, slice_duration_ms=330):
	print("preprocessing %s" % video_file_path)

	face_detector = FaceDetector()

	with VideoFileReader(video_file_path) as reader:
		features = np.zeros(shape=(reader.get_frame_count(), 512), dtype=np.float32)
		for i in range(reader.get_frame_count()):
			frame = reader.read_next_frame()

			face = face_detector.crop_face(frame)
			face = cv2.resize(face, (224, 224))

			x = np.expand_dims(face, axis=0)
			x = x.astype(np.float32)

			x[:, :, :, 0] -= 93.5940
			x[:, :, :, 1] -= 104.7624
			x[:, :, :, 2] -= 129.1863

			features[i, :] = vgg_model.predict(x)

		frames_per_slice = int(math.ceil((float(slice_duration_ms) / 1000) * reader.get_frame_rate()))
		n_slices = int(float(reader.get_frame_count()) / frames_per_slice)

		slices = [
			features[(i * frames_per_slice):((i + 1) * frames_per_slice)]
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
	video_samples = map(try_preprocess_video_sample, data.video_paths())
	audio_samples = thread_pool.map(preprocess_audio_sample, data.audio_paths())

	invalid_sample_ids = [i for i, sample in enumerate(video_samples) if sample is None]
	video_samples = [sample for i, sample in enumerate(video_samples) if i not in invalid_sample_ids]
	audio_samples = [sample for i, sample in enumerate(audio_samples) if i not in invalid_sample_ids]

	return np.concatenate(video_samples), np.concatenate(audio_samples)


def normalize(video_samples, normalization_cache):
	return


def apply_normalization(video_samples, normalization_cache):
	return
