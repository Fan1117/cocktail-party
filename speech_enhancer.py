import argparse
import os
import shutil
import random
from datetime import datetime

import numpy as np

from mediaio.audio_io import AudioSignal, AudioMixer
from mediaio.dsp.spectrogram import MelConverter
from dataset import AudioVisualDataset


def enhance_speech(speaker_file_path, noise_file_path, speech_prediction_path, speech_profile):
	print("enhancing mix of %s, %s" % (speaker_file_path, noise_file_path))

	speaker_source_signal = AudioSignal.from_wav_file(speaker_file_path)
	noise_source_signal = AudioSignal.from_wav_file(noise_file_path)

	while noise_source_signal.get_number_of_samples() < speaker_source_signal.get_number_of_samples():
		noise_source_signal = AudioSignal.concat([noise_source_signal, noise_source_signal])

	noise_source_signal = noise_source_signal.slice(0, speaker_source_signal.get_number_of_samples())
	mixed_signal = AudioMixer.mix([speaker_source_signal, noise_source_signal])

	predicted_speech_signal = AudioSignal.from_wav_file(speech_prediction_path)

	signals = [mixed_signal, predicted_speech_signal]
	max_length = max([signal.get_number_of_samples() for signal in signals])
	for signal in signals:
		signal.pad_with_zeros(max_length)

	mel_converter = MelConverter(mixed_signal.get_sample_rate(), n_mel_freqs=128, freq_min_hz=0, freq_max_hz=4000)
	mixed_spectrogram, original_phase = mel_converter.signal_to_mel_spectrogram(mixed_signal, get_phase=True)
	predicted_speech_spectrogram = mel_converter.signal_to_mel_spectrogram(predicted_speech_signal)

	speech_enhancement_mask = np.zeros(shape=mixed_spectrogram.shape)

	thresholds = np.zeros(shape=(speech_enhancement_mask.shape[0]))
	for f in range(speech_enhancement_mask.shape[0]):
		thresholds[f] = np.percentile(speech_profile[f, :], 85)

	for f in range(speech_enhancement_mask.shape[0]):
		for t in range(speech_enhancement_mask.shape[1]):
			if predicted_speech_spectrogram[f, t] > thresholds[f]:
				speech_enhancement_mask[f, t] = 1
				continue

	enhanced_speech_spectrogram = mixed_spectrogram * speech_enhancement_mask
	enhanced_speech_signal = mel_converter.reconstruct_signal_from_mel_spectrogram(enhanced_speech_spectrogram, original_phase)

	return mixed_signal, enhanced_speech_signal


def build_speech_profile(speaker_speech_dir, max_files=50):
	print("building speech profile...")

	speech_file_paths = [os.path.join(speaker_speech_dir, f) for f in os.listdir(speaker_speech_dir)][:max_files]
	speech_signals = [AudioSignal.from_wav_file(f) for f in speech_file_paths]

	mel_converter = MelConverter(speech_signals[0].get_sample_rate(), n_mel_freqs=128, freq_min_hz=0, freq_max_hz=4000)
	speech_spectrograms = [mel_converter.signal_to_mel_spectrogram(signal) for signal in speech_signals]

	speech_profile = np.concatenate(speech_spectrograms, axis=1)
	return speech_profile


def apply_speech_enhancement(dataset_dir, speaker_id, noise_dir, prediction_input_dir, enhancement_output_dir):
	enhancement_output_dir = os.path.join(enhancement_output_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
	os.mkdir(enhancement_output_dir)

	speech_profile = build_speech_profile(os.path.join(prediction_input_dir, speaker_id))

	for speaker_file_path, noise_file_path in list_source_pairs(dataset_dir, speaker_id, noise_dir):
		try:
			speaker_file_name = os.path.splitext(os.path.basename(speaker_file_path))[0]
			noise_file_name = os.path.splitext(os.path.basename(noise_file_path))[0]

			speech_enhancement_dir_path = os.path.join(enhancement_output_dir, speaker_file_name + "_" + noise_file_name)
			os.mkdir(speech_enhancement_dir_path)

			speech_prediction_path = os.path.join(prediction_input_dir, speaker_id, speaker_file_name + ".wav")
			mixed_signal, enhanced_speech_signal = enhance_speech(
				speaker_file_path, noise_file_path, speech_prediction_path, speech_profile
			)

			shutil.copy(speaker_file_path, os.path.join(speech_enhancement_dir_path, "source.wav"))
			enhanced_speech_signal.save_to_wav_file(os.path.join(speech_enhancement_dir_path, "enhanced.wav"))
			mixed_signal.save_to_wav_file(os.path.join(speech_enhancement_dir_path, "mixture.wav"))

		except Exception as e:
			print("failed to enhance (%s). skipping" % e)


def list_source_pairs(dataset_dir, speaker_id, noise_dir):
	dataset = AudioVisualDataset(dataset_dir)
	speaker_file_paths = dataset.subset([speaker_id], max_files=20, shuffle=True).audio_paths()
	noise_file_paths = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir)]

	random.shuffle(speaker_file_paths)
	random.shuffle(noise_file_paths)

	return zip(speaker_file_paths, noise_file_paths)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_dir", type=str)
	parser.add_argument("speaker", type=str)
	parser.add_argument("noise_dir", type=str)
	parser.add_argument("prediction_input_dir", type=str)
	parser.add_argument("enhancement_output_dir", type=str)
	args = parser.parse_args()

	apply_speech_enhancement(
		args.dataset_dir, args.speaker, args.noise_dir, args.prediction_input_dir, args.enhancement_output_dir
	)


if __name__ == "__main__":
	main()
