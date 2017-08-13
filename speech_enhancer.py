import argparse
import os
import shutil
import random
from datetime import datetime

import numpy as np

from mediaio.audio_io import AudioSignal, AudioMixer
from dsp.spectrogram import MelConverter
from dataset import AudioVisualDataset


def enhance_speech(speaker_file_path, noise_file_path, speech_prediction_path, enhancement_threshold):
	print("enhancing mix of %s, %s" % (speaker_file_path, noise_file_path))

	speaker_source_signal = AudioSignal.from_wav_file(speaker_file_path)
	noise_source_signal = AudioSignal.from_wav_file(noise_file_path)
	predicted_speech_signal = AudioSignal.from_wav_file(speech_prediction_path)

	while noise_source_signal.get_number_of_samples() < speaker_source_signal.get_number_of_samples():
		noise_source_signal = AudioSignal.concat([noise_source_signal, noise_source_signal])

	noise_source_signal = noise_source_signal.slice(0, speaker_source_signal.get_number_of_samples())

	mixed_signal = AudioMixer.mix([speaker_source_signal, noise_source_signal])

	signals = [mixed_signal, predicted_speech_signal]
	max_length = max([signal.get_number_of_samples() for signal in signals])
	for signal in signals:
		signal.pad_with_zeros(max_length)

	mel_converter = MelConverter(mixed_signal.get_sample_rate())
	mixed_spectrogram = mel_converter.signal_to_mel_spectrogram(mixed_signal)
	predicted_speech_spectrogram = mel_converter.signal_to_mel_spectrogram(predicted_speech_signal)

	speech_enhancement_mask = np.zeros(shape=mixed_spectrogram.shape)
	speech_enhancement_mask[predicted_speech_spectrogram > enhancement_threshold] = 1

	enhanced_speech_spectrogram = mixed_spectrogram * speech_enhancement_mask
	enhanced_speech_signal = mel_converter.reconstruct_signal_from_mel_spectrogram(enhanced_speech_spectrogram)

	return mixed_signal, enhanced_speech_signal


def apply_speech_enhancement(dataset_dir, speaker_id, noise_dir, prediction_input_dir, enhancement_output_dir, enhancement_threshold):
	enhancement_output_dir = os.path.join(enhancement_output_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
	os.mkdir(enhancement_output_dir)

	for speaker_file_path, noise_file_path in list_source_pairs(dataset_dir, speaker_id, noise_dir):
		try:
			speaker_file_name = os.path.splitext(os.path.basename(speaker_file_path))[0]
			noise_file_name = os.path.splitext(os.path.basename(noise_file_path))[0]

			speech_enhancement_dir_path = os.path.join(enhancement_output_dir, speaker_file_name + "_" + noise_file_name)
			os.mkdir(speech_enhancement_dir_path)

			speech_prediction_path = os.path.join(prediction_input_dir, speaker_id, speaker_file_name + ".wav")
			mixed_signal, enhanced_speech_signal = enhance_speech(
				speaker_file_path, noise_file_path, speech_prediction_path, enhancement_threshold
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
	parser.add_argument("threshold", type=int)
	args = parser.parse_args()

	apply_speech_enhancement(
		args.dataset_dir, args.speaker, args.noise_dir, args.prediction_input_dir,
		args.enhancement_output_dir, args.threshold
	)


if __name__ == "__main__":
	main()
