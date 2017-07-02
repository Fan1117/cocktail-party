import argparse
import os
import shutil
import glob
import random
from datetime import datetime

import numpy as np

from mediaio.audio_io import AudioSignal, AudioMixer
from dsp.spectrogram import MelConverter


def enhance(dataset_dir, speaker_id, noise_dir, speech_prediction_dir,
			enhancement_output_dir, max_pairs=None, enhancement_threshold=70):

	speaker_files = glob.glob(os.path.join(dataset_dir, speaker_id, "audio", "*.wav"))
	noise_files = glob.glob(os.path.join(noise_dir, "*.wav"))

	random.shuffle(speaker_files)
	random.shuffle(noise_files)

	source_file_pairs = zip(speaker_files, noise_files)[:max_pairs]

	enhancement_output_dir = os.path.join(enhancement_output_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
	os.mkdir(enhancement_output_dir)

	for speaker_file_path, noise_file_path in source_file_pairs:
		print("predicting mix of %s, %s" % (speaker_file_path, noise_file_path))

		speaker_source_signal = AudioSignal.from_wav_file(speaker_file_path)
		noise_source_signal = AudioSignal.from_wav_file(noise_file_path)

		while noise_source_signal.get_number_of_samples() < speaker_source_signal.get_number_of_samples():
			noise_source_signal = AudioSignal.concat([noise_source_signal, noise_source_signal])

		noise_source_signal = noise_source_signal.slice(0, speaker_source_signal.get_number_of_samples())

		mixed_signal = AudioMixer.mix([speaker_source_signal, noise_source_signal])

		mel_converter = MelConverter(mixed_signal.get_sample_rate())
		mixed_spectrogram = mel_converter.signal_to_mel_spectrogram(mixed_signal)

		speaker_file_name = os.path.splitext(os.path.basename(speaker_file_path))[0]
		noise_file_name = os.path.splitext(os.path.basename(noise_file_path))[0]

		speech_prediction_path = glob.glob(os.path.join(speech_prediction_dir, speaker_id, speaker_file_name + ".wav"))
		if len(speech_prediction_path) != 1:
			continue

		predicted_speech_signal = AudioSignal.from_wav_file(speech_prediction_path[0])
		predicted_speech_spectrogram = mel_converter.signal_to_mel_spectrogram(predicted_speech_signal)

		if mixed_spectrogram.shape[1] > predicted_speech_spectrogram.shape[1]:
			mixed_spectrogram = mixed_spectrogram[:, :predicted_speech_spectrogram.shape[1]]
		else:
			predicted_speech_spectrogram = predicted_speech_spectrogram[:, :mixed_spectrogram.shape[1]]

		speech_enhancement_mask = np.zeros(shape=mixed_spectrogram.shape)
		speech_enhancement_mask[predicted_speech_spectrogram > enhancement_threshold] = 1

		enhanced_speech_spectrogram = mixed_spectrogram * speech_enhancement_mask

		reconstructed_speech_signal = mel_converter.reconstruct_signal_from_mel_spectrogram(enhanced_speech_spectrogram)

		speech_enhancement_dir_path = os.path.join(enhancement_output_dir, speaker_file_name + "_" + noise_file_name)
		os.mkdir(speech_enhancement_dir_path)

		reconstructed_speech_signal.save_to_wav_file(os.path.join(speech_enhancement_dir_path, "speech.wav"))
		mixed_signal.save_to_wav_file(os.path.join(speech_enhancement_dir_path, "mix.wav"))

		shutil.copy(speaker_file_path, speech_enhancement_dir_path)
		shutil.copy(noise_file_path, speech_enhancement_dir_path)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_dir", type=str)
	parser.add_argument("speaker", type=str)
	parser.add_argument("noise_dir", type=str)
	parser.add_argument("speech_prediction_dir", type=str)
	parser.add_argument("enhancement_output_dir", type=str)
	args = parser.parse_args()

	enhance(args.dataset_dir, args.speaker, args.noise_dir, args.speech_prediction_dir, args.enhancement_output_dir, max_pairs=10)


if __name__ == "__main__":
	main()
