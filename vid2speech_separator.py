import argparse
import os
import shutil
import glob
from datetime import datetime

import numpy as np

from mediaio.audio_io import AudioSignal, AudioMixer
from dsp.spectogram import MelConverter
from audio_separator import list_audio_source_file_pairs


def predict(dataset_dir, vid2speech_prediction_dir, separation_output_dir, speakers, separation_threshold=5):
	source_file_pairs = list_audio_source_file_pairs(dataset_dir, speakers, max_pairs=10)

	separation_output_dir = os.path.join(separation_output_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
	os.mkdir(separation_output_dir)

	for source_file_path1, source_file_path2 in source_file_pairs:
		print("predicting mix of %s, %s" % (source_file_path1, source_file_path2))
		source_signal1 = AudioSignal.from_wav_file(source_file_path1)
		source_signal2 = AudioSignal.from_wav_file(source_file_path2)
		mixed_signal = AudioMixer.mix([source_signal1, source_signal2])

		mel_converter = MelConverter(mixed_signal.get_sample_rate())
		mixed_spectogram = mel_converter.signal_to_mel_spectogram(mixed_signal)

		source_name1 = os.path.splitext(os.path.basename(source_file_path1))[0]
		source_name2 = os.path.splitext(os.path.basename(source_file_path2))[0]

		vid2speech_prediction_path1 = glob.glob(os.path.join(vid2speech_prediction_dir, speakers[0], source_name1 + ".wav"))
		vid2speech_prediction_path2 = glob.glob(os.path.join(vid2speech_prediction_dir, speakers[1], source_name2 + ".wav"))

		if len(vid2speech_prediction_path1) != 1 or len(vid2speech_prediction_path2) != 1:
			continue

		vid2speech_signal1 = AudioSignal.from_wav_file(vid2speech_prediction_path1[0])
		vid2speech_signal2 = AudioSignal.from_wav_file(vid2speech_prediction_path2[0])

		spectogram1 = mel_converter.signal_to_mel_spectogram(vid2speech_signal1)
		spectogram2 = mel_converter.signal_to_mel_spectogram(vid2speech_signal2)

		if mixed_spectogram.shape[1] > spectogram1.shape[1]:
			mixed_spectogram = mixed_spectogram[:, :spectogram1.shape[1]]
		else:
			spectogram1 = spectogram1[:, :mixed_spectogram.shape[1]]
			spectogram2 = spectogram2[:, :mixed_spectogram.shape[1]]

		mask1 = np.zeros(shape=mixed_spectogram.shape)
		mask1[spectogram1 > (spectogram2 + separation_threshold)] = 1

		mask2 = np.zeros(shape=mixed_spectogram.shape)
		mask2[(spectogram1 + separation_threshold) < spectogram2] = 1

		# mask2 = 1 - mask1

		separated_spectogram1 = mixed_spectogram * mask1
		separated_spectogram2 = mixed_spectogram * mask2

		reconstructed_signal1 = mel_converter.reconstruct_signal_from_mel_spectogram(separated_spectogram1)
		reconstructed_signal2 = mel_converter.reconstruct_signal_from_mel_spectogram(separated_spectogram2)

		source_separation_dir_path = os.path.join(separation_output_dir, source_name1 + "_" + source_name2)
		os.mkdir(source_separation_dir_path)

		reconstructed_signal1.save_to_wav_file(os.path.join(source_separation_dir_path, "predicted1.wav"))
		reconstructed_signal2.save_to_wav_file(os.path.join(source_separation_dir_path, "predicted2.wav"))

		mixed_signal.save_to_wav_file(os.path.join(source_separation_dir_path, "mix.wav"))

		shutil.copy(source_file_path1, source_separation_dir_path)
		shutil.copy(source_file_path2, source_separation_dir_path)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_dir", type=str)
	parser.add_argument("vid2speech_prediction_dir", type=str)
	parser.add_argument("separation_output_dir", type=str)
	parser.add_argument("speakers", nargs=2, type=str)
	args = parser.parse_args()

	predict(args.dataset_dir, args.vid2speech_prediction_dir, args.separation_output_dir, args.speakers)


if __name__ == "__main__":
	main()
