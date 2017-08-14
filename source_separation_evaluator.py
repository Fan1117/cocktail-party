import argparse
import os
import glob

import numpy as np
import mir_eval

from mediaio.audio_io import AudioSignal


def evaluate(source_file_paths, estimated_file_paths):
	source_signals = [AudioSignal.from_wav_file(f) for f in source_file_paths]
	estimated_signals = [AudioSignal.from_wav_file(f) for f in estimated_file_paths]

	signals = source_signals + estimated_signals
	max_length = max([signal.get_number_of_samples() for signal in signals])
	for signal in signals:
		signal.pad_with_zeros(max_length)

	source_data = [signal.get_data(channel_index=0) for signal in source_signals]
	estimated_data = [signal.get_data(channel_index=0) for signal in estimated_signals]

	source_data = np.stack(source_data)
	estimated_data = np.stack(estimated_data)

	return mir_eval.separation.bss_eval_sources(source_data, estimated_data, compute_permutation=True)


def evaluate_all(separation_dir_path):
	sdrs = []
	sirs = []
	sars = []

	sample_dir_names = os.listdir(separation_dir_path)
	for sample_dir_name in sample_dir_names:
		sample_dir_path = os.path.join(separation_dir_path, sample_dir_name)
		source_file_paths = sorted(glob.glob(os.path.join(sample_dir_path, "source-*.wav")))
		estimated_file_paths = sorted(glob.glob(os.path.join(sample_dir_path, "estimated-*.wav")))

		sdr, sir, sar, _ = evaluate(source_file_paths, estimated_file_paths)
		print("SDR: %s | SIR: %s | SAR: %s" % (sdr, sir, sar))

		sdrs.append(sdr)
		sirs.append(sir)
		sars.append(sar)

	print("mean: SDR: %f | SIR: %f | SAR: %f" % (np.mean(sdrs), np.mean(sirs), np.mean(sars)))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("separation_dir", type=str)
	args = parser.parse_args()

	evaluate_all(args.separation_dir)


if __name__ == "__main__":
	main()



