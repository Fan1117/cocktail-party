import argparse
import os
import glob

import numpy as np
import mir_eval

from mediaio.audio_io import AudioSignal


def evaluate(separation_dir_path):
	sdrs = []
	sirs = []
	sars = []

	sample_dir_names = os.listdir(separation_dir_path)
	for sample_dir_name in sample_dir_names:
		sample_dir_path = os.path.join(separation_dir_path, sample_dir_name)
		source_paths = sorted(glob.glob(os.path.join(sample_dir_path, "source-*.wav")))
		estimated_paths = sorted(glob.glob(os.path.join(sample_dir_path, "estimated-*.wav")))

		source_signals = [AudioSignal.from_wav_file(f).get_data(channel_index=0) for f in source_paths]
		estimated_signals = [AudioSignal.from_wav_file(f).get_data(channel_index=0) for f in estimated_paths]

		source_data = np.stack(source_signals)
		estimated_data = np.stack(estimated_signals)

		# sometimes the estimated signal is shorter (due to some transformations)
		source_data = source_data[:, :estimated_data.shape[1]]

		sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(source_data, estimated_data, compute_permutation=True)
		print("SDR: %s | SIR: %s | SAR: %s" % (sdr, sir, sar))

		sdrs.append(sdr)
		sirs.append(sir)
		sars.append(sar)

	print("mean: SDR: %f | SIR: %f | SAR: %f" % (np.mean(sdrs), np.mean(sirs), np.mean(sars)))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("separation_dir", type=str)
	args = parser.parse_args()

	evaluate(args.separation_dir)


if __name__ == "__main__":
	main()



