import argparse
import os
import subprocess
import tempfile
import re
import uuid

import numpy as np

from mediaio import ffmpeg


def pesq(pesq_bin_path, source_file_path, estimation_file_path):
	temp_dir = tempfile.gettempdir()

	temp_source_path = os.path.join(temp_dir, str(uuid.uuid4()) + ".wav")
	temp_estimation_path = os.path.join(temp_dir, str(uuid.uuid4()) + ".wav")

	ffmpeg.downsample(source_file_path, temp_source_path, sample_rate=16000)
	ffmpeg.downsample(estimation_file_path, temp_estimation_path, sample_rate=16000)

	output = subprocess.check_output(
		[pesq_bin_path, "+16000", temp_source_path, temp_estimation_path]
	)

	match = re.search("\(Raw MOS, MOS-LQO\):\s+= ([0-9.]+?)\t([0-9.]+?)$", output, re.MULTILINE)
	mos = float(match.group(1))
	moslqo = float(match.group(2))

	os.remove(temp_source_path)
	os.remove(temp_estimation_path)

	return mos, moslqo


def evaluate(enhancement_dir_path, pesq_bin_path):
	enhancement_pesqs = []
	noisy_pesqs = []

	sample_dir_names = os.listdir(enhancement_dir_path)
	for sample_dir_name in sample_dir_names:
		sample_dir_path = os.path.join(enhancement_dir_path, sample_dir_name)
		source_file_path = os.path.join(sample_dir_path, "source.wav")
		enhanced_file_path = os.path.join(sample_dir_path, "enhanced.wav")
		mixture_file_path = os.path.join(sample_dir_path, "mixture.wav")

		mos, _ = pesq(pesq_bin_path, source_file_path, enhanced_file_path)
		print("enhancement pesq: %f" % mos)

		enhancement_pesqs.append(mos)

		mos, _ = pesq(pesq_bin_path, source_file_path, mixture_file_path)
		print("noisy pesq: %f" % mos)

		noisy_pesqs.append(mos)

	print("##############################")
	print("enhancement mean pesq: %f" % np.mean(enhancement_pesqs))
	print("noisy mean pesq: %f" % np.mean(noisy_pesqs))
	print("##############################")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("enhancement_dir", type=str)
	parser.add_argument("pesq_bin_path", type=str)
	args = parser.parse_args()

	evaluate(args.enhancement_dir, args.pesq_bin_path)


if __name__ == "__main__":
	main()
