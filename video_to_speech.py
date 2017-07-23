import argparse
import os
import shutil
from datetime import datetime

import numpy as np

from dataset import AudioVisualDataset
global data_processor
global VideoToSpeechNet


def preprocess(args):
	speaker_ids = list_speakers(args)
	dataset = AudioVisualDataset(args.dataset_dir)

	for speaker_id in speaker_ids:
		data_subset = dataset.subset([speaker_id], shuffle=True)

		video_samples, audio_samples = data_processor.preprocess_data(data_subset)
		video_samples = data_processor.normalize_video_samples(video_samples)

		preprocessed_speaker_path = os.path.join(args.preprocessed_dir, speaker_id)
		np.savez(preprocessed_speaker_path, video_samples=video_samples, audio_samples=audio_samples)


def train(args):
	speaker_ids = list_speakers(args)

	video_samples, audio_samples = load_preprocessed_samples(
		args.preprocessed_dir, speaker_ids, max_speaker_samples=5000, max_total_samples=100000
	)

	network = VideoToSpeechNet.build(video_samples.shape[1:], audio_samples.shape[1])
	network.train(video_samples, audio_samples)
	network.dump(args.model_cache, args.weights_cache)


def predict(args):
	speaker_ids = list_speakers(args)
	dataset = AudioVisualDataset(args.dataset_dir)

	prediction_output_dir = os.path.join(args.prediction_output_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
	os.mkdir(prediction_output_dir)

	for speaker_id in speaker_ids:
		video_samples, audio_samples = load_preprocessed_samples(
			args.preprocessed_dir, [speaker_id], max_speaker_samples=800
		)

		network = VideoToSpeechNet.load(args.model_cache, args.weights_cache)
		network.fine_tune(video_samples, audio_samples)

		speaker_prediction_dir = os.path.join(prediction_output_dir, speaker_id)
		os.mkdir(speaker_prediction_dir)

		video_file_paths = dataset.subset([speaker_id]).video_paths()
		for video_file_path in video_file_paths:
			try:
				video_sample = data_processor.normalize_video_samples(
					data_processor.preprocess_video_sample(video_file_path)
				)

				predicted_audio_sample = network.predict(video_sample)

				sample_name = os.path.splitext(os.path.basename(video_file_path))[0]

				reconstructed_signal = data_processor.reconstruct_audio_signal(predicted_audio_sample, sample_rate=44100)
				reconstructed_signal.save_to_wav_file(os.path.join(speaker_prediction_dir, "%s.wav" % sample_name))

				shutil.copy(video_file_path, speaker_prediction_dir)

			except Exception as e:
				print("failed to preprocess %s (%s). skipping" % (video_file_path, e))


def list_speakers(args):
	if args.speakers is None:
		dataset = AudioVisualDataset(args.dataset_dir)
		speaker_ids = dataset.list_speakers()
	else:
		speaker_ids = args.speakers

	if args.ignored_speakers is not None:
		for speaker_id in args.ignored_speakers:
			speaker_ids.remove(speaker_id)

	return speaker_ids


def load_preprocessed_samples(preprocessed_dir, speaker_ids, max_speaker_samples=None, max_total_samples=None):
	video_samples = []
	audio_samples = []

	for speaker_id in speaker_ids:
		with np.load(os.path.join(preprocessed_dir, speaker_id + ".npz")) as data:
			video_samples.append(data["video_samples"][:max_speaker_samples])
			audio_samples.append(data["audio_samples"][:max_speaker_samples])

	video_samples, audio_samples = np.concatenate(video_samples), np.concatenate(audio_samples)

	permutation = np.random.permutation(video_samples.shape[0])
	video_samples = video_samples[permutation]
	audio_samples = audio_samples[permutation]

	return video_samples[:max_total_samples], audio_samples[:max_total_samples]


def load_framework(model):
	global data_processor
	global VideoToSpeechNet

	if model == "vid2speech":
		from video2speech import data_processor
		from video2speech.network import VideoToSpeechNet
	else:
		from video2speech_vggface import data_processor
		from video2speech_vggface.network import VideoToSpeechNet


def main():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument("model", type=str, choices=["vid2speech", "vggface"])

	action_parsers = parser.add_subparsers()

	preprocess_parser = action_parsers.add_parser("preprocess")
	preprocess_parser.add_argument("dataset_dir", type=str)
	preprocess_parser.add_argument("preprocessed_dir", type=str)
	preprocess_parser.add_argument("--speakers", nargs="+", type=str)
	preprocess_parser.add_argument("--ignored_speakers", nargs="+", type=str)
	preprocess_parser.set_defaults(func=preprocess)

	train_parser = action_parsers.add_parser("train")
	train_parser.add_argument("preprocessed_dir", type=str)
	train_parser.add_argument("model_cache", type=str)
	train_parser.add_argument("weights_cache", type=str)
	train_parser.add_argument("--speakers", nargs="+", type=str)
	train_parser.add_argument("--ignored_speakers", nargs="+", type=str)
	train_parser.set_defaults(func=train)

	predict_parser = action_parsers.add_parser("predict")
	predict_parser.add_argument("dataset_dir", type=str)
	predict_parser.add_argument("preprocessed_dir", type=str)
	predict_parser.add_argument("model_cache", type=str)
	predict_parser.add_argument("weights_cache", type=str)
	predict_parser.add_argument("prediction_output_dir", type=str)
	predict_parser.add_argument("--speakers", nargs="+", type=str)
	predict_parser.add_argument("--ignored_speakers", nargs="+", type=str)
	predict_parser.set_defaults(func=predict)

	args = parser.parse_args()

	load_framework(args.model)
	args.func(args)

if __name__ == "__main__":
	main()
