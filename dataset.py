import os
import glob
import random


class AudioVisualDataset:

	def __init__(self, base_path):
		self._base_path = base_path

	def subset(self, speaker_ids, max_files=None, shuffle=False):
		video_file_paths = []

		for speaker_id in speaker_ids:
			video_file_paths.extend(glob.glob(os.path.join(self._base_path, speaker_id, "video", "*.mpg")))

		if shuffle:
			random.shuffle(video_file_paths)

		return AudioVisualSubset(video_file_paths[:max_files])

	def list_speakers(self):
		return os.listdir(self._base_path)


class AudioVisualSubset:

	def __init__(self, video_file_paths):
		self._video_file_paths = video_file_paths

	def video_paths(self):
		return self._video_file_paths

	def audio_paths(self):
		return [self._video_to_audio_path(v) for v in self._video_file_paths]

	@staticmethod
	def _video_to_audio_path(video_file_path):
		return video_file_path.replace("video", "audio").replace(".mpg", ".wav")
