import os


def list_dir_by_name(dir_path):
	return sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path)])
