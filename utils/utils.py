import json
import pickle

def check_if_missing(list_target, list_source):
	set1 = set(list_target)
	set2 = set(list_source)
	missing = list(sorted(set1 - set2))
	return missing

def pickle_object(obj, path):
	with open(path, "wb") as segmentation_as_file:
		pickle.dump(obj, segmentation_as_file, protocol=pickle.HIGHEST_PROTOCOL)

def unpickle_object(path):
	with open(path, "rb") as segmentation_as_file:
		return pickle.load(segmentation_as_file)

def save_as_dict(dictionnary, path):
	with open(path, 'w') as f:
		json.dump(dictionnary, f)


def load_json_to_dict(path):
	with open(path, 'r') as f:
		return json.load(f)

def get_name_from_path(path):
	basename = path.split('/')[-1].split('.')[0]
	dossier = "_".join(basename.split('_')[:-1])
	ident = basename.split('_')[-1]
	return dossier, int(ident)


def format_coordinates(coords):
	rounded = [round(item) for item in coords]
	return [[rounded[0], rounded[1]], [rounded[2], rounded[3]]]
