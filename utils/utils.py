import json
import glob




def get_name_from_path(path):
	basename = path.split('/')[-1].split('.')[0]
	dossier = "_".join(basename.split('_')[:-1])
	ident = basename.split('_')[-1]
	return dossier, int(ident)


def simplify_coordinate(coords):
	return [round(item) for item in coords]
