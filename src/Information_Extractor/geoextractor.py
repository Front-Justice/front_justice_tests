import json
import src.utils.utils as utils

class GeoExtractor():
	def __init__(self):
		with open("src/Information_Extractor/databases/referentiel_communes.json", "r") as input_json:
			self.geodict = json.load(input_json)


	def retrieve_coordinates(self, input_data):
		match = False

		for key, row in self.geodict.items():
			if input_data == row["nom_1801"]:
				print("Found you")
				print(row)
				latitude = row["latitude"]
				longitude = row["longitude"]
				current_name = row["nom_actuel"]
				match = True
			elif input_data == row["nom_1999"]:
				print("Found you 2")
				print(row)
				latitude = row["latitude"]
				longitude = row["longitude"]
				current_name = row["nom_actuel"]
				match = True
		if match == False:
			liste_des_communes_1801 = [item["nom_1801"] for key, item in self.geodict.items()]
			liste_des_communes_1999 = [item["nom_1999"] for key, item in self.geodict.items()]
			closest = utils.find_closest_word_in_list(liste_des_communes_1999, input_data)
			print(closest)


		return {
			"lat": latitude,
			"long": longitude,
			"nom_actuel": current_name
		}