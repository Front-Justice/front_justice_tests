import PIL
import party.pred
import PIL.Image as Image
import lightning.fabric as Fabric
import kraken.containers as Containers
from party.fusion import PartyModel
import src.utils.utils as utils
from yaspin import yaspin
import time

class PartyPredict:
	def __init__(self):
		self.fabric = Fabric.Fabric(accelerator="cuda",
							   precision="bf16-mixed")
		self.model = PartyModel.from_safetensors(
			"/home/mgl/Bureau/Travail/scripts_et_programmes/party/models/final.safetensors"
		)

	def create_baseline(self, coords, corresponding_image:str) -> Containers.Segmentation:
		"""
		Crée un objet kraken Segmentation à partir des coordonnées de la baseline.
		:param coords: les coordonnées
		:param corresponding_image: le nom de l'image
		:return:
		"""
		# En général, on ne transcrira qu'une ligne avec Party, mais dans certains cas on a besoin de plusieurs lignes
		# On va donc vérifier la profondeur des coordonnées: si elle est de 2, on a une seule ligne. En cas de plusieurs
		# lignes elle sera de 3.
		profondeur = utils.list_depth(coords)
		assert profondeur == 3, ("Les coordonnées doivent être encapsulées dans une liste même en cas de ligne unique:\n"
								 "[[[2611, 555], [3203, 555]]]")
		baseline = [Containers.BaselineLine(id='test', baseline=coord, boundary=None) for coord in coords]
		segmentation = Containers.Segmentation(type="baselines",
											   imagename=corresponding_image,
											   text_direction="horizontal-lr",
											   lines=baseline,
											   script_detection=False)
		return segmentation

	def inference(self, segmentation, image):
		prediction = party.pred.batched_pred(model=self.model, im=image, bounds=segmentation, fabric=self.fabric)
		lines = list(prediction)
		if len(lines) != 1:
			return lines
		else:
			return lines[0]

	def timed_party_inference(self, segmentation, image, objet_transcrit):
		"""
		Prédit avec party, en utilisant le spinner yaspin, et en calculant le temps d'inférence.
		:param segmentation:
		:param image:
		:param objet_transcrit:
		:return:
		"""
		with yaspin(text=f"Transcription de {objet_transcrit} avec Party") as sp:
			start = time.time()
			prediction = self.inference(segmentation=segmentation, image=image)
			end = time.time()
		print(f"Transcription de {objet_transcrit} faite en {end - start} secondes")
		return prediction

if __name__ == '__main__':
	corresponding_image = "/home/mgl/Bureau/Travail/scripts_et_programmes/party/11_J_77-0355.jpg"
	processed_baseline = [[1464, 3386],
						  [2517, 3396]]
	predictor = PartyPredict()
	as_image = PIL.Image.open(corresponding_image)
	segmentation = predictor.create_baseline(processed_baseline, corresponding_image)
	prediction = predictor.inference(segmentation=segmentation, image=as_image)
	print(prediction)


