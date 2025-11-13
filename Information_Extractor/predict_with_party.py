# Ce script utilise un modèle pour transcrire une partie de ligne, pratique pour des vérifications
# et du double-check (un passage par kraken, et on repasse avec party sur les segments critiques)



import PIL
import party.pred
import lxml.etree as ET
import PIL.Image as Image
import lightning.fabric as Fabric
import kraken.containers as Containers
from party.fusion import PartyModel
import kraken

alto_nspace = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}

def main(path):
	xml_tree = ET.parse(path)
	corresponding_image = path.replace(".xml", ".jpg")
	lines = xml_tree.xpath("//alto:TextLine", namespaces=alto_nspace)
	texts = xml_tree.xpath("//alto:TextLine/alto:String/@CONTENT", namespaces=alto_nspace)
	correct_line_idx = next(idx for idx, item in enumerate(texts) if "Maddi" in item)
	polygon = lines[correct_line_idx].xpath("alto:Shape/alto:Polygon/@POINTS", namespaces=alto_nspace)
	processed_polygon = [round(int(item)) for item in polygon[0].split(" ")]
	correct_baseline = lines[correct_line_idx].xpath("@BASELINE")
	processed_baseline = [round(int(item)) for item in correct_baseline[0].split(" ")]
	as_image = PIL.Image.open(corresponding_image)
	# TODO: Ici il manque une fonction pour passer de [1464, 3386, 2517, 3396]
	# TODO:  à [[1464, 3386], [2517, 3396]]

	# TODO: il manquerait ici la fonction pour identifier la partie de la ligne qui nous intéresse
	# TODO: (par exemple, celle qui comprend le nom du soldat)
	processed_baseline = [[1464, 3386], [2517, 3396]]
	fabric = Fabric.Fabric(accelerator="auto",
					devices=16,
					precision="bf16-mixed")
	model = PartyModel.from_safetensors("models/final.safetensors")
	baseline = Containers.BaselineLine(id='test', baseline=processed_baseline, boundary=None)
	segmentation = Containers.Segmentation(type="baselines",
										   imagename=corresponding_image,
										   text_direction="horizontal-lr",
										   lines=[baseline],
										   script_detection=False)
	prediction = party.pred.batched_pred(model=model, im=as_image, bounds=segmentation, fabric=fabric)
	line = next(prediction)
	print(line)
	print(line.cuts)
	print(line.confidences)

if __name__ == '__main__':
	filename = "11_J_77-0355.xml"
	main(filename)


