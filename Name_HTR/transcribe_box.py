
from kraken.containers import (BaselineLine, BaselineOCRRecord, BBoxLine,
                               BBoxOCRRecord, Segmentation)
import kraken.rpred as rpred
import kraken.kraken
import kraken.blla as blla
from kraken import binarization
import PIL.Image as Image
from kraken.lib.models import load_any
import lxml.etree as ET
from kraken.lib import vgsl
import glob

model = load_any("../../../inference/dataset/models/250p_best.mlmodel")
segmodel = vgsl.TorchVGSLModel.load_model("../../../inference/dataset/models/lignes.mlmodel")


def rpred_bbox(overfit_line, baseline_position):
        """
        Tests recognition with default tag.
        """
        simple_bl_seg = Segmentation(type='baselines',
                                          imagename = 'jacquot.png',
                                          lines=[BaselineLine(id='foo',
                                                              baseline=[
                                                                  [1, round(overfit_line.size[1]*baseline_position)],
                                                                  [overfit_line.size[0] - 1, round(overfit_line.size[1]*baseline_position)]
                                                              ],
                                                              boundary = [[1, 1],
                                                                          [overfit_line.size[0] -1, 1],
                                                                          [overfit_line.size[0] - 1, overfit_line.size[1] - 1],
                                                                          [1, overfit_line.size[1] - 1]
                                                                          ]
                                                              )],
                                           text_direction='horizontal-lr',
                                           script_detection=False
                                         )
        simple_box_seg = Segmentation(type='bbox',
                                           imagename = 'jacquot.png',
                                           lines=[BBoxLine(id='foo',
                                                           bbox=[0, 0, overfit_line.size[0], overfit_line.size[1]])],
                                           text_direction='horizontal-lr',
                                           script_detection=False
                                          )
        pred = rpred.rpred(model, overfit_line, simple_box_seg, True)
        if len(pred) > 0:
            return pred
        else:
            return None



for xml_file in glob.glob("/media/mgl/Disque_B/Front_Justice/roboflow/FJ_Page_1-1/data/*.xml"):
    corresponding_image_name = xml_file.split("/")[-1].split(".rf")[0].split("_jpg")[0]
    corresponding_image = xml_file.replace(".xml", ".jpg")
    as_tree = ET.parse(xml_file)
    noms_de_soldat = as_tree.xpath("//object/name[text() = 'Nom du soldat']")
    as_image = Image.open(corresponding_image)
    for nom in noms_de_soldat:
        print("\n\n\nNew Name")
        coords = nom.getparent().xpath("bndbox")[0]
        xmin, ymin, xmax, ymax = [int(item) for item in (coords.xpath("xmin/text()")[0],
                                  coords.xpath("ymin/text()")[0],
                                  coords.xpath("xmax/text()")[0],
                                  coords.xpath("ymax/text()")[0])]
        cropped_img = as_image.crop((xmin, ymin, xmax, ymax))
        predicted_name = rpred_bbox(cropped_img, baseline_position=.6)
        as_text = next(predicted_name)
        if predicted_name is not None:
            try:
                cropped_img.save(f"results/{as_text}.jpg")
            except ValueError as e:
                continue