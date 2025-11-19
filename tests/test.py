import unittest
import src.Information_Extractor.utils as utils
import re

class UtilsTest(unittest.TestCase):
	def test_split_before_keep_delimiter(self):
		test_sentence = "rendu par le CONSEIL DE GUERRE permanent du Q. G. de la 2^e Armée seant aux Armée"
		regexp = re.compile("s[ée]ant")
		result = utils.split_before_keep_delimiter(target_string=test_sentence, delimiter=regexp)
		print(result)
		self.assertEqual(result, ["rendu par le CONSEIL DE GUERRE permanent du Q. G. de la 2^e Armée",
								 "seant aux Armée"])



if __name__ == '__main__':
	unittest.main()