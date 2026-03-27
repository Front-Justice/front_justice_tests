import unicodedata
import re
from difflib import SequenceMatcher

def nfc_normalize(input_string: str) -> str:
	"""
	Cette fonction applique une normalisation unicode NFC à la chaîne de caractères voulue.
	:param input_string:
	:return:
	"""
	assert isinstance(input_string, str), (f"Input string should be a string. "
										   f"Actually: {type(input_string)}."
										   f"Current string: {input_string}")
	return unicodedata.normalize('NFC', input_string)


def strip_punctuation(string: str | None, debug=False) -> str | None:
	"""
	Cette fonction supprime la ponctuation en début et fin de chaîne
	:param string: la chaîne à nettoyer
	:return: la chaîne nettoyée
	"""
	if string is None:
		return None
	orig_string = string
	punctuation = "[\(\),;.!?\-:]"
	expression = "^" + punctuation + "\s{0,}|\s{0,}" + punctuation + "$"
	punct_regexp = re.compile(expression)
	string = string.strip()
	string = re.sub(punct_regexp, "", string)
	string = string.strip()
	if debug:
		print(f"|{orig_string}| -> |{string}|")
	return string


def check_word_in_list(word_list: list, target_word: str, sensibility=0.7) -> (bool, str | None):
	"""
	Cette fonction vérifie si un mot (pouvant présenter des coquilles) est présent dans une liste de mots
	:param sentence: la phrase cible
	:param target_word: le mot à chercher
	:return: vrai ou faux et le mot identifié (ou None)
	"""
	distances = []
	matching_words = []
	target_word = target_word.lower()
	for word in word_list:
		word_lower = word.lower()
		dist = similarite_ratcliff(word_lower, target_word)
		if dist > sensibility:
			matching_words.append(word)
			distances.append(dist)
	if len(distances) == 0:
		return False, target_word
	max_dist: int = distances.index(max(distances))
	return True, matching_words[max_dist]



def similarite_ratcliff(string_a, string_b):
	string_a = nfc_normalize(string_a)
	string_b = nfc_normalize(string_b)
	return SequenceMatcher(None, string_a, string_b).ratio()

