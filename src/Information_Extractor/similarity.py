import torch
import src.utils.utils as utils


def retrieve_most_similar_sentence(sentence:str, queries:list, embedder, mode="semantic"):
	"""
	Cette fonction renvoie la phrase la plus proche sémantiquement d'une liste de phrases.
    Elle utilise une méthode de classification 0-shot (topk closest sentences)
	:param sentence:
	:param queries:
	:return: La phrase la plus proche
	"""
	if mode == "semantic":
		try:
			queries_embeddings = embedder.encode(queries, convert_to_tensor=True)
		except IndexError:
			return None
		top_k = min(5, len(queries))
		query_embedding = embedder.encode(sentence, convert_to_tensor=True)
		similarity_scores = embedder.similarity(query_embedding, queries_embeddings)[0]
		scores, indices = torch.topk(similarity_scores, k=top_k)
		return queries[indices[0]]
	else:
		distances = []
		for query in queries:
			distances.append(utils.levensthein_distance(sentence, query))
		return queries[distances.index(min(distances))]


def find_closest_word_in_list(word_list: list, target_word: str, replacement_mapping: dict = None, load_file=False) -> list:
	"""
	Cette fonction cherche le mot le plus proche dans une liste de mots
	:param sentence: la phrase cible
	:param target_word: le mot à chercher
	:param replacement_mapping: un mapping des caractères à modifier {"orig": "reg"}
	:return: le mot le plus proche et les distances
	"""
	if load_file:
		with open(word_list, "r") as input_file:
			list_of_words = [item.replace("\n", "") for item in input_file.readlines()]
		word_list = list_of_words
	distances = []
	target_word = target_word.lower()
	print(word_list)
	if replacement_mapping:
		for key, value in replacement_mapping.items():
			word_lower = target_word.replace(key, value)
	for word in word_list:
		if word is None:
			distances.append(99)
			continue
		word_lower = word.lower()
		if replacement_mapping:
			for key, value in replacement_mapping.items():
				word_lower = word_lower.replace(key, value)
		dist = utils.weighted_levenshtein_distance(word_lower, target_word)
		distances.append(dist)
	try:
		min_dist_index = distances.index(min(distances))
	except ValueError:
		return None, None
	return word_list[min_dist_index], min(distances)