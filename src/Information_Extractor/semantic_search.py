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
		queries_embeddings = embedder.encode(queries, convert_to_tensor=True)
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
