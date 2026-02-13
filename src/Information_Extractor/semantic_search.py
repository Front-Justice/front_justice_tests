import torch
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("dangvantuan/sentence-camembert-large")


def retrieve_most_similar_sentence(sentence:str, queries:list):
	"""
	Cette fonction renvoie la phrase la plus proche sémantiquement d'une liste de phrases.
	:param sentence:
	:param queries:
	:return: La phrase la plus proche
	"""

	queries_embeddings = embedder.encode(queries, convert_to_tensor=True)
	top_k = min(5, len(queries))
	query_embedding = embedder.encode(sentence, convert_to_tensor=True)
	similarity_scores = embedder.similarity(query_embedding, queries_embeddings)[0]
	scores, indices = torch.topk(similarity_scores, k=top_k)
	return queries[indices[0]]
