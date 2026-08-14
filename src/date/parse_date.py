import re
import src.date.parser as parser
import src.date.lexer as lexer
import src.date.utils as utils
import json


class Date:
	"""
	Objet de classe date, permettant de manipuler les items de date (mois, mois_annee, annee, jour_mois_annee)
	et de les convertir en format jjmmyyyy
	"""
	def __init__(self):

		self.number_dict = {"un": 1,
							"deux": 2,
							"trois": 3,
							"quatre": 4,
							"cinq": 5,
							"six": 6,
							"sept": 7,
							"huit": 8,
							"neuf": 9,
							"dix": 10,
							"onze": 11,
							"douze": 12,
							"treize": 13,
							"quatorze": 14,
							"quinze": 15,
							"seize": 16,
							"dix sept":17,
							"dix huit": 18,
							"dix neuf": 19,
							"vingt": 20,
							"vingt-et-un": 21,
							"vingt deux": 22,
							"vingt trois": 23,
							"vingt quatre": 24,
							"vingt cinq": 25,
							"vingt six": 26,
							"vingt sept": 27,
							"vingt huit": 28,
							"vingt neuf": 29,
							"trente": 30,
							"trente et un": 31,
							"mil": 1000,
							"cent": 100}

		self.month_dict = {"janvier": "01",
						   "février": "02",
						   "mars": "03",
						   "avril": "04",
						   "mai": "05",
						   "juin": "06",
						   "juillet": "07",
						   "août": "08",
						   "septembre": "09",
						   "octobre": "10",
						   "novembre": "11",
						   "décembre": "12",
						   }

		self.month: str = ""
		self.year: str = ""
		self.day: str = ""
		self.converted_date = None

	def clean_date(self):
		"""
		Cette fonction nettoie la date formatée et supprime les / superflus
		:return:
		"""
		self.converted_date = self.converted_date.replace("//", "/")
		begin_regexp = re.compile(r"^/")
		end_regexp = re.compile(r"/$")
		self.converted_date = re.sub(begin_regexp, "", str(self.converted_date))
		self.converted_date = re.sub(end_regexp, "", str(self.converted_date))

	def convert_to_dd_mm_yyyy(self):
		self.month = self.month_dict[self.month] if self.month != "" else ""
		if len(str(self.day)) == 1:
			self.day = f"0{self.day}"
		self.converted_date = f"{self.day}/{self.month}/{self.year}"
		self.clean_date()
		return self.converted_date

def second_pass(parsed_tree):
	"""
	Seconde passe après le parsing récursif de l'arbre. On est devant un objet linéaire
	que l'on va convertir pour avoir la date la plus complète possible, et les
	alternatives (range, and, range/and, approx, etc)
	:param parsed_tree:
	:return:
	"""
	if isinstance(parsed_tree, str):
		return {"when": parsed_tree}
	elif isinstance(parsed_tree, dict):
		# Les dictionnaires enchassés ne sont pas gérés ici
		return parsed_tree
	elif isinstance(parsed_tree, list):
		# Si la longueur est 3, on a {alternative}, mois, année
		if len(parsed_tree) > 1:
			result = []
			month_and_or_year = "/".join(parsed_tree[1:])
			# On considère que le dictionnaire ne peut apparaître qu'en début d'expression
			dictionnary = parsed_tree[0]
			dict_key = next(iter(dictionnary))
			dict_values = next(iter(dictionnary.values()))
			if dict_key == "range":
				for item in dict_values:
					correct_date = f"{item}/{month_and_or_year}"
					result.append(correct_date)
				return {"range": result}
			elif dict_key == "range/and":
				for item in dict_values:
					correct_date = f"{item}/{month_and_or_year}"
					result.append(correct_date)
				return {"range/and": result}
			elif dict_key == "and":
				for item in dict_values:
					correct_date = f"{item}/{month_and_or_year}"
					result.append(correct_date)
				return {"and": result}

def parse_ast(ast) -> str|dict:
	# On fait 2 cas principaux: le premier cas où la complexité est absente: c'est
	# le cas des "dates atomiques", à savoir un dictionnaire pouvant comprendre un jour, un mois, une année
	if isinstance(ast, dict) and all([item not in ast for item in ['range', 'and', 'courant', 'andor']]):
		myDate = Date()
		for key, value in ast.items():
			if key == "mois":
				myDate.month = value
			elif key == "annee":
				myDate.year = value
			elif key == "jour":
				myDate.day = value
		return myDate.convert_to_dd_mm_yyyy()
	else:
		# Si l'ast a une longueur de 1, on a une traitement "simple" à effectuer sur les items de cet ast
		if len(ast) == 1:
			key = next(iter(ast))
			value = next(iter(ast.values()))
			if key == "and":
				out_dict = {"and": []}
				for item in ast[key]:
					out_dict["and"].append(parse_ast(item))
				return out_dict
			elif key == "courant":
				out_dict = {"approx": None}
				result = parse_ast(value)
				out_dict["approx"] = result
				return out_dict
			elif key == "range":
				out_dict = {"range": None}
				result = parse_ast(value)
				out_dict["range"] = result
				return out_dict
			elif key == "andor":
				out_dict = {"range/and": None}
				result = parse_ast(value)
				out_dict["range/and"] = result
				return out_dict
		else:
			if isinstance(ast, dict):
				result = []
				for key, value in ast.items():
					if key in ['annee', 'mois']:
						date = parse_ast({key: value})
						result.append(date)
					elif key == "range":
						date_range = parse_ast(value)
						result.append({"range": date_range})
					elif key == "andor":
						date_range = parse_ast(value)
						result.append({"range/and": date_range})
					elif key == "and":
						date_range = parse_ast(value)
						result.append({"and": date_range})
				return result
			elif isinstance(ast, list):
				items = []
				for item in ast:
					result = parse_ast(item)
					items.append(result)
				return items



def process_date(date, debug=False):
	"""
	Cette fonction initialise le lexeur et le parseur prévu pour le traitement des dates
	:param text: le texte
	:return: l'objet date parsé
	"""
	ast = build_grammar(debug=debug, text=date)
	parsed = parse_ast(ast)
	result = second_pass(parsed)
	return result

def build_grammar(debug: bool=False, text: str="19 février 1914, 21 mars 1915") -> list:
	"""
	This function builds an Abstract Syntax Tree from a query
	:param debug: outputs parsing information
	:param query: the query to build the AST from
	:return: the ast
	"""
	MyLexer = lexer.Lexer()
	MyLexer.tokenize(text, debug=debug)
	MyParser = parser.Parser(MyLexer, debug=debug)
	if debug is True:
		print(json.dumps(MyParser.ast,
						 sort_keys=False, indent=4))
	return MyParser.ast



def correct_date(date: str) -> str:
	"""
	Cette fonction vise à corriger une date extraite et où seraient présentes des erreurs
	d'HTR:
	:param date: la chaîne de caractères à corriger
	:return: la date corrigée
	"""
	number_dict = {"un": 1,
				   "deux": 2,
				   "trois": 3,
				   "quatre": 4,
				   "cinq": 5,
				   "six": 6,
				   "sept": 7,
				   "huit": 8,
				   "neuf": 9,
				   "dix": 10,
				   "onze": 11,
				   "douze": 12,
				   "treize": 13,
				   "quatorze": 14,
				   "quinze": 15,
				   "seize": 16,
				   "vingt": 20,
				   "trente": 30,
				   "mil": 1000,
				   "cent": 100}

	month_dict = {"janvier": "01",
				  "février": "02",
				  "mars": "03",
				  "avril": "04",
				  "mai": "05",
				  "juin": "06",
				  "juillet": "07",
				  "août": "08",
				  "septembre": "09",
				  "octobre": "10",
				  "novembre": "11",
				  "décembre": "12",
				  }
	date = utils.nfc_normalize(date)
	date = date.lower().strip()
	clean_regexp = re.compile(r"(\d+)\^?er?")
	date = re.sub(clean_regexp, r'\g<1>', date)
	le_regexp = re.compile(r"^[Ll]e ")
	date = re.sub(le_regexp, r"", date)
	date = utils.strip_punctuation(date)

	# On corrige les erreurs fŕequentes
	common_mistakes = {"aout": "août",
					   "dix": "dix ",
					   "vingt": "vingt ",
					   "trente": "trente "}
	for orig, reg in common_mistakes.items():
		date = date.replace(orig, reg)
	splits = re.compile(r"[\s+\-]")
	splitted = re.split(splits, date)
	result = []
	for token in splitted:
		if token in common_mistakes:
			result.append(common_mistakes[token])
		elif token in month_dict or token in number_dict or token in ['de', 'du', 'an', 'et', 'en']:
			result.append(token)
		else:
			matching, corrected = utils.check_word_in_list(list(month_dict.keys()) + list(number_dict.keys()),
													 token,
													 sensibility=0.7 if len(token) > 4 else 0.57)
			if matching:
				result.append(corrected)
			else:
				result.append(token)
	normalized = " ".join([item for item in result if item != ""])
	normalized = normalized.lower()
	return normalized


def test():
	dates_examples = [
		"17 septembre 1918",
		"17 mars 1918",
		"13 janvier et 18 mars 1918",
		"mars et avril 1917",
		"courant de 1915 et 1916",
		"courant juin 1917",
		"16 au 28 juillet 1918",
		"an 1917 et an 1918",
		"avril à juillet 1917",
		"janvier 1917",
		"année 1915",
		"19 février - 21 mars 1915",
		"17 août 17",
		"19 février 1914 - 21 mars 1915",
		"19 février 1914, 21 mars 1915",
		"en août 1917."
		"mai et juin 1917",
		"juin 1917 et 25 septembre 17",
		"du 27 au 28 septembre 1915",
		"5 janvier au 21 février 1917",
	"vingt neuf novembre l'an mil neuf cent dix sept",
	"vingt septembre an mil neuf cent dix sept",
	"sept avril de l'an mil neuf cent dix sept",
	"trente et un octobre de l'an mil neuf cent dixsept",
	"juin juillet 1917",
	"an mil neuf cent dix sept le dix huit août",
	"mai 1916 à mai 1917",
	"an mil neuf cent dix sept",
	"an mil neuf cent dix sept le dix huit août",
	"l'an mil neuf cent dix huit le huit avril",
	"l'an mil neuf cent dix huit le dix huit avril",
	"onze janvier mil neuf cent dix huit",
	"l'an mil neuf cent seize le six octobre",
	"l'an mil neuf cent dix huit le dix sept juin",
	"l'an mil neuf cent seize le deux mai"]
	# dates_examples = [dates_examples[-1]]
	for example in dates_examples:
		print("---")
		print(example)
		example = example.lower()
		corrected = correct_date(example)
		date = process_date(corrected, debug=False)
		print(f"Date: {date}")
		print(f"Corrected date: {corrected}")
		print(f"Processed date: {date}")

if __name__ == '__main__':
	test()
	date = process_date()
	print(date)
