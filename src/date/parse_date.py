import re
import sys
import parser
import lexer
import json
import datetime

datetime.datetime(2014, 10, 21, 0, 0)


class Date:
	def __init__(self):
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
		self.converted_date = self.converted_date.replace("//", "/")
		begin_regexp = re.compile(r"^/")
		end_regexp = re.compile(r"/$")
		self.converted_date = re.sub(begin_regexp, "", str(self.converted_date))
		self.converted_date = re.sub(end_regexp, "", str(self.converted_date))

	def convert_to_dd_mm_yyyy(self):
		self.month = self.month_dict[self.month] if self.month != "" else ""
		self.converted_date = f"{self.day}/{self.month}/{self.year}"
		self.clean_date()
		return self.converted_date

def second_pass(parsed_tree):
	print(parsed_tree)
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

def parse_ast(ast) -> None:
	# On fait 2 cas principaux: le premier cas où la complexité est absente
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
						range = parse_ast(value)
						result.append({"range": range})
					elif key == "andor":
						range = parse_ast(value)
						result.append({"range/and": range})
					elif key == "and":
						range = parse_ast(value)
						result.append({"and": range})
				return result
			elif isinstance(ast, list):
				items = []
				for item in ast:
					result = parse_ast(item)
					items.append(result)
				return items


def build_grammar(debug: bool, query: str) -> list:
	"""
	This function builds an Abstract Syntax Tree from a query
	:param debug: outputs parsing information
	:param query: the query to build the AST from
	:return: the ast
	"""
	print("---")
	print(query)
	MyLexer = lexer.Lexer()
	MyLexer.tokenize(query, debug=debug)
	MyParser = parser.Parser(MyLexer, debug=debug)
	if debug:
		print(json.dumps(MyParser.ast,
						 sort_keys=False, indent=4))
	return MyParser.ast


def test():
	dates_examples = [
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
		"19 février 1914, 21 mars 1915"]
	# dates_examples = ["16 au 28 juillet 1918"]
	for example in dates_examples:
		ast = build_grammar(debug=False, query=example)
		parsed = parse_ast(ast)
		result = second_pass(parsed)
		print(f"Résultat: {result}")
		if result is None:
			print("ERROR")

if __name__ == '__main__':
	test()
