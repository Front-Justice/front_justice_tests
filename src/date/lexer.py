import ply.lex as lex
import copy
import src.date.utils as utils


class Lexer(object):
    """
    Le lexer pour tokéniser la date.

    """
    tokens = (
        'COURANT',
		'ANNEE',
		'MOIS',
		'JOUR',
		'AND',
        'ANDOR',
        "A",
		'RANGE',
        "AN",
        "SPACE",
        "DE",
       #  "LE",
        "CHIFFRE_UNITE",
        "CHIFFRE_DIZAINE",
        "CHIFFRE_CENTAINE",
        "CHIFFRE_MILLIER",
    )

    tokens = [utils.nfc_normalize(token) for token in tokens]

    # Pour l'instant les années à 2 chiffres n'apparaissent qu'en fin de chaîne
    t_ANNEE = r"\d{4}|\d{2}$"
    t_JOUR = r"\d{1,2}"
    t_AND = r"et|,"
    t_ANDOR = r"-"
   # t_LE = r"[lL]e"
    t_RANGE = utils.nfc_normalize(r"à\s|au\s|a\s")


    def t_AN(self, t):
        r"l\s?'?\s?ann[eé]e\s|ann[eé]e\s|an\s|l\s?'?\s?an\s?"
        t.value = utils.nfc_normalize(t.value)
        return t


    def t_MOIS(self, t):
        r'''janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre'''
        t.value = utils.nfc_normalize(t.value)
        return t

    def t_CHIFFRE_UNITE(self, t):
        r'''premier|un|deux|trois|quatre|cinq|six|sept|huit|neuf'''
        return t


    def t_DE(self, t):
        r"de|du|d'"
        return t

    def t_CHIFFRE_DIZAINE(self, t):
        r'''dix|onze|douze|treize|quatorze|quinze|seize|vingt|trente'''
        return t


    def t_CHIFFRE_MILLIER(self, t):
        r'''mil'''
        return t

    def t_CHIFFRE_CENTAINE(self, t):
        r'''cent'''
        return t

    def t_COURANT(self, t):
        r'courant(\sde)?|en\s'
        return t

    def t_SPACE(self, t):
        r'\s+'  # Capturer les espaces, mais ne pas les retourner
        pass

    def t_LE(self, t):
        r'[Ll]e'
        pass


    # Error handling rule
    def t_error(self, t):
        # print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)

    def tokenize(self, text:str, debug:bool=False):
        self.lexer = lex.lex(module=self)
        self.lexer.input(text)

        if debug:
            print(f"Date normalisée: {text}")
            debug_lexer = copy.deepcopy(self.lexer)
            while True:
                tok = debug_lexer.token()
                if not tok:
                    break  # No more input
                print(tok)

    def token(self):
        return self.lexer.token()

