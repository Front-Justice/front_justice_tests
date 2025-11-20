import ply.lex as lex
import copy

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
		'RANGE',
        "AN",
        "SPACE"
    )

    # Pour l'instant les années à 2 chiffres n'apparaissent qu'en fin de chaîne
    t_ANNEE = r"\d{4}|\d{2}$"
    t_JOUR = r"\d{1,2}"
    t_MOIS = r"janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre"
    t_AND = r"et|,"
    t_ANDOR = r"-"
    t_RANGE = r"à|au"
    t_AN = r"ann[ée]e|an"


    def t_COURANT(self, t):
        r'courant(\sde)?'
        return t

    def t_SPACE(self, t):
        r'\s+'  # Capturer les espaces, mais ne pas les retourner
        pass


    # Error handling rule
    def t_error(self, t):
        print("Illegal character '%s'" % t.value[0])
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

