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
		'JOUR_OU_ANNEE',
		'AND',
        'ANDOR',
		'RANGE',
        "AN",
        "SPACE"
    )

    t_ANNEE = r"\d{4}"
    t_JOUR_OU_ANNEE = r"\d{2}"
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

    def tokenize(self, query:str, debug:bool=False):
        self.lexer = lex.lex(module=self)
        self.lexer.input(query)

        if debug:
            debug_lexer = copy.deepcopy(self.lexer)
            while True:
                tok = debug_lexer.token()
                if not tok:
                    break  # No more input
                print(tok)

    def token(self):
        return self.lexer.token()

