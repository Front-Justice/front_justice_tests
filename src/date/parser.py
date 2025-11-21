import ply.yacc as yacc
import src.date.lexer as lexer

class Parser(lexer.Lexer):
    """
    The parser. Builds the Ast with the tokens produced by the lexer.
    """
    tokens = lexer.Lexer.tokens

    precedence = (
        ('left', 'AND'),  # Priorité pour "et"
    )


    def p_mois_annee(self, p):
        '''
        mois_annee : mois ANNEE
        '''
        p[0] = {**p[1], "annee": p[2]}

    def p_plage_dates(self, p):
        '''
        plage : jour_mois AND jour_mois
        '''
        p[0] = {"and": [p[1], p[3]]}

    def p_annee_et_annee(self, p):
        '''
        annee_et_annee : annee AND annee
        '''
        p[0] = {"and": [p[1], p[3]]}



    def p_courant_de_groupe(self, p):
        '''
        courant_de_groupe : COURANT annee_et_annee
                            | COURANT ANNEE
                            | COURANT mois_annee
        '''
        p[0] = {"courant": p[2]}

    def p_succession_mois_annees(self, p):
        '''
        groupe_annee : succession_mois ANNEE
        '''
        p[0] = {**p[1], "annee": int(p[2])}

    def p_annee(self, p):
        '''
        annee : AN ANNEE
                    | ANNEE
        '''
        if len(p) == 3:
            p[0] = {"annee": int(p[2])}
        else:
            p[0] = {"annee": int(p[1])}

    def p_date_complete_date_complete(self, p):
        '''
        deux_dates_completes : jour_mois_annee AND jour_mois_annee
        '''
        p[0] = {"and": [p[1], p[3]]}




    def p_mois(self, p):
        '''
        mois : MOIS
        '''
        p[0] = {"mois": p[1]}

    def p_date_complete(self, p):
        '''
        date_complete : plage ANNEE
                     | jour_mois ANNEE
                     | groupe_annee
                     | courant_de_groupe
                     | JOUR mois ANNEE
                     | mois_annee
                     | annee
                     | range_mois ANNEE
                     | range_jour ANNEE
                     | annee_et_annee
                     | andor_jour_mois
                     | andor_date_complete
                     | jour_mois_annee
                     | deux_dates_completes
                     | range_jour_mois ANNEE
        '''
        if len(p) == 3:
            p[0] = {**p[1], "annee": int(p[2])}
        elif len(p) == 4:
            if int(p[3]) < 100:
                annee = int(p[3]) + 1900
            else:
                annee = int(p[3])
            p[0] = {"jour": int(p[1]), **p[2], "annee": annee}
        else:
            p[0] = p[1]

    def p_start(self, p):
        '''
        start : date_complete
        '''
        p[0] = p[1]

    def p_range_mois(self, p):
        '''
        range_mois : mois RANGE mois
        '''
        p[0] = {"range": [p[1],
                          p[3]]}


    def p_range_jour_mois(self, p):
        '''
        range_jour_mois : jour_mois RANGE jour_mois
        '''
        p[0] = {"range": [p[1],
                          p[3]]}

    def p_range_jour(self, p):
        '''
        range_jour : JOUR RANGE JOUR
                    | JOUR RANGE JOUR mois
        '''
        if len(p) == 4:
            p[0] = {"range": [{"jour": p[1]},
                              {"jour": p[3]}]}
        else:
            p[0] = {"range": [{"jour": p[1]},
                              {"jour": p[3]}],
                    **p[4]}

    def p_jour_mois(self, p):
        '''
        jour_mois : JOUR mois
        '''
        p[0] = {"jour": p[1], **p[2]}

    def p_andor_date_complete(self, p):
        '''
        andor_date_complete : jour_mois_annee ANDOR jour_mois_annee
        '''
        p[0] = {"andor": [p[1], p[3]]}


    def p_jour_mois_annee(self, p):
        'jour_mois_annee : JOUR mois ANNEE'
        p[0] = {"jour": p[1], **p[2], "annee": p[3]}

    def p_andor_jour_mois(self, p):
        '''
        andor_jour_mois : jour_mois ANDOR jour_mois
                        | jour_mois ANDOR jour_mois ANNEE
        '''
        if len(p) == 4:
            p[0] = {"andor": [p[1], p[3]]}
        elif len(p) == 5:
            p[0] = {"andor": [p[1], p[3]], "annee": int(p[4])}

    def p_succession_mois(self, p):
        '''
        succession_mois : mois AND mois
        '''
        p[0] = {
            "and": [p[1], p[3]]
        }

    def p_error(self, p):
        if p:
            print(f"Erreur de syntaxe à '{p.value}'")
        else:
            print("Erreur de syntaxe : fin de fichier inattendue")

    def __init__(self, lexer, debug):
        self.lexer = lexer
        # self.parser = yacc.yacc(module=self, start='mois_et_annee', debug=debug)
        self.parser = yacc.yacc(module=self, start='start', debug=debug)
        self.ast = self.parser.parse(lexer=self.lexer, tracking=True, debug=debug)
