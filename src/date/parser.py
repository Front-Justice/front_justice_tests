import ply.yacc as yacc
import src.date.lexer as lexer

chiffre_dict = {"un": 1,
                "premier": 1,
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
                "cent": 100
                }


class Parser(lexer.Lexer):
    """
    The parser. Builds the Ast with the tokens produced by the lexer.
    """
    tokens = lexer.Lexer.tokens


    precedence = (
        ('left', 'AND'),  # Priorité pour "et"
    )

    def p_mois_annee_range(self, p):
        '''
        mois_annee_range : mois_annee RANGE mois_annee
        '''
        p[0] = {"range": [p[1], p[3]]}


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
                            | COURANT AN annee
                            | COURANT mois_annee
        '''
        if len(p) == 3:
            p[0] = {"courant": p[2]}
        else:
            p[0] = {"courant": p[3]}


    def p_date_jour(self, p):
        '''
        jour_toutes_lettres : unite
            | dizaine unite
            | dizaine
        '''
        p[0] = {"jour": sum([item for item in p[1:]])}



    def p_mois(self, p):
        '''
        mois : MOIS
        '''
        p[0] = {"mois": p[1]}

    def p_jour_toutes_lettres_mois(self, p):
        '''
        jour_toutes_lettres_mois : jour_toutes_lettres mois
        '''
        p[0] = {**p[1], **p[2]}


    def p_unite(self, p):
        '''
        unite : CHIFFRE_UNITE
        '''
        p[0] = chiffre_dict[p[1]]

    def p_centaine(self, p):
        '''
        centaine : CHIFFRE_CENTAINE
                | unite CHIFFRE_CENTAINE
        '''
        if len(p) == 3:
            p[0] = p[1] * chiffre_dict[p[2]]
        else:
            p[0] = chiffre_dict[p[1]]

    def p_dizaine(self, p):
        '''
        dizaine : CHIFFRE_DIZAINE unite
                | CHIFFRE_DIZAINE
                | CHIFFRE_DIZAINE AND unite
        '''
        if len(p) == 3:
            p[0] = chiffre_dict[p[1]] + p[2]
        elif len(p) == 4:
            p[0] = chiffre_dict[p[1]] + p[3]
        else:
            p[0] = chiffre_dict[p[1]]


    def p_millier(self, p):
        '''
        millier : CHIFFRE_MILLIER
        '''
        p[0] = chiffre_dict[p[1]]

    def p_an_date_toute_lettre(self, p):
        '''
        token_an_toutes_lettres : DE AN
                                | AN
        '''
        p[0] = None

    def p_date_an(self, p):
        '''
        an_toutes_lettres : token_an_toutes_lettres millier centaine dizaine unite
                            | token_an_toutes_lettres millier centaine dizaine
        '''

        p[0] = {"annee": sum([item for item in p[2:]])}


    def p_succession_mois_annees(self, p):
        '''
        groupe_annee : succession_mois ANNEE
        '''
        p[0] = {**p[1], "annee": int(p[2])}

    def p_annee(self, p):
        '''
        annee : AN ANNEE
                    | ANNEE
                    | AN an_toutes_lettres
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


    def p_date_complete_toutes_lettres_inversee(self, p):
        '''
        date_toutes_lettres_inversee : an_toutes_lettres jour_toutes_lettres_mois
        '''
        p[0] = {**p[1], **p[2]}

    def p_date_complete_toutes_lettres(self, p):
        '''
        date_toutes_lettres : jour_toutes_lettres_mois an_toutes_lettres
        '''
        p[0] = {**p[1], **p[2]}


    def p_date_complete(self, p):
        '''
        date_complete : plage ANNEE
                     | jour_mois ANNEE
                     | jour_toutes_lettres_mois ANNEE
                     | groupe_annee
                     | courant_de_groupe
                     | JOUR mois ANNEE
                     | mois_annee
                     | mois_andor_mois_annee
                     | annee
                     | an_toutes_lettres
                     | range_mois ANNEE
                     | range_jour ANNEE
                     | annee_et_annee
                     | andor_jour_mois_jour_mois
                     | andor_date_complete
                     | jour_mois_annee
                     | deux_dates_completes
                     | range_jour_mois ANNEE
                     | date_toutes_lettres
                     | date_toutes_lettres_inversee
                     | mois_annee_range
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

    def p_andor_jour_mois_jour_mois(self, p):
        '''
        andor_jour_mois : jour_mois ANDOR jour_mois
                                    | jour_mois jour_mois
        '''
        if len(p) == 4:
            p[0] = {"andor": [p[1], p[3]],}
        else:
            p[0] = {"andor": [p[1], p[2]],}


    def p_andor_jour_mois(self, p):
        '''
        andor_jour_mois_jour_mois : andor_jour_mois
                        | andor_jour_mois ANNEE
        '''
        if len(p) == 2:
            p[0] = p[1]
        elif len(p) == 3:
            p[0] = {**p[1], "annee": int(p[2])}

    def p_succession_mois(self, p):
        '''
        succession_mois : mois AND mois
        '''
        p[0] = {
            "and": [p[1], p[3]]
        }

    def p_mois_andor_mois(self, p):
        '''
        mois_andor_mois : mois ANDOR mois
                        | mois mois
        '''
        if len(p) == 3:
            p[0] = {
                "andor": [p[1], p[2]]
            }
        else:
            p[0] = {
                "andor": [p[1], p[3]]
            }

    def p_mois_andor_mois_annee(self, p):
        '''
        mois_andor_mois_annee : mois_andor_mois ANNEE
        '''
        p[0] = {**p[1], "annee": p[2]}

    def p_error(self, p):
        pass
        # if p:
        #     print(f"Erreur de syntaxe à '{p.value}'")
        # else:
        #     print("Erreur de syntaxe : fin de fichier inattendue")

    def __init__(self, lexer, debug):
        self.lexer = lexer
        # self.parser = yacc.yacc(module=self, start='mois_et_annee', debug=debug)
        self.parser = yacc.yacc(module=self, start='start', debug=debug)
        self.ast = self.parser.parse(lexer=self.lexer, tracking=True, debug=debug)
