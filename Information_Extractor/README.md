# Extraction des données


## `extract.py`

Fichier qui permet l'extraction des données à partir de:
- un fichier image
- un fichier de sortie de YOLO
- un fichier de sortie de kraken

Fonctionne pour l'instant sur la première page du procès

## `predict_with_party.py`

Script qui met en place l'OCR via Party. 



## Améliorations

### Identification des lignes importantes

à l'aide de classification par random forest à nouveau ? Pour éviter d'avoir 
à toucher le modèle de segmentation.

### Multiples passages d'OCR

Pour les lignes importantes:
- numéro d'ordre
- numéro de jugement
- date du crime
- nom des magistrats
- nom du soldat

On peut essayer de comparer les résultats de l'OCR par kraken et par Party. 
Party est particulièrement lourd et on ne peut pas l'utiliser pour tout traiter, 
mais on peut comparer les résultats et s'en servir pour faciliter la correction.


