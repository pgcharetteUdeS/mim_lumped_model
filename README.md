Script qui utilise le modèle RLC groupé de l'article "Ultra‐Narrowband Metamaterial
    Absorbers for High Spectral Resolution Infrared Spectroscopy" [Kang, 2019] pour
    modéliser la réponse d'un filtre à base d'un réseau de nano-structures MIM
    en forme de croix.

Auteur: Paul Charette

Notes:
1) Les propriétés des matériaux pour le métal et l'isolant sont lues à partir
        de fichiers Excel lors de la création de l'objet de classe Materials dans
        la fonction main(), voir les exemples "Ciesielski-Au.xlsx" et
        "Kischkat-SiO2.xlsx" pour le format des fichiers. 
2) La plage des longueurs d'onde prises en compte dans les calculs est
        la portion commune des plages de longueurs d'onde des données optiques
        pour le metal et l'isolant lues dans les deux fichiers Excel. 
3) Les propriétés optiques des matériaux sont modélisées par des polynômes dont
        les ordres sont spécifiés lors de la création de l'objet de classe Materials,
        il faut valider visuellement les modèles avec le paramètre "debug=True". 
4) La géométrie de référence des structures MIM est spécifiée lors de la création
        de l'objet de classe Geometry dans la fonction main().