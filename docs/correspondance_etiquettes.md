# Correspondance des étiquettes : TotalSegmentator ↔ MM-WHS

Ce document fixe, **avant l'étape 6**, la règle qui relie les étiquettes produites par TotalSegmentator (étape 4) à nos étiquettes de vérité terrain (MM-WHS renumérotées à l'étape 3). Il a été écrit sans ouvrir le lot de test.

## Les deux numérotations

- **TotalSegmentator, tâche `heartchambers_highres`.** Le fichier multi-étiquettes, écrit avec l'option `ml=True` par `scripts/04_totalsegmentator.py`, contient les étiquettes 1 à 7. Elles sont définies dans `totalsegmentator/map_to_binary.py` (dictionnaire `class_map["heartchambers_highres"]`). La version vérifiée est TotalSegmentator 2.18.0, celle installée dans le notebook.
- **MM-WHS renumérotées.** Les codes d'origine (205, 420…) sont convertis en 1 à 7 par la table `CORRESPONDANCE` de `scripts/03_preparer.py`, et les noms sont repris dans `dataset.json`.

| Étiquette | TotalSegmentator | MM-WHS d'origine | Notre nom (`dataset.json`) | Même définition ? |
|---|---|---|---|---|
| 0 | fond | 0 | `background` | oui |
| 1 | `heart_myocardium` | 205 | `myocarde_VG` | **à vérifier** : TotalSegmentator inclut peut-être aussi la paroi du ventricule droit |
| 2 | `heart_atrium_left` | 420 | `oreillette_gauche` | **à vérifier** : l'auricule gauche et le départ des veines pulmonaires sont peut-être traités autrement |
| 3 | `heart_ventricle_left` | 500 | `cavite_VG` | probablement proche |
| 4 | `heart_atrium_right` | 550 | `oreillette_droite` | **à vérifier** : la limite avec les veines caves peut différer |
| 5 | `heart_ventricle_right` | 600 | `ventricule_droit` | **à vérifier** : cavité seule, ou avec la paroi |
| 6 | `aorta` | 820 | `aorte_ascendante` | **non, sur l'étendue** : MM-WHS s'arrête au bord supérieur des oreillettes (« from the aortic valve to the superior level of the atria ») ; TotalSegmentator inclut probablement d'autres segments de l'aorte, par exemple l'aorte descendante |
| 7 | `pulmonary_artery` | 850 | `artere_pulmonaire` | **à vérifier, sur l'étendue** : MM-WHS ne garde que le tronc, de la valve à la bifurcation ; TotalSegmentator inclut peut-être le début des branches |

**Règle :** l'étiquette *k* de TotalSegmentator correspond à notre étiquette *k*, pour *k* = 1 à 7. Aucune renumérotation n'est nécessaire. Les numéros coïncident, mais les définitions peuvent différer : la dernière colonne le signale.

Sources des définitions MM-WHS : https://zmiclab.github.io/zxh/0/mmwhs/data.html. TotalSegmentator ne publie pas de définition précise de ses classes ; les écarts marqués « à vérifier » sont des hypothèses.

## Comment lever les doutes sans ouvrir le lot de test

La section 11 du notebook fait tourner TotalSegmentator sur **deux cas d'entraînement**, un de chaque famille d'examens (1003 à 0,45 mm, 1002 à 0,625 mm). Elle superpose ensuite ses contours à la vérité terrain, structure par structure. Ces cas ne font pas partie du lot de test, et les masques produits restent hors de `nnunet_raw` : ils ne serviront jamais à l'entraînement (règle 5).

Les règles de comparaison de l'étape 6 seront fixées d'après ce que montre cette section, puis tracées dans le journal **avant** tout calcul de score. Par exemple :
- rapporter à part les structures définies différemment ;
- limiter la comparaison à une zone fixée d'avance ;
- comparer seulement les 4 cavités et le myocarde.
