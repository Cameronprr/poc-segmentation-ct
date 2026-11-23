# Étape 6 : évaluation sur le lot de test (lecture unique)

Cas : 1004, 1010, 1011, 1013, 1014. 1014 : seul cas à 0,45 mm (D20).

Définition de la HD95 : HD95 symétrique de MONAI 1.6.1 : 95e percentile des distances bord de P → bord de V, idem V → P, et le plus grand des deux ; en mm avec l'espacement de l'image. Non directement comparable aux HD95 calculées par d'autres logiciels (percentile de l'union des distances). « (plafond) » : structure oubliée (D1), HD95 = diagonale de l'image.

Arrondis à l'affichage seulement (§ 8.1) ; une différence de Dice de 0,005 n'est pas une différence. Pas de test statistique ni d'intervalle de confiance (§ 8.3).

## Verdict (pipeline A seulement, § 8.4)

**ACCEPTABLE**

| structure | médiane du Dice (non arrondie) | n | seuil | C1 |
|---|---|---|---|---|
| myocarde VG | 0.9215323328971863 | 5 | 0.85 | rempli |
| OG | 0.9282763600349426 | 5 | 0.85 | rempli |
| cavité VG | 0.9642567038536072 | 5 | 0.85 | rempli |
| OD | 0.8813151717185974 | 5 | 0.85 | rempli |
| VD | 0.9024218916893005 | 5 | 0.85 | rempli |
| aorte | 0.9819454550743103 | 5 | 0.85 | rempli |
| AP | 0.8961717486381531 | 5 | 0.8 | rempli |

C2 (aucun échec silencieux) : rempli.

Limites (§ 8.4, § 12) : seuils fixés après la validation croisée (pas totalement indépendants de nos résultats) ; avec 5 patients, verdict indicatif ; Zhuang et al. 2019 non directement comparable.

## (a) Pipeline principal : A post-traité, 7 structures

### Modèle A

| cas | structure | Dice | HD95 (mm) | V prédit (mL) | V vérité (mL) | écart (mL) | écart (%) | NSD 2 mm | remarque |
|---|---|---|---|---|---|---|---|---|---|
| 1004 | myocarde VG | 0.922 | 1.3 | 75.9 | 70.3 | +5.6 | +8 % | 0.989 |  |
| 1004 | OG | 0.857 | 8.4 | 35.4 | 28.4 | +7.1 | +25 % | 0.740 |  |
| 1004 | cavité VG | 0.979 | 1.1 | 84.9 | 87.7 | -2.8 | -3 % | 0.993 |  |
| 1004 | OD | 0.906 | 3.3 | 33.5 | 32.9 | +0.6 | +2 % | 0.849 |  |
| 1004 | VD | 0.924 | 6.0 | 92.3 | 105.4 | -13.1 | -12 % | 0.911 |  |
| 1004 | aorte | 0.983 | 0.7 | 38.5 | 37.9 | +0.6 | +2 % | 0.997 |  |
| 1004 | AP | 0.838 | 17.6 | 29.2 | 29.1 | +0.1 | +0 % | 0.828 |  |
| 1004 | **Dice moyen (7 structures)** | **0.915** (n = 7) | | | | | | | |
| 1010 | myocarde VG | 0.911 | 2.0 | 109.6 | 98.5 | +11.1 | +11 % | 0.950 |  |
| 1010 | OG | 0.962 | 2.3 | 74.4 | 76.9 | -2.5 | -3 % | 0.948 |  |
| 1010 | cavité VG | 0.964 | 1.6 | 97.6 | 100.1 | -2.6 | -3 % | 0.974 |  |
| 1010 | OD | 0.881 | 8.4 | 64.0 | 54.9 | +9.1 | +17 % | 0.801 |  |
| 1010 | VD | 0.915 | 2.7 | 119.3 | 137.9 | -18.6 | -14 % | 0.800 |  |
| 1010 | aorte | 0.984 | 0.6 | 64.3 | 62.6 | +1.7 | +3 % | 0.994 |  |
| 1010 | AP | 0.947 | 2.6 | 55.9 | 52.7 | +3.2 | +6 % | 0.941 |  |
| 1010 | **Dice moyen (7 structures)** | **0.938** (n = 7) | | | | | | | |
| 1011 | myocarde VG | 0.920 | 1.5 | 102.4 | 103.4 | -1.0 | -1 % | 0.970 |  |
| 1011 | OG | 0.945 | 3.5 | 108.0 | 97.7 | +10.3 | +11 % | 0.891 |  |
| 1011 | cavité VG | 0.973 | 1.3 | 113.8 | 112.2 | +1.6 | +1 % | 0.988 |  |
| 1011 | OD | 0.894 | 5.0 | 76.0 | 65.8 | +10.2 | +15 % | 0.744 |  |
| 1011 | VD | 0.902 | 11.1 | 130.2 | 134.3 | -4.1 | -3 % | 0.848 |  |
| 1011 | aorte | 0.966 | 1.3 | 63.6 | 65.8 | -2.2 | -3 % | 0.975 |  |
| 1011 | AP | 0.883 | 8.9 | 89.8 | 73.8 | +16.0 | +22 % | 0.875 |  |
| 1011 | **Dice moyen (7 structures)** | **0.926** (n = 7) | | | | | | | |
| 1013 | myocarde VG | 0.950 | 1.9 | 172.9 | 182.8 | -9.9 | -5 % | 0.958 |  |
| 1013 | OG | 0.900 | 11.0 | 98.5 | 89.8 | +8.6 | +10 % | 0.811 |  |
| 1013 | cavité VG | 0.927 | 2.9 | 63.3 | 59.4 | +3.9 | +7 % | 0.902 |  |
| 1013 | OD | 0.880 | 9.7 | 85.2 | 73.6 | +11.5 | +16 % | 0.782 |  |
| 1013 | VD | 0.896 | 5.0 | 100.6 | 109.8 | -9.2 | -8 % | 0.791 |  |
| 1013 | aorte | 0.980 | 1.1 | 55.9 | 56.7 | -0.8 | -1 % | 0.989 |  |
| 1013 | AP | 0.901 | 14.5 | 83.7 | 87.1 | -3.3 | -4 % | 0.866 |  |
| 1013 | **Dice moyen (7 structures)** | **0.919** (n = 7) | | | | | | | |
| 1014 | myocarde VG | 0.935 | 1.9 | 112.7 | 112.5 | +0.2 | +0 % | 0.966 | seul cas à 0,45 mm (D20) |
| 1014 | OG | 0.928 | 6.3 | 63.7 | 64.8 | -1.1 | -2 % | 0.842 | seul cas à 0,45 mm (D20) |
| 1014 | cavité VG | 0.962 | 2.0 | 82.4 | 85.5 | -3.1 | -4 % | 0.960 | seul cas à 0,45 mm (D20) |
| 1014 | OD | 0.864 | 11.4 | 51.6 | 48.4 | +3.2 | +7 % | 0.727 | seul cas à 0,45 mm (D20) |
| 1014 | VD | 0.893 | 8.2 | 99.0 | 93.6 | +5.4 | +6 % | 0.785 | seul cas à 0,45 mm (D20) |
| 1014 | aorte | 0.982 | 0.9 | 55.0 | 53.3 | +1.7 | +3 % | 0.993 | seul cas à 0,45 mm (D20) |
| 1014 | AP | 0.896 | 15.6 | 52.9 | 59.1 | -6.3 | -11 % | 0.860 | seul cas à 0,45 mm (D20) |
| 1014 | **Dice moyen (7 structures)** | **0.923** (n = 7) | | | | | | | |

Agrégats (tous les cas, alertes comprises) :

| structure | Dice médiane / moyenne / min (n) | HD95 médiane / moyenne / max (n) | écart mL : méd. / moy. signées, méd. abs., pire (n) | écart % : méd. / moy. signées, méd. abs., pire (n) | NSD médiane / moyenne / min (n) |
|---|---|---|---|---|---|
| myocarde VG | 0.922 / 0.927 / 0.911 (5) | 1.9 / 1.7 / 2.0 (5) | +0.2 / +1.2, 5.6, +11.1 (5) | +0 % / +3 %, 5 %, +11 % (5) | 0.966 / 0.967 / 0.950 (5) |
| OG | 0.928 / 0.918 / 0.857 (5) | 6.3 / 6.3 / 11.0 (5) | +7.1 / +4.5, 7.1, +10.3 (5) | +10 % / +8 %, 10 %, +25 % (5) | 0.842 / 0.846 / 0.740 (5) |
| cavité VG | 0.964 / 0.961 / 0.927 (5) | 1.6 / 1.8 / 2.9 (5) | -2.6 / -0.6, 2.8, +3.9 (5) | -3 % / -0 %, 3 %, +7 % (5) | 0.974 / 0.963 / 0.902 (5) |
| OD | 0.881 / 0.885 / 0.864 (5) | 8.4 / 7.6 / 11.4 (5) | +9.1 / +6.9, 9.1, +11.5 (5) | +15 % / +11 %, 15 %, +17 % (5) | 0.782 / 0.781 / 0.727 (5) |
| VD | 0.902 / 0.906 / 0.893 (5) | 6.0 / 6.6 / 11.1 (5) | -9.2 / -7.9, 9.2, -18.6 (5) | -8 % / -6 %, 8 %, -14 % (5) | 0.800 / 0.827 / 0.785 (5) |
| aorte | 0.982 / 0.979 / 0.966 (5) | 0.9 / 0.9 / 1.3 (5) | +0.6 / +0.2, 1.7, -2.2 (5) | +2 % / +1 %, 3 %, -3 % (5) | 0.993 / 0.990 / 0.975 (5) |
| AP | 0.896 / 0.893 / 0.838 (5) | 14.5 / 11.8 / 17.6 (5) | +0.1 / +2.0, 3.3, +16.0 (5) | +0 % / +3 %, 6 %, +22 % (5) | 0.866 / 0.874 / 0.828 (5) |

Complément, **pas un résultat principal** (§ 8.2) : agrégats SANS les cas en alerte (1004, 1010) :

| structure | Dice médiane / moyenne / min (n) | HD95 médiane / moyenne / max (n) | écart mL : méd. / moy. signées, méd. abs., pire (n) | écart % : méd. / moy. signées, méd. abs., pire (n) | NSD médiane / moyenne / min (n) |
|---|---|---|---|---|---|
| myocarde VG | 0.935 / 0.935 / 0.920 (3) | 1.9 / 1.8 / 1.9 (3) | -1.0 / -3.6, 1.0, -9.9 (3) | -1 % / -2 %, 1 %, -5 % (3) | 0.966 / 0.965 / 0.958 (3) |
| OG | 0.928 / 0.924 / 0.900 (3) | 6.3 / 6.9 / 11.0 (3) | +8.6 / +5.9, 8.6, +10.3 (3) | +10 % / +6 %, 10 %, +11 % (3) | 0.842 / 0.848 / 0.811 (3) |
| cavité VG | 0.962 / 0.954 / 0.927 (3) | 2.0 / 2.1 / 2.9 (3) | +1.6 / +0.8, 3.1, +3.9 (3) | +1 % / +1 %, 4 %, +7 % (3) | 0.960 / 0.950 / 0.902 (3) |
| OD | 0.880 / 0.879 / 0.864 (3) | 9.7 / 8.7 / 11.4 (3) | +10.2 / +8.3, 10.2, +11.5 (3) | +15 % / +13 %, 15 %, +16 % (3) | 0.744 / 0.751 / 0.727 (3) |
| VD | 0.896 / 0.897 / 0.893 (3) | 8.2 / 8.1 / 11.1 (3) | -4.1 / -2.6, 5.4, -9.2 (3) | -3 % / -2 %, 6 %, -8 % (3) | 0.791 / 0.808 / 0.785 (3) |
| aorte | 0.980 / 0.976 / 0.966 (3) | 1.1 / 1.1 / 1.3 (3) | -0.8 / -0.4, 1.7, -2.2 (3) | -1 % / -0 %, 3 %, -3 % (3) | 0.989 / 0.986 / 0.975 (3) |
| AP | 0.896 / 0.893 / 0.883 (3) | 14.5 / 13.0 / 15.6 (3) | -3.3 / +2.1, 6.3, +16.0 (3) | -4 % / +2 %, 11 %, +22 % (3) | 0.866 / 0.867 / 0.860 (3) |

Contrôle de plausibilité (alertes officielles) : 2 cas en alerte sur 5 (1004, 1010).

**Alertes « erreur probable »**, classées (vraies ou fausses) : 2

| cas | règle | structure ou rapport | mesure | seuil | classement | justification |
|---|---|---|---|---|---|---|
| 1004 | R5 | AP_moins_VD | {"ecart_hu": 284, "hu_AP": 536, "hu_VD": 252} | {"ecart_hu_max": 176} | vraie | Dice AP 0.8376510143280029 ; Dice VD 0.9243962168693542 (vraie si l'un < 0,85) |
| 1010 | R5 | AP_moins_VD | {"ecart_hu": 182, "hu_AP": 379, "hu_VD": 197} | {"ecart_hu_max": 176} | fausse | Dice AP 0.9469611644744873 ; Dice VD 0.915107786655426 (vraie si l'un < 0,85) |

**Alertes « hors domaine » (limite du modèle, pas une erreur de segmentation)**, classées (vraies ou fausses) : 0


**Alertes non classables** (rubrique à part, D43) : 0

Définition : Une alerte est non classable quand la grandeur que la règle de classement du § 7.3 doit comparer n'est pas définie pour ce cas : (i) la vérité de la structure en cause est vide (cas dégénérés D2 ou D3, § 6), donc Dice et écart de volume sont non définis ; (ii) la mesure de densité exigée par R6 ou R7 ne peut pas être faite sur la vérité (structure absente ou trop petite pour le module figé, ou vérité que le module figé déclare non mesurable, par ex. pour sa géométrie). Une alerte non classable n'est ni vraie ni fausse : elle est rapportée à part, avec sa raison, et n'entre ni dans l'estimation des fausses alertes ni dans le compte des vraies alertes. Règle à plusieurs conditions reliées par « ou » (R3, R4, R5, R8) : l'alerte est vraie dès qu'UNE condition définie est vraie ; non classable seulement si aucune condition définie n'est vraie et qu'au moins une n'est pas définie ; fausse si toutes sont définies et fausses.

**Estimation officielle des fausses alertes (D36)** : 1 sur 5 cas avec au moins une alerte « erreur probable » classée fausse (1010). Hors domaine classées fausses, à part : 0 sur 5 cas. Vraies alertes : 1 « erreur probable », 0 hors domaine. Alertes non classables : 0, rapportées à part ; elles n'entrent ni dans l'estimation ni dans le compte des vraies alertes (D43).

Échecs silencieux (Dice < 0,50 sans aucune alerte sur le cas) : aucun.

## (b) Méthodes à armes égales : N0 et S0 bruts, 7 structures

### Modèle N0

| cas | structure | Dice | HD95 (mm) | V prédit (mL) | V vérité (mL) | écart (mL) | écart (%) | NSD 2 mm | remarque |
|---|---|---|---|---|---|---|---|---|---|
| 1004 | myocarde VG | 0.922 | 1.3 | 76.1 | 70.3 | +5.8 | +8 % | 0.988 |  |
| 1004 | OG | 0.845 | 9.5 | 36.0 | 28.4 | +7.7 | +27 % | 0.723 |  |
| 1004 | cavité VG | 0.981 | 1.0 | 85.9 | 87.7 | -1.8 | -2 % | 0.995 |  |
| 1004 | OD | 0.908 | 3.0 | 33.2 | 32.9 | +0.3 | +1 % | 0.870 |  |
| 1004 | VD | 0.922 | 6.5 | 91.9 | 105.4 | -13.5 | -13 % | 0.908 |  |
| 1004 | aorte | 0.982 | 0.7 | 38.5 | 37.9 | +0.6 | +2 % | 0.994 |  |
| 1004 | AP | 0.462 | 31.7 | 14.9 | 29.1 | -14.2 | -49 % | 0.469 |  |
| 1004 | **Dice moyen (7 structures)** | **0.860** (n = 7) | | | | | | | |
| 1010 | myocarde VG | 0.909 | 2.1 | 109.7 | 98.5 | +11.2 | +11 % | 0.946 |  |
| 1010 | OG | 0.962 | 2.2 | 74.8 | 76.9 | -2.1 | -3 % | 0.952 |  |
| 1010 | cavité VG | 0.964 | 1.6 | 97.6 | 100.1 | -2.5 | -3 % | 0.974 |  |
| 1010 | OD | 0.879 | 10.5 | 63.5 | 54.9 | +8.6 | +16 % | 0.798 |  |
| 1010 | VD | 0.911 | 3.0 | 118.5 | 137.9 | -19.3 | -14 % | 0.764 |  |
| 1010 | aorte | 0.983 | 0.6 | 64.4 | 62.6 | +1.8 | +3 % | 0.994 |  |
| 1010 | AP | 0.941 | 2.5 | 56.2 | 52.7 | +3.5 | +7 % | 0.932 |  |
| 1010 | **Dice moyen (7 structures)** | **0.936** (n = 7) | | | | | | | |
| 1011 | myocarde VG | 0.907 | 1.9 | 101.9 | 103.4 | -1.6 | -2 % | 0.956 |  |
| 1011 | OG | 0.942 | 4.1 | 108.9 | 97.7 | +11.1 | +11 % | 0.876 |  |
| 1011 | cavité VG | 0.970 | 1.3 | 114.4 | 112.2 | +2.2 | +2 % | 0.987 |  |
| 1011 | OD | 0.887 | 5.5 | 77.8 | 65.8 | +12.0 | +18 % | 0.729 |  |
| 1011 | VD | 0.894 | 11.9 | 127.2 | 134.3 | -7.1 | -5 % | 0.823 |  |
| 1011 | aorte | 0.965 | 1.2 | 63.8 | 65.8 | -1.9 | -3 % | 0.973 |  |
| 1011 | AP | 0.878 | 9.5 | 90.2 | 73.8 | +16.4 | +22 % | 0.876 |  |
| 1011 | **Dice moyen (7 structures)** | **0.921** (n = 7) | | | | | | | |
| 1013 | myocarde VG | 0.950 | 1.9 | 175.4 | 182.8 | -7.5 | -4 % | 0.961 |  |
| 1013 | OG | 0.892 | 12.3 | 99.7 | 89.8 | +9.9 | +11 % | 0.808 |  |
| 1013 | cavité VG | 0.927 | 2.5 | 61.0 | 59.4 | +1.6 | +3 % | 0.913 |  |
| 1013 | OD | 0.874 | 10.4 | 87.2 | 73.6 | +13.6 | +18 % | 0.769 |  |
| 1013 | VD | 0.894 | 5.1 | 100.6 | 109.8 | -9.2 | -8 % | 0.796 |  |
| 1013 | aorte | 0.978 | 1.1 | 55.7 | 56.7 | -1.0 | -2 % | 0.988 |  |
| 1013 | AP | 0.877 | 18.2 | 82.8 | 87.1 | -4.3 | -5 % | 0.821 |  |
| 1013 | **Dice moyen (7 structures)** | **0.913** (n = 7) | | | | | | | |
| 1014 | myocarde VG | 0.931 | 1.9 | 111.3 | 112.5 | -1.2 | -1 % | 0.964 | seul cas à 0,45 mm (D20) |
| 1014 | OG | 0.932 | 5.7 | 63.4 | 64.8 | -1.4 | -2 % | 0.842 | seul cas à 0,45 mm (D20) |
| 1014 | cavité VG | 0.962 | 2.0 | 83.6 | 85.5 | -1.9 | -2 % | 0.958 | seul cas à 0,45 mm (D20) |
| 1014 | OD | 0.858 | 10.4 | 50.5 | 48.4 | +2.2 | +5 % | 0.702 | seul cas à 0,45 mm (D20) |
| 1014 | VD | 0.895 | 8.5 | 102.3 | 93.6 | +8.7 | +9 % | 0.793 | seul cas à 0,45 mm (D20) |
| 1014 | aorte | 0.981 | 1.0 | 54.9 | 53.3 | +1.6 | +3 % | 0.987 | seul cas à 0,45 mm (D20) |
| 1014 | AP | 0.895 | 15.6 | 53.3 | 59.1 | -5.9 | -10 % | 0.860 | seul cas à 0,45 mm (D20) |
| 1014 | **Dice moyen (7 structures)** | **0.922** (n = 7) | | | | | | | |

Agrégats (tous les cas, alertes comprises) :

| structure | Dice médiane / moyenne / min (n) | HD95 médiane / moyenne / max (n) | écart mL : méd. / moy. signées, méd. abs., pire (n) | écart % : méd. / moy. signées, méd. abs., pire (n) | NSD médiane / moyenne / min (n) |
|---|---|---|---|---|---|
| myocarde VG | 0.922 / 0.924 / 0.907 (5) | 1.9 / 1.8 / 2.1 (5) | -1.2 / +1.4, 5.8, +11.2 (5) | -1 % / +3 %, 4 %, +11 % (5) | 0.961 / 0.963 / 0.946 (5) |
| OG | 0.932 / 0.914 / 0.845 (5) | 5.7 / 6.7 / 12.3 (5) | +7.7 / +5.0, 7.7, +11.1 (5) | +11 % / +9 %, 11 %, +27 % (5) | 0.842 / 0.840 / 0.723 (5) |
| cavité VG | 0.964 / 0.961 / 0.927 (5) | 1.6 / 1.7 / 2.5 (5) | -1.8 / -0.5, 1.9, -2.5 (5) | -2 % / -0 %, 2 %, +3 % (5) | 0.974 / 0.966 / 0.913 (5) |
| OD | 0.879 / 0.881 / 0.858 (5) | 10.4 / 7.9 / 10.5 (5) | +8.6 / +7.3, 8.6, +13.6 (5) | +16 % / +12 %, 16 %, +18 % (5) | 0.769 / 0.774 / 0.702 (5) |
| VD | 0.895 / 0.903 / 0.894 (5) | 6.5 / 7.0 / 11.9 (5) | -9.2 / -8.1, 9.2, -19.3 (5) | -8 % / -6 %, 9 %, -14 % (5) | 0.796 / 0.817 / 0.764 (5) |
| aorte | 0.981 / 0.978 / 0.965 (5) | 1.0 / 0.9 / 1.2 (5) | +0.6 / +0.2, 1.6, -1.9 (5) | +2 % / +1 %, 3 %, +3 % (5) | 0.988 / 0.987 / 0.973 (5) |
| AP | 0.878 / 0.811 / 0.462 (5) | 15.6 / 15.5 / 31.7 (5) | -4.3 / -0.9, 5.9, +16.4 (5) | -5 % / -7 %, 10 %, -49 % (5) | 0.860 / 0.792 / 0.469 (5) |

Contrôle de plausibilité (alertes informatives) : 3 cas en alerte sur 5 (1004, 1010, 1013).

**Alertes « erreur probable »**, classées (vraies ou fausses) : 5

| cas | règle | structure ou rapport | mesure | seuil | classement | justification |
|---|---|---|---|---|---|---|
| 1004 | R2 | 7 AP | {"volume_ml": 14.853} | {"bas": 18.292, "haut": 154.085} | vraie | Dice 0.46199002861976624 ; écart de volume -48.961959826897825 % (vraie si Dice < 0,85 ou |écart| > 30 %) |
| 1004 | R3 | AP_sur_aorte | {"rapport": 0.3859} | {"bas": 0.4705, "haut": 2.7905} | vraie | Dice AP 0.46199002861976624 ; Dice aorte 0.9821924567222595 (vraie si l'un < 0,85) |
| 1004 | R8 | 7 AP | {"morceaux_gros": 2, "morceaux_total": 5, "part_plus_gros": 0.8466} | {"morceaux_gros_max": 1} | vraie | 0 morceau(x) égaré(s) de plus de 0,5 mL ; Dice 0.46199002861976624 |
| 1010 | R5 | AP_moins_VD | {"ecart_hu": 182, "hu_AP": 379, "hu_VD": 197} | {"ecart_hu_max": 176} | fausse | Dice AP 0.9413285255432129 ; Dice VD 0.9109665155410767 (vraie si l'un < 0,85) |
| 1013 | R8 | 7 AP | {"morceaux_gros": 2, "morceaux_total": 6, "part_plus_gros": 0.9831} | {"morceaux_gros_max": 1} | vraie | 1 morceau(x) égaré(s) de plus de 0,5 mL ; Dice 0.8766909837722778 |

**Alertes « hors domaine » (limite du modèle, pas une erreur de segmentation)**, classées (vraies ou fausses) : 0


**Alertes non classables** (rubrique à part, D43) : 0

Définition : Une alerte est non classable quand la grandeur que la règle de classement du § 7.3 doit comparer n'est pas définie pour ce cas : (i) la vérité de la structure en cause est vide (cas dégénérés D2 ou D3, § 6), donc Dice et écart de volume sont non définis ; (ii) la mesure de densité exigée par R6 ou R7 ne peut pas être faite sur la vérité (structure absente ou trop petite pour le module figé, ou vérité que le module figé déclare non mesurable, par ex. pour sa géométrie). Une alerte non classable n'est ni vraie ni fausse : elle est rapportée à part, avec sa raison, et n'entre ni dans l'estimation des fausses alertes ni dans le compte des vraies alertes. Règle à plusieurs conditions reliées par « ou » (R3, R4, R5, R8) : l'alerte est vraie dès qu'UNE condition définie est vraie ; non classable seulement si aucune condition définie n'est vraie et qu'au moins une n'est pas définie ; fausse si toutes sont définies et fausses.

Échecs silencieux (Dice < 0,50 sans aucune alerte sur le cas) : aucun.

### Modèle S0

| cas | structure | Dice | HD95 (mm) | V prédit (mL) | V vérité (mL) | écart (mL) | écart (%) | NSD 2 mm | remarque |
|---|---|---|---|---|---|---|---|---|---|
| 1004 | myocarde VG | 0.922 | 1.3 | 70.5 | 70.3 | +0.1 | +0 % | 0.985 |  |
| 1004 | OG | 0.832 | 10.0 | 38.1 | 28.4 | +9.7 | +34 % | 0.677 |  |
| 1004 | cavité VG | 0.976 | 1.3 | 85.1 | 87.7 | -2.6 | -3 % | 0.983 |  |
| 1004 | OD | 0.864 | 4.7 | 33.3 | 32.9 | +0.5 | +1 % | 0.706 |  |
| 1004 | VD | 0.929 | 3.3 | 93.3 | 105.4 | -12.1 | -12 % | 0.883 |  |
| 1004 | aorte | 0.976 | 4.1 | 39.3 | 37.9 | +1.4 | +4 % | 0.950 |  |
| 1004 | AP | 0.750 | 12.3 | 22.6 | 29.1 | -6.5 | -22 % | 0.695 |  |
| 1004 | **Dice moyen (7 structures)** | **0.893** (n = 7) | | | | | | | |
| 1010 | myocarde VG | 0.900 | 2.3 | 111.1 | 98.5 | +12.6 | +13 % | 0.933 |  |
| 1010 | OG | 0.956 | 2.8 | 73.4 | 76.9 | -3.5 | -5 % | 0.916 |  |
| 1010 | cavité VG | 0.956 | 2.3 | 98.6 | 100.1 | -1.6 | -2 % | 0.942 |  |
| 1010 | OD | 0.874 | 8.2 | 66.8 | 54.9 | +11.9 | +22 % | 0.763 |  |
| 1010 | VD | 0.910 | 3.2 | 117.1 | 137.9 | -20.8 | -15 % | 0.758 |  |
| 1010 | aorte | 0.982 | 0.6 | 64.1 | 62.6 | +1.5 | +2 % | 0.997 |  |
| 1010 | AP | 0.917 | 54.6 | 60.4 | 52.7 | +7.6 | +15 % | 0.860 |  |
| 1010 | **Dice moyen (7 structures)** | **0.928** (n = 7) | | | | | | | |
| 1011 | myocarde VG | 0.918 | 1.5 | 99.9 | 103.4 | -3.5 | -3 % | 0.977 |  |
| 1011 | OG | 0.956 | 2.8 | 104.4 | 97.7 | +6.7 | +7 % | 0.927 |  |
| 1011 | cavité VG | 0.967 | 1.7 | 112.6 | 112.2 | +0.4 | +0 % | 0.967 |  |
| 1011 | OD | 0.891 | 5.3 | 79.3 | 65.8 | +13.5 | +21 % | 0.760 |  |
| 1011 | VD | 0.899 | 13.2 | 130.7 | 134.3 | -3.6 | -3 % | 0.838 |  |
| 1011 | aorte | 0.963 | 1.7 | 64.3 | 65.8 | -1.5 | -2 % | 0.960 |  |
| 1011 | AP | 0.869 | 15.2 | 88.6 | 73.8 | +14.8 | +20 % | 0.846 |  |
| 1011 | **Dice moyen (7 structures)** | **0.923** (n = 7) | | | | | | | |
| 1013 | myocarde VG | 0.923 | 3.3 | 165.0 | 182.8 | -17.8 | -10 % | 0.873 |  |
| 1013 | OG | 0.899 | 10.4 | 99.0 | 89.8 | +9.2 | +10 % | 0.770 |  |
| 1013 | cavité VG | 0.904 | 4.0 | 66.4 | 59.4 | +7.0 | +12 % | 0.811 |  |
| 1013 | OD | 0.883 | 9.6 | 87.5 | 73.6 | +13.9 | +19 % | 0.768 |  |
| 1013 | VD | 0.883 | 6.4 | 111.3 | 109.8 | +1.5 | +1 % | 0.707 |  |
| 1013 | aorte | 0.976 | 1.2 | 55.3 | 56.7 | -1.4 | -2 % | 0.981 |  |
| 1013 | AP | 0.895 | 22.1 | 84.1 | 87.1 | -3.0 | -3 % | 0.866 |  |
| 1013 | **Dice moyen (7 structures)** | **0.909** (n = 7) | | | | | | | |
| 1014 | myocarde VG | 0.833 | 5.0 | 88.5 | 112.5 | -24.0 | -21 % | 0.818 | seul cas à 0,45 mm (D20) |
| 1014 | OG | 0.910 | 8.4 | 59.8 | 64.8 | -5.0 | -8 % | 0.810 | seul cas à 0,45 mm (D20) |
| 1014 | cavité VG | 0.921 | 5.9 | 76.5 | 85.5 | -9.0 | -10 % | 0.793 | seul cas à 0,45 mm (D20) |
| 1014 | OD | 0.755 | 56.3 | 49.3 | 48.4 | +1.0 | +2 % | 0.476 | seul cas à 0,45 mm (D20) |
| 1014 | VD | 0.870 | 6.0 | 97.5 | 93.6 | +3.9 | +4 % | 0.702 | seul cas à 0,45 mm (D20) |
| 1014 | aorte | 0.973 | 3.2 | 54.8 | 53.3 | +1.5 | +3 % | 0.935 | seul cas à 0,45 mm (D20) |
| 1014 | AP | 0.796 | 62.4 | 41.8 | 59.1 | -17.4 | -29 % | 0.695 | seul cas à 0,45 mm (D20) |
| 1014 | **Dice moyen (7 structures)** | **0.865** (n = 7) | | | | | | | |

Agrégats (tous les cas, alertes comprises) :

| structure | Dice médiane / moyenne / min (n) | HD95 médiane / moyenne / max (n) | écart mL : méd. / moy. signées, méd. abs., pire (n) | écart % : méd. / moy. signées, méd. abs., pire (n) | NSD médiane / moyenne / min (n) |
|---|---|---|---|---|---|
| myocarde VG | 0.918 / 0.899 / 0.833 (5) | 2.3 / 2.7 / 5.0 (5) | -3.5 / -6.5, 12.6, -24.0 (5) | -3 % / -4 %, 10 %, -21 % (5) | 0.933 / 0.917 / 0.818 (5) |
| OG | 0.910 / 0.911 / 0.832 (5) | 8.4 / 6.9 / 10.4 (5) | +6.7 / +3.4, 6.7, +9.7 (5) | +7 % / +8 %, 8 %, +34 % (5) | 0.810 / 0.820 / 0.677 (5) |
| cavité VG | 0.956 / 0.945 / 0.904 (5) | 2.3 / 3.1 / 5.9 (5) | -1.6 / -1.2, 2.6, -9.0 (5) | -2 % / -1 %, 3 %, +12 % (5) | 0.942 / 0.899 / 0.793 (5) |
| OD | 0.874 / 0.854 / 0.755 (5) | 8.2 / 16.8 / 56.3 (5) | +11.9 / +8.1, 11.9, +13.9 (5) | +19 % / +13 %, 19 %, +22 % (5) | 0.760 / 0.695 / 0.476 (5) |
| VD | 0.899 / 0.898 / 0.870 (5) | 6.0 / 6.4 / 13.2 (5) | -3.6 / -6.2, 3.9, -20.8 (5) | -3 % / -5 %, 4 %, -15 % (5) | 0.758 / 0.777 / 0.702 (5) |
| aorte | 0.976 / 0.974 / 0.963 (5) | 1.7 / 2.2 / 4.1 (5) | +1.4 / +0.3, 1.5, -1.5 (5) | +2 % / +1 %, 2 %, +4 % (5) | 0.960 / 0.965 / 0.935 (5) |
| AP | 0.869 / 0.845 / 0.750 (5) | 22.1 / 33.3 / 62.4 (5) | -3.0 / -0.9, 7.6, -17.4 (5) | -3 % / -4 %, 20 %, -29 % (5) | 0.846 / 0.792 / 0.695 (5) |

Contrôle de plausibilité (alertes informatives) : 4 cas en alerte sur 5 (1004, 1010, 1011, 1014).

**Alertes « erreur probable »**, classées (vraies ou fausses) : 7

| cas | règle | structure ou rapport | mesure | seuil | classement | justification |
|---|---|---|---|---|---|---|
| 1004 | R5 | AP_moins_VD | {"ecart_hu": 267, "hu_AP": 520, "hu_VD": 253} | {"ecart_hu_max": 176} | vraie | Dice AP 0.749895453453064 ; Dice VD 0.928868293762207 (vraie si l'un < 0,85) |
| 1004 | R8 | 7 AP | {"morceaux_gros": 2, "morceaux_total": 24, "part_plus_gros": 0.9213} | {"morceaux_gros_max": 1} | vraie | 0 morceau(x) égaré(s) de plus de 0,5 mL ; Dice 0.749895453453064 |
| 1010 | R5 | AP_moins_VD | {"ecart_hu": 179, "hu_AP": 376, "hu_VD": 197} | {"ecart_hu_max": 176} | fausse | Dice AP 0.9166954159736633 ; Dice VD 0.909853994846344 (vraie si l'un < 0,85) |
| 1010 | R8 | 7 AP | {"morceaux_gros": 2, "morceaux_total": 32, "part_plus_gros": 0.9751} | {"morceaux_gros_max": 1} | vraie | 1 morceau(x) égaré(s) de plus de 0,5 mL ; Dice 0.9166954159736633 |
| 1011 | R8 | 7 AP | {"morceaux_gros": 2, "morceaux_total": 56, "part_plus_gros": 0.9856} | {"morceaux_gros_max": 1} | vraie | 1 morceau(x) égaré(s) de plus de 0,5 mL ; Dice 0.8686425089836121 |
| 1014 | R8 | 4 OD | {"morceaux_gros": 3, "morceaux_total": 78, "part_plus_gros": 0.8413} | {"morceaux_gros_max": 1} | vraie | 2 morceau(x) égaré(s) de plus de 0,5 mL ; Dice 0.7550816535949707 |
| 1014 | R8 | 7 AP | {"morceaux_gros": 2, "morceaux_total": 46, "part_plus_gros": 0.9724} | {"morceaux_gros_max": 1} | vraie | 1 morceau(x) égaré(s) de plus de 0,5 mL ; Dice 0.7958928346633911 |

**Alertes « hors domaine » (limite du modèle, pas une erreur de segmentation)**, classées (vraies ou fausses) : 0


**Alertes non classables** (rubrique à part, D43) : 0

Définition : Une alerte est non classable quand la grandeur que la règle de classement du § 7.3 doit comparer n'est pas définie pour ce cas : (i) la vérité de la structure en cause est vide (cas dégénérés D2 ou D3, § 6), donc Dice et écart de volume sont non définis ; (ii) la mesure de densité exigée par R6 ou R7 ne peut pas être faite sur la vérité (structure absente ou trop petite pour le module figé, ou vérité que le module figé déclare non mesurable, par ex. pour sa géométrie). Une alerte non classable n'est ni vraie ni fausse : elle est rapportée à part, avec sa raison, et n'entre ni dans l'estimation des fausses alertes ni dans le compte des vraies alertes. Règle à plusieurs conditions reliées par « ou » (R3, R4, R5, R8) : l'alerte est vraie dès qu'UNE condition définie est vraie ; non classable seulement si aucune condition définie n'est vraie et qu'au moins une n'est pas définie ; fausse si toutes sont définies et fausses.

Échecs silencieux (Dice < 0,50 sans aucune alerte sur le cas) : aucun.

### Règle descriptive P6 (pas une preuve)

Écart des médianes > 0,02 (inégalité stricte, valeurs flottantes non arrondies, sans tolérance) et mieux sur au moins 4 cas ; règle appliquée seulement avec 5 cas utilisables pour chaque modèle (sinon « pas de différence démontrée », D43).

| structure | médiane N0 | médiane S0 | cas où N0 fait mieux | cas où S0 fait mieux | conclusion |
|---|---|---|---|---|---|
| myocarde VG | 0.922 | 0.918 | 3 sur 5 | 2 sur 5 | pas de différence démontrée |
| OG | 0.932 | 0.910 | 3 sur 5 | 2 sur 5 | pas de différence démontrée |
| cavité VG | 0.964 | 0.956 | 5 sur 5 | 0 sur 5 | pas de différence démontrée |
| OD | 0.879 | 0.874 | 3 sur 5 | 2 sur 5 | pas de différence démontrée |
| VD | 0.895 | 0.899 | 3 sur 5 | 2 sur 5 | pas de différence démontrée |
| aorte | 0.981 | 0.976 | 5 sur 5 | 0 sur 5 | pas de différence démontrée |
| AP | 0.878 | 0.869 | 3 sur 5 | 2 sur 5 | pas de différence démontrée |

## (c) Référence externe : TotalSegmentator (T), structures 1 à 5, en regard de A

> Prudence (D23) : sur deux cas d'entraînement, les volumes de TotalSegmentator s'écartaient de la vérité (1002 : OD +46 %, VD −18 %), avec des frontières moins nettes. Un mauvais score sur l'OD ou le VD peut venir de la définition de la structure, pas de la qualité de l'outil.

### Modèle T

| cas | structure | Dice | HD95 (mm) | V prédit (mL) | V vérité (mL) | écart (mL) | écart (%) | NSD 2 mm | remarque |
|---|---|---|---|---|---|---|---|---|---|
| 1004 | myocarde VG | 0.916 | 1.4 | 74.4 | 70.3 | +4.1 | +6 % | 0.986 |  |
| 1004 | OG | 0.857 | 6.8 | 34.8 | 28.4 | +6.5 | +23 % | 0.727 |  |
| 1004 | cavité VG | 0.970 | 1.7 | 83.5 | 87.7 | -4.2 | -5 % | 0.971 |  |
| 1004 | OD | 0.902 | 3.4 | 34.9 | 32.9 | +2.0 | +6 % | 0.829 | prudence (D23) |
| 1004 | VD | 0.928 | 5.0 | 93.6 | 105.4 | -11.8 | -11 % | 0.903 | prudence (D23) |
| 1004 | **Dice moyen (5 structures)** | **0.915** (n = 5) | | | | | | | |
| 1010 | myocarde VG | 0.919 | 1.7 | 107.6 | 98.5 | +9.1 | +9 % | 0.976 |  |
| 1010 | OG | 0.958 | 2.4 | 72.7 | 76.9 | -4.1 | -5 % | 0.946 |  |
| 1010 | cavité VG | 0.967 | 1.4 | 97.8 | 100.1 | -2.3 | -2 % | 0.978 |  |
| 1010 | OD | 0.884 | 6.9 | 63.5 | 54.9 | +8.6 | +16 % | 0.795 | prudence (D23) |
| 1010 | VD | 0.936 | 2.1 | 124.7 | 137.9 | -13.2 | -10 % | 0.939 | prudence (D23) |
| 1010 | **Dice moyen (5 structures)** | **0.933** (n = 5) | | | | | | | |
| 1011 | myocarde VG | 0.931 | 1.4 | 104.8 | 103.4 | +1.4 | +1 % | 0.980 |  |
| 1011 | OG | 0.964 | 2.0 | 97.2 | 97.7 | -0.5 | -1 % | 0.954 |  |
| 1011 | cavité VG | 0.976 | 1.3 | 110.4 | 112.2 | -1.9 | -2 % | 0.984 |  |
| 1011 | OD | 0.904 | 4.2 | 76.1 | 65.8 | +10.3 | +16 % | 0.772 | prudence (D23) |
| 1011 | VD | 0.914 | 10.3 | 130.8 | 134.3 | -3.5 | -3 % | 0.873 | prudence (D23) |
| 1011 | **Dice moyen (5 structures)** | **0.938** (n = 5) | | | | | | | |
| 1013 | myocarde VG | 0.895 | 4.5 | 156.0 | 182.8 | -26.8 | -15 % | 0.801 |  |
| 1013 | OG | 0.902 | 8.8 | 93.0 | 89.8 | +3.2 | +4 % | 0.820 |  |
| 1013 | cavité VG | 0.863 | 7.1 | 72.3 | 59.4 | +12.9 | +22 % | 0.715 |  |
| 1013 | OD | 0.878 | 9.3 | 83.1 | 73.6 | +9.5 | +13 % | 0.754 | prudence (D23) |
| 1013 | VD | 0.887 | 4.4 | 111.4 | 109.8 | +1.6 | +1 % | 0.707 | prudence (D23) |
| 1013 | **Dice moyen (5 structures)** | **0.885** (n = 5) | | | | | | | |
| 1014 | myocarde VG | 0.909 | 2.9 | 102.3 | 112.5 | -10.2 | -9 % | 0.907 | seul cas à 0,45 mm (D20) |
| 1014 | OG | 0.943 | 3.8 | 62.5 | 64.8 | -2.3 | -3 % | 0.898 | seul cas à 0,45 mm (D20) |
| 1014 | cavité VG | 0.959 | 2.1 | 80.7 | 85.5 | -4.7 | -6 % | 0.958 | seul cas à 0,45 mm (D20) |
| 1014 | OD | 0.863 | 11.4 | 49.8 | 48.4 | +1.5 | +3 % | 0.738 | prudence (D23); seul cas à 0,45 mm (D20) |
| 1014 | VD | 0.877 | 6.0 | 100.8 | 93.6 | +7.2 | +8 % | 0.689 | prudence (D23); seul cas à 0,45 mm (D20) |
| 1014 | **Dice moyen (5 structures)** | **0.910** (n = 5) | | | | | | | |

Agrégats (tous les cas, alertes comprises) :

| structure | Dice médiane / moyenne / min (n) | HD95 médiane / moyenne / max (n) | écart mL : méd. / moy. signées, méd. abs., pire (n) | écart % : méd. / moy. signées, méd. abs., pire (n) | NSD médiane / moyenne / min (n) |
|---|---|---|---|---|---|
| myocarde VG | 0.916 / 0.914 / 0.895 (5) | 1.7 / 2.4 / 4.5 (5) | +1.4 / -4.5, 9.1, -26.8 (5) | +1 % / -1 %, 9 %, -15 % (5) | 0.976 / 0.930 / 0.801 (5) |
| OG | 0.943 / 0.925 / 0.857 (5) | 3.8 / 4.8 / 8.8 (5) | -0.5 / +0.5, 3.2, +6.5 (5) | -1 % / +3 %, 4 %, +23 % (5) | 0.898 / 0.869 / 0.727 (5) |
| cavité VG | 0.967 / 0.947 / 0.863 (5) | 1.7 / 2.7 / 7.1 (5) | -2.3 / -0.0, 4.2, +12.9 (5) | -2 % / +1 %, 5 %, +22 % (5) | 0.971 / 0.921 / 0.715 (5) |
| OD (prudence, D23) | 0.884 / 0.886 / 0.863 (5) | 6.9 / 7.0 / 11.4 (5) | +8.6 / +6.4, 8.6, +10.3 (5) | +13 % / +11 %, 13 %, +16 % (5) | 0.772 / 0.778 / 0.738 (5) |
| VD (prudence, D23) | 0.914 / 0.908 / 0.877 (5) | 5.0 / 5.6 / 10.3 (5) | -3.5 / -3.9, 7.2, -13.2 (5) | -3 % / -3 %, 8 %, -11 % (5) | 0.873 / 0.822 / 0.689 (5) |

Contrôle de plausibilité (alertes à part, informatives) : 0 cas en alerte sur 5 (aucun).

Aucune alerte.
Échecs silencieux (Dice < 0,50 sans aucune alerte sur le cas) : aucun.

### Modèle A (rappel des valeurs de (a) sur les structures 1 à 5)

| cas | structure | Dice | HD95 (mm) | V prédit (mL) | V vérité (mL) | écart (mL) | écart (%) | NSD 2 mm | remarque |
|---|---|---|---|---|---|---|---|---|---|
| 1004 | myocarde VG | 0.922 | 1.3 | 75.9 | 70.3 | +5.6 | +8 % | 0.989 |  |
| 1004 | OG | 0.857 | 8.4 | 35.4 | 28.4 | +7.1 | +25 % | 0.740 |  |
| 1004 | cavité VG | 0.979 | 1.1 | 84.9 | 87.7 | -2.8 | -3 % | 0.993 |  |
| 1004 | OD | 0.906 | 3.3 | 33.5 | 32.9 | +0.6 | +2 % | 0.849 |  |
| 1004 | VD | 0.924 | 6.0 | 92.3 | 105.4 | -13.1 | -12 % | 0.911 |  |
| 1004 | **Dice moyen (5 structures)** | **0.917** (n = 5) | | | | | | | |
| 1010 | myocarde VG | 0.911 | 2.0 | 109.6 | 98.5 | +11.1 | +11 % | 0.950 |  |
| 1010 | OG | 0.962 | 2.3 | 74.4 | 76.9 | -2.5 | -3 % | 0.948 |  |
| 1010 | cavité VG | 0.964 | 1.6 | 97.6 | 100.1 | -2.6 | -3 % | 0.974 |  |
| 1010 | OD | 0.881 | 8.4 | 64.0 | 54.9 | +9.1 | +17 % | 0.801 |  |
| 1010 | VD | 0.915 | 2.7 | 119.3 | 137.9 | -18.6 | -14 % | 0.800 |  |
| 1010 | **Dice moyen (5 structures)** | **0.927** (n = 5) | | | | | | | |
| 1011 | myocarde VG | 0.920 | 1.5 | 102.4 | 103.4 | -1.0 | -1 % | 0.970 |  |
| 1011 | OG | 0.945 | 3.5 | 108.0 | 97.7 | +10.3 | +11 % | 0.891 |  |
| 1011 | cavité VG | 0.973 | 1.3 | 113.8 | 112.2 | +1.6 | +1 % | 0.988 |  |
| 1011 | OD | 0.894 | 5.0 | 76.0 | 65.8 | +10.2 | +15 % | 0.744 |  |
| 1011 | VD | 0.902 | 11.1 | 130.2 | 134.3 | -4.1 | -3 % | 0.848 |  |
| 1011 | **Dice moyen (5 structures)** | **0.927** (n = 5) | | | | | | | |
| 1013 | myocarde VG | 0.950 | 1.9 | 172.9 | 182.8 | -9.9 | -5 % | 0.958 |  |
| 1013 | OG | 0.900 | 11.0 | 98.5 | 89.8 | +8.6 | +10 % | 0.811 |  |
| 1013 | cavité VG | 0.927 | 2.9 | 63.3 | 59.4 | +3.9 | +7 % | 0.902 |  |
| 1013 | OD | 0.880 | 9.7 | 85.2 | 73.6 | +11.5 | +16 % | 0.782 |  |
| 1013 | VD | 0.896 | 5.0 | 100.6 | 109.8 | -9.2 | -8 % | 0.791 |  |
| 1013 | **Dice moyen (5 structures)** | **0.910** (n = 5) | | | | | | | |
| 1014 | myocarde VG | 0.935 | 1.9 | 112.7 | 112.5 | +0.2 | +0 % | 0.966 | seul cas à 0,45 mm (D20) |
| 1014 | OG | 0.928 | 6.3 | 63.7 | 64.8 | -1.1 | -2 % | 0.842 | seul cas à 0,45 mm (D20) |
| 1014 | cavité VG | 0.962 | 2.0 | 82.4 | 85.5 | -3.1 | -4 % | 0.960 | seul cas à 0,45 mm (D20) |
| 1014 | OD | 0.864 | 11.4 | 51.6 | 48.4 | +3.2 | +7 % | 0.727 | seul cas à 0,45 mm (D20) |
| 1014 | VD | 0.893 | 8.2 | 99.0 | 93.6 | +5.4 | +6 % | 0.785 | seul cas à 0,45 mm (D20) |
| 1014 | **Dice moyen (5 structures)** | **0.916** (n = 5) | | | | | | | |

Agrégats (tous les cas, alertes comprises) :

| structure | Dice médiane / moyenne / min (n) | HD95 médiane / moyenne / max (n) | écart mL : méd. / moy. signées, méd. abs., pire (n) | écart % : méd. / moy. signées, méd. abs., pire (n) | NSD médiane / moyenne / min (n) |
|---|---|---|---|---|---|
| myocarde VG | 0.922 / 0.927 / 0.911 (5) | 1.9 / 1.7 / 2.0 (5) | +0.2 / +1.2, 5.6, +11.1 (5) | +0 % / +3 %, 5 %, +11 % (5) | 0.966 / 0.967 / 0.950 (5) |
| OG | 0.928 / 0.918 / 0.857 (5) | 6.3 / 6.3 / 11.0 (5) | +7.1 / +4.5, 7.1, +10.3 (5) | +10 % / +8 %, 10 %, +25 % (5) | 0.842 / 0.846 / 0.740 (5) |
| cavité VG | 0.964 / 0.961 / 0.927 (5) | 1.6 / 1.8 / 2.9 (5) | -2.6 / -0.6, 2.8, +3.9 (5) | -3 % / -0 %, 3 %, +7 % (5) | 0.974 / 0.963 / 0.902 (5) |
| OD | 0.881 / 0.885 / 0.864 (5) | 8.4 / 7.6 / 11.4 (5) | +9.1 / +6.9, 9.1, +11.5 (5) | +15 % / +11 %, 15 %, +17 % (5) | 0.782 / 0.781 / 0.727 (5) |
| VD | 0.902 / 0.906 / 0.893 (5) | 6.0 / 6.6 / 11.1 (5) | -9.2 / -7.9, 9.2, -18.6 (5) | -8 % / -6 %, 8 %, -14 % (5) | 0.800 / 0.827 / 0.785 (5) |

