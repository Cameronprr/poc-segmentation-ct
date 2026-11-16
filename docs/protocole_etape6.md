# Protocole de test de l'étape 6 : évaluation sur le lot de test

> **Statut : PROPOSÉ, À VALIDER.** Ce document n'est pas encore figé.
> Une fois validé par l'utilisateur (questions P1 à P10 tranchées, section 13), il sera **figé** comme les seuils du contrôle de plausibilité (D36) : statut passé à « figé », date de gel, commit Git, empreinte SHA-256 du fichier inscrite au journal des décisions. Après le gel, plus aucune modification ; un écart devient une **déviation** consignée (section 11).
>
> Rédigé le 2026-11-16, **sans ouvrir le lot de test** : aucun masque de `labelsTs` n'a été lu. Les seuls chiffres cités viennent des cas d'entraînement (étape 5) et du journal.

Documents qui font foi, dans cet ordre : `docs/fiche_projet.pdf`, `docs/journal_decisions.md` (D16 à D41), `docs/specification_controle_plausibilite.md` (figée, D36), `docs/correspondance_etiquettes.md`, `config/seuils_plausibilite.json` (figé, D36). En cas de contradiction, le journal l'emporte sur ce protocole.

---

## 1. Objet

L'étape 6 mesure, **une seule fois**, la qualité des segmentations sur les **5 patients du lot de test**, jamais vus pendant l'entraînement ni pendant les réglages (CLAUDE.md, règle 4 ; D16, D18, D19). C'est l'équivalent d'une **PQ** (qualification de performance) : le lot de test est sous clé depuis le début.

Ce protocole fixe **avant l'ouverture** :
- ce qui est évalué (trois comparaisons, section 4) ;
- comment on mesure (métriques, règles de calcul, cas particuliers, sections 5 et 6) ;
- comment on présente les résultats (section 8) ;
- dans quel ordre on travaille, et ce qu'on fait si quelque chose se passe mal (sections 9 à 11).

**Pourquoi tout fixer avant ?** Si l'on choisit une métrique, un seuil ou un modèle après avoir vu les chiffres du test, on peut, même sans le vouloir, choisir ce qui « arrange » le résultat. Le verdict ne vaudrait plus rien (fiche projet, « Critères de réussite » ; D22, D23).

**Ouvrir le lot de test** veut dire, dans ce document : lire pour la première fois un masque de vérité terrain du test (`labelsTs`). La *vérité terrain* est la segmentation de référence, faite par un expert.

## 2. Ce qui est figé avant l'ouverture

| Élément | Où il est fixé | Décision |
|---|---|---|
| Découpage entraînement / test | `splits/mmwhs_split.json` (SHA-256 `25881b21…4a05c`, relevé par 05j) | D20 |
| Modèles évalués et leurs poids | section 3.3 ; empreintes relevées avant l'ouverture (section 10, phase B) | D31, D33, D40, D41 |
| Post-traitement du pipeline principal | plus grand morceau pour OG, OD, VD, aorte ; rien d'autre | D33 |
| Seuils du contrôle de plausibilité | `config/seuils_plausibilite.json`, SHA-256 `c1b247070765d193a46e90b538cc655787dbb65f8fdd13f113a39efd3ef90672` | D36 |
| Structures comparées pour TotalSegmentator | étiquettes 1 à 5 seulement | D23 |
| Métriques, règles de calcul, cas dégénérés, agrégats | sections 5 et 6 de ce protocole | ce protocole |
| Règle de classement des alertes (vraie / fausse) | section 7.3 | ce protocole (prolonge D35, D36) |
| Critères du verdict | section 8.4 (**selon la réponse à P1**) | fiche projet, D10 |
| Présentation des résultats | section 8 | D29 (Q5, Q8) |

**Aucune métrique, aucun agrégat, aucun seuil, aucun post-traitement et aucun choix de modèle ne peut être ajouté ou modifié après l'ouverture** (section 10).

## 3. Données et modèles évalués

### 3.1 Les 5 cas de test

| Cas | Épaisseur de coupe | Remarque |
|---|---|---|
| 1004 | 0,625 mm | |
| 1010 | 0,625 mm | |
| 1011 | 0,625 mm | |
| 1013 | 0,625 mm | |
| 1014 | **0,45 mm** | **seul cas de la famille à 0,45 mm** (D20) : il sert à repérer un échec grossier sur cette famille, pas à mesurer une performance propre à elle |

Le test contient **5 patients indépendants**, chacun avec **un seul examen** (D20). Dans ce protocole, **un cas = un patient** : il n'y a pas de paire à moyenner, contrairement à la validation croisée (D30, D31). Les moyennes « par cas » sont donc aussi des moyennes « par patient ».

Fichiers (Drive, `/content/drive/MyDrive/poc_ct/`, chemins passés en paramètres) : scanners dans `nnunet_raw/Dataset101_MMWHS/imagesTs`, vérité dans `nnunet_raw/Dataset101_MMWHS/labelsTs` (D18).

### 3.2 Structures et étiquettes

Numérotation de `dataset.json` (étape 3) ; TotalSegmentator utilise les mêmes numéros (`docs/correspondance_etiquettes.md`).

| Étiquette | Structure | (a) et (b) | (c) TotalSegmentator |
|---|---|---|---|
| 1 | myocarde VG (paroi musculaire du ventricule gauche) | oui | oui |
| 2 | OG (oreillette gauche) | oui | oui |
| 3 | cavité VG | oui | oui |
| 4 | OD (oreillette droite) | oui | oui, **avec prudence** (D23) |
| 5 | VD (ventricule droit) | oui | oui, **avec prudence** (D23) |
| 6 | aorte ascendante | oui | **non** : définition différente (D23) |
| 7 | AP (artère pulmonaire, tronc) | oui | **non** : définition différente (D23) |

### 3.3 Modèles et prédictions

| Repère | Modèle | Prédiction sur le test | Post-traitement |
|---|---|---|---|
| **A** | nnU-Net v2 2.8.1, `3d_fullres`, entraîneur `nnUNetTrainer_250epochs_reprise10`, plans `nnUNetPlans`, **ensemble des 5 plis** (`checkpoint_final`) | à produire à l'étape 6 (`nnUNetv2_predict -f 0 1 2 3 4`, réglages par défaut, voir P7) | **oui** : `postprocessing.pkl` de `crossval_results_folds_0_1_2_3_4` (05i), soit plus grand morceau pour 2, 4, 5, 6 (D33) |
| **N0** | nnU-Net, mêmes réglages, **pli 0 seul** (`checkpoint_final`) | à produire (`-f 0`) | **non** (D41) |
| **S0** | SegResNet (Auto3DSeg, MONAI 1.6.1, modèles `21ed8e5`), **pli 0**, 1 250 époques, `model_final.pt` (dossier `auto3dseg/entrainement_segresnet_1250/pli_0/`) | à produire, dans la géométrie de l'image d'origine, comme l'évaluation finale de 05m | **non** (D41) |
| **T** | TotalSegmentator 2.18.0, tâche `heartchambers_highres` | **déjà faite à l'étape 4** (D19), fichiers `mmwhs_XXXX.nii.gz` à côté de `tracabilite_etape4.json` | non |

Un *ensemble* combine les prédictions de plusieurs modèles (ici les 5 modèles de la validation croisée) en faisant la moyenne de leurs probabilités. Un *pli* est un des 5 modèles, entraîné sur 12 des 15 cas d'entraînement.

## 4. Les trois comparaisons

### (a) Pipeline principal : ce que le projet livre

- **Quoi** : prédiction A, post-traitée (D33), puis contrôle de plausibilité complet (R0 à R8, seuils figés, D36).
- **Contre quoi** : la vérité terrain, sur les **7 structures**.
- **Question** : le pipeline livré est-il acceptable (section 8.4), et le contrôle de plausibilité sonne-t-il à bon escient ?
- C'est la **seule comparaison qui porte le verdict** (P1).

### (b) Méthodes à armes égales : nnU-Net pli 0 contre SegResNet pli 0

- **Quoi** : N0 et S0, **sans post-traitement**, entraînés sur les **mêmes 12 cas** pendant le **même temps de calcul** (environ 8 h sur L4) (D40, D41).
- **Contre quoi** : la vérité terrain, sur les **7 structures** (D23 : l'aorte et l'AP restent comparables entre deux modèles qui ont appris la définition MM-WHS).
- **Question** : à données et temps égaux, une méthode segmente-t-elle mieux que l'autre sur ces 5 cas ? **Description seulement**, sans test statistique (section 8.3, P6).
- Pourquoi pas l'ensemble A contre S0 : on mélangerait l'effet de la méthode et l'effet de l'ensemble de 5 modèles (D41).
- Contrôle de plausibilité : **à titre informatif seulement** (P4).

### (c) Référence externe : TotalSegmentator

- **Quoi** : prédiction T de l'étape 4, **structures 1 à 5 seulement** ; étiquettes 6 et 7 de TotalSegmentator ignorées (D23).
- **Contre quoi** : la vérité terrain, et **en regard du pipeline A** sur les mêmes 5 structures (P5).
- **Mention de prudence obligatoire** à côté des résultats de l'OD et du VD : sur deux cas d'entraînement, les volumes de TotalSegmentator s'écartaient de la vérité (1002 : OD +46 %, VD −18 %), avec des frontières moins nettes (D23). Un mauvais score sur ces deux structures peut venir de la définition, pas de la qualité de l'outil.
- Contrôle de plausibilité **restreint** (D29, Q7) : R0 et R7, et les règles par structure (R1, R2, R8) **sur les structures 1 à 5 seulement** ; R3 à R6 ne sont pas appliquées (elles portent sur l'aorte ou l'AP). Alertes rapportées **à part**, à titre d'information. Le module figé (`scripts/controle_plausibilite.py`) contrôle toujours les 7 étiquettes : la restriction se fait **en écartant à la lecture** les résultats qui portent sur les étiquettes 6 et 7, sans modifier ni le module ni les seuils.
- Rappel : la phase 1 vise une chaîne de mesure fiable, **pas à battre TotalSegmentator** (fiche projet).

## 5. Métriques et règles de calcul

### 5.1 Pourquoi ces métriques (Metrics Reloaded)

*Metrics Reloaded* (Maier-Hein et al., *Nature Methods* 2024) est un guide de consensus international pour choisir les métriques de validation en analyse d'image. Notre problème y relève de la **segmentation sémantique** (chaque voxel reçoit une étiquette). Ce que le guide dit, et ce que nous en tirons :

| Ce que dit Metrics Reloaded | Conséquence ici |
|---|---|
| En segmentation, « l'utilisateur a typiquement un intérêt pour les **frontières**, les **centres** ou les **volumes** des structures » (éléments d'empreinte FP2.1 à FP2.3) ; il faut combiner des métriques aux forces complémentaires | trois familles : recouvrement (Dice), contour (HD95), volume (écart de volume) |
| Sous-processus S6 : métrique de **recouvrement** ; le DSC (Dice) et l'IoU sont mathématiquement liés, le choix relève de la préférence de la communauté | **Dice**, déjà utilisé à l'étape 5 (D26 à D41) |
| Sous-processus S7 : métrique de **contour**, à ajouter quand la frontière compte. La famille comprend la NSD, la distance de Hausdorff et sa variante au 95e percentile (HD95), et les distances de surface moyennes (*contenu de la figure S7 cité de mémoire : à relire dans l'Extended Data Fig. 7 avant le gel*) | **HD95** (exigée par la fiche projet) ; NSD en option (P2) |
| « Les valeurs de métrique doivent d'abord être calculées **par image**, puis agrégées sur l'ensemble des images », pour respecter la structure hiérarchique des données ; et, en cas de cas non indépendants, « respecter la structure hiérarchique lors de l'agrégation » (Extended Data Tab. 1, FP4.5) | calcul par cas et par structure, puis agrégation sur les 5 cas ; ici cas = patient (section 3.1) |
| « Utiliser les **implémentations de référence** » (définitions non standardisées, cas limites non définis) ; toutes les métriques du guide sont implémentées dans **MONAI** | calcul avec MONAI 1.6.1 (CLAUDE.md, règle 3) ; cas limites fixés en section 6 |
| « Inclure une visualisation des **valeurs brutes** » des métriques | toutes les valeurs par cas sont publiées, pas seulement les moyennes (section 8) |
| « Rapporter la **qualité de la référence** (variabilité intra- et inter-annotateurs) » ; choisir le nombre de décimales selon la pertinence et l'incertitude de la référence | variabilité inter-annotateurs inconnue pour MM-WHS : limite déclarée (section 12) ; arrondis fixés (section 8.1) |
| Envisager plusieurs passages sur le test quand l'algorithme n'est pas déterministe | un seul passage par modèle (lecture unique) : limite déclarée (section 12) |
| Analyse détaillée des échecs et performance **par sous-groupe** | tableaux par cas ; 1014 (0,45 mm) signalé ; alertes classées (section 7) |

Le volume n'est pas une métrique du « pool » de Metrics Reloaded pour la segmentation, mais le guide reconnaît l'intérêt pour les volumes (FP2.3) et prévoit des **métriques propres à l'application**. L'écart de volume est retenu à ce titre : c'est une mesure dérivée utile au clinicien (D31) et un critère de la fiche projet.

L'article compagnon (Reinke et al., *Nature Methods* 2024) décrit les pièges de ces métriques. Trois nous concernent directement : le Dice pénalise davantage une même erreur sur une petite structure ; la HD95 n'a pas une définition unique selon les logiciels ; et les cas où une structure manque rendent certaines métriques indéfinies. Les sections 5.2 et 6 tranchent ces points d'avance.

### 5.2 Définitions exactes

Toutes les métriques sont calculées **par cas et par structure**, sur des masques binaires (structure *k* contre le reste), dans la **géométrie d'origine** du scanner : pas de rééchantillonnage, espacement des voxels lu dans l'en-tête de l'image. Prédiction et vérité doivent avoir la même forme et la même affine (R0 ; l'*affine* décrit la position, la taille et l'orientation des voxels dans l'espace).

| Métrique | Définition | Outil | Unité, sens |
|---|---|---|---|
| **Dice** (DSC) | 2 × \|P ∩ V\| / (\|P\| + \|V\|), où P = voxels prédits de la structure, V = voxels de la vérité | MONAI 1.6.1, `compute_dice` (fond exclu) | 0 à 1 ; plus haut = mieux |
| **HD95** (distance de Hausdorff à 95 %) | On prend les voxels de **bord** de P et de V. Pour chaque point du bord de P, distance euclidienne au bord de V le plus proche ; on garde le **95e percentile** de ces distances. Même chose de V vers P. **HD95 = le plus grand des deux** (version *symétrique*). Distances calculées **en mm**, avec l'espacement réel des voxels | MONAI 1.6.1, `compute_hausdorff_distance(percentile=95, directed=False, spacing=espacement de l'image)` | mm ; plus bas = mieux |
| **Volume** | nombre de voxels × produit des 3 espacements / 1000, comme 05h et le contrôle de plausibilité | calcul direct (numpy) | mL |
| **Écart de volume** | **signé** : V_pred − V_ref en mL ; et 100 × (V_pred − V_ref) / V_ref en %. Positif = sur-segmentation | calcul direct | mL et % ; plus proche de 0 = mieux |

Précisions :
- **HD95** : d'autres logiciels prennent le 95e percentile de l'union des deux ensembles de distances ; les valeurs ne sont pas directement comparables à celles d'autres publications. C'est le piège « définition non standardisée » de Reinke et al. : on fixe la définition MONAI ci-dessus et on l'écrit dans le rapport.
- Le 95e percentile écarte les 5 % de distances les plus grandes : la HD95 résiste mieux qu'une Hausdorff simple à un point isolé, mais **un fragment égaré de plus de 5 % du bord la fait exploser**. C'est voulu : c'est le « pire endroit » de la segmentation.
- **Versions** : MONAI **1.6.1** (épinglée, D37), nnU-Net v2 **2.8.1** (D24), TotalSegmentator **2.18.0** (étape 4). Versions de Python, PyTorch, numpy, nibabel, scipy relevées et écrites dans la traçabilité.

### 5.3 Agrégats

Pour chaque comparaison, chaque modèle et chaque structure, sur les 5 cas :

| Métrique | Agrégats | « Pire cas » |
|---|---|---|
| Dice | médiane, moyenne, **minimum** | le minimum |
| HD95 | médiane, moyenne, **maximum** | le maximum |
| Écart de volume (mL et %) | médiane et moyenne **des valeurs signées** (biais : tendance à sur- ou sous-segmenter) ; médiane des **valeurs absolues** (taille typique de l'erreur) | l'écart de plus grande **valeur absolue**, affiché avec son signe |

Par cas, en plus : **Dice moyen des structures** (moyenne des 7 structures pour A, N0, S0 ; moyenne des structures 1 à 5 pour A et T dans la comparaison (c)), pour rester comparable au Dice moyen hors pli de l'étape 5 (0,906, D31). C'est un agrégat, déclaré ici, pas une nouvelle métrique.

Chaque cellule d'agrégat indique **n**, le nombre de cas pris en compte (5 sauf cas dégénéré, section 6). Les agrégats ne remplacent jamais les valeurs par cas, toujours publiées.

**Pas d'intervalle de confiance, pas de test de significativité** : avec 5 patients, ils seraient très larges ou trompeurs. On décrit, on ne conclut pas statistiquement (section 8.3).

## 6. Cas dégénérés (règles fixées d'avance)

Un cas est *dégénéré* pour une structure quand la métrique n'est pas définie normalement : une des deux régions est vide. Les comportements par défaut des logiciels varient (MONAI renvoie selon les cas `nan` ou `inf`, avec un avertissement) : le code de l'étape 6 **traite ces cas lui-même, avant d'appeler MONAI**, selon ce tableau.

| Situation (structure *k*, cas *c*) | Dice | HD95 | Écart de volume | Dans les agrégats |
|---|---|---|---|---|
| **D1. Prédiction vide**, vérité présente (structure oubliée) | **0** | **non calculable → valeur plafond** = diagonale physique du volume de l'image, en mm (la plus grande distance possible dans ce scanner), marquée « plafond » | −V_ref mL, **−100 %** | **compté**, avec ces valeurs (c'est un échec, il doit peser) |
| **D2. Vérité vide**, prédiction présente | non défini | non défini | +V_pred mL ; % non défini | **exclu**, n diminué, cas signalé comme **anomalie de la vérité** |
| **D3. Les deux vides** | non défini (jamais compté comme 1) | non défini | 0 mL ; % non défini | **exclu**, n diminué, anomalie signalée |
| **D4. Prédiction minuscule** (quelques voxels) | calculé normalement | calculé normalement | calculé normalement | compté (ce n'est pas un cas dégénéré) |

Remarques :
- La valeur plafond (P3) suit la recommandation générale de donner à un échec la **pire valeur possible**, plutôt que de l'écarter : l'écarter embellirait le résultat (même logique que D29, Q5).
- Les 7 structures sont présentes dans la vérité des 15 cas d'entraînement (spécification, R1). D2 et D3 ne sont donc pas attendus ; s'ils arrivent, c'est une **anomalie** rapportée telle quelle (section 11), sans correction de la vérité.
- Le seuil de 1 mL de R1 sert au **contrôle de plausibilité**, pas aux métriques : une structure de 0,3 mL n'est pas « vide » pour le Dice.
- Une prédiction de forme ou d'affine différente de l'image est détectée **avant l'ouverture** (R0, section 10, phase B) et corrigée à ce moment-là, puisqu'aucune vérité n'a encore été vue. Si cela arrivait malgré tout après l'ouverture, **aucun rééchantillonnage** : toutes les structures du cas sont traitées comme D1, et l'anomalie est consignée.

## 7. Contrôle de plausibilité sur le test

### 7.1 Application

| Comparaison | Règles | Seuils | Statut des alertes |
|---|---|---|---|
| (a) A post-traité | R0 à R8 | figés (D36) | **officielles** : comptées dans l'estimation des fausses alertes demandée par D36 |
| (b) N0 et S0 bruts | R0 à R8 | figés (D36) | **informatives** (P4) : non comptées dans l'estimation des fausses alertes, car les seuils ont été vérifiés sur des prédictions post-traitées (D33, D36) |
| (c) T | R0, R7 ; R1, R2, R8 sur les structures 1 à 5 | figés (D36) | **à part, informatives** (D29, Q7) |

Le contrôle ne lit **aucune vérité** (spécification, § 3) : il est donc exécuté **avant l'ouverture**, et son verdict (OK / ALERTE) est figé dans le manifeste (section 10, phase B). On ne peut pas l'influencer par la suite.

### 7.2 Présentation des alertes

Chaque alerte est nommée : **cas, règle, structure (ou rapport), mesure, seuil, catégorie** (« hors domaine » pour R6 et R7 ; « erreur probable » pour les autres). Les deux catégories sont présentées **séparément** (D29, Q8) : une alerte hors domaine décrit une **limite du modèle** (un examen inhabituel), pas une erreur de segmentation.

### 7.3 Vraie ou fausse alerte : règle fixée d'avance

D36 annonce que le test donnera l'**estimation indépendante des fausses alertes**. Il faut donc décider, avant de voir la vérité, ce qui rend une alerte « vraie ». Règle proposée (à valider, P8) ; le repère de 0,85 est le seuil « Dice faible » de D30, fixé avant de voir les plis 2 à 4 :

| Règle déclenchée | L'alerte est **vraie** si… | sinon |
|---|---|---|
| R0 | toujours vraie (masque techniquement inutilisable) | — |
| R1 (structure *k* absente) | la vérité de *k* fait plus de 1 mL | fausse |
| R2 (volume de *k*) | Dice(*k*) < 0,85 **ou** \|écart de volume de *k*\| > 30 % | fausse |
| R3 (rapport AP / aorte) | Dice(AP) < 0,85 **ou** Dice(aorte) < 0,85 | fausse |
| R4 (densité aorte / cavité VG) | Dice(aorte) < 0,85 **ou** Dice(cavité VG) < 0,85 | fausse |
| R5 (densité AP / VD) | Dice(AP) < 0,85 **ou** Dice(VD) < 0,85 | fausse |
| R8 (morceaux de *k*) | au moins un morceau de plus de 0,5 mL de la prédiction de *k* recouvre **moins de la moitié** de ses voxels avec la vérité de *k* (morceau égaré), **ou** Dice(*k*) < 0,85 | fausse |
| R6 (hors domaine : aorte moins dense que l'AP) | la même mesure, faite **sur la vérité** (densités médianes au cœur des structures, fonction du module figé), donne aussi aorte − AP < 0 HU | fausse |
| R7 (hors domaine : contraste faible) | la même mesure **sur la vérité** donne une densité de la cavité VG < 245 HU (seuil figé) | fausse |

Principe : une alerte « erreur probable » est vraie si la structure en cause est **réellement mal segmentée** ; une alerte « hors domaine » est vraie si **l'examen** est réellement hors domaine, ce qu'on vérifie en appliquant la même règle à la vérité.

On rapporte aussi l'inverse, les **échecs silencieux** : tout couple (cas, structure) avec un **Dice < 0,50** (repère « échec » de D30) **sans aucune alerte** sur ce cas. C'est exactement ce que D28 veut éviter.

Un examen visuel des cas en alerte (coupes avec contours superposés, comme 05g) est **permis et documenté**, mais il **ne change pas** le classement issu de la règle ; un désaccord entre l'œil et la règle est noté comme tel.

## 8. Présentation des résultats

### 8.1 Tableaux

Pour chaque comparaison :
1. **Tableau par cas et par structure** : Dice, HD95, volume prédit, volume de la vérité, écart de volume (mL, %), pour chaque modèle. Le cas **1014** est marqué « seul cas à 0,45 mm » (D20).
2. **Tableau des agrégats** (section 5.3), avec n.
3. **Alertes du contrôle** (section 7.2), avec leur classement vrai / faux (section 7.3) et les échecs silencieux.

Arrondis : Dice à 3 décimales (comme le journal), HD95 à 0,1 mm, volumes à 0,1 mL, pourcentages à l'unité. Metrics Reloaded rappelle qu'au-delà de la variabilité entre annotateurs, les dernières décimales n'ont pas de sens : une différence de Dice de 0,005 n'est pas une différence.

### 8.2 Métriques et alertes (D29, Q5)

Pour (a) : les métriques **principales** portent sur **les 5 cas, alertes comprises** ; le nombre de cas en alerte et les règles déclenchées sont donnés **à côté**. Les métriques **sans les cas en alerte** sont données **en complément**, clairement libellées, jamais comme résultat principal.

### 8.3 Pas de test statistique

Aucun test de significativité (valeur *p*), aucun intervalle de confiance : 5 patients ne le permettent pas. Pour (b), le rapport **décrit** : médianes, pires cas, et pour chaque structure le nombre de cas (sur 5) où N0 fait mieux que S0. Il ne déclare un « meilleur » que selon la règle descriptive de P6, si elle est retenue.

### 8.4 Verdict (selon P1)

La fiche projet demande « un verdict clair : acceptable ou non, et pourquoi », avec des seuils « fixés avant d'ouvrir le lot de test » (D10). **Proposition** (P1), pour le pipeline A seulement :

- **C1, performance** : pour chaque structure, **médiane du Dice sur les 5 cas ≥ 0,85** (structures 1 à 6) et **≥ 0,75 pour l'AP** (définition floue de son étendue, D31 ; Dice hors pli 0,827 après post-traitement, D33) ;
- **C2, sécurité** (D28) : **aucun échec silencieux** (aucun Dice < 0,50 sur un cas sans alerte du contrôle) ;
- **HD95 et volumes** : **décrits, sans seuil** (aucune référence vérifiée pour ces valeurs sur MM-WHS, variabilité entre annotateurs inconnue) ;
- **verdict** : « acceptable » si C1 et C2 sont remplis ; sinon « non acceptable », avec le critère manqué et la structure en cause. (b) et (c) n'entrent pas dans le verdict.

## 9. Vérifications au démarrage (avant toute lecture du test)

Le script de l'étape 6 s'**arrête**, sans lire ni `imagesTs` ni `labelsTs`, si l'une de ces vérifications échoue :

1. **Protocole figé** : empreinte SHA-256 de `docs/protocole_etape6.md` égale à celle inscrite au journal lors du gel.
2. **Fichier de seuils** : SHA-256 de `config/seuils_plausibilite.json` = `c1b247070765d193a46e90b538cc655787dbb65f8fdd13f113a39efd3ef90672` (D36) ; champ `statut` = « figé ».
3. **Les quatre valeurs non relues par le code** (README, « À faire avant l'étape 6 ») identiques entre le fichier figé et le code :
   - R0 : `tolerance_affine_mm` = 0,01 et `TOLERANCE_R0_MM` de `scripts/controle_plausibilite.py` ;
   - R0 : `etiquettes_autorisees` = 0 à 7 et `ETIQUETTES_AUTORISEES` ;
   - R8 : `morceau_min_ml` = 0,5 et `MORCEAU_ML` de `scripts/05i_postraitement_nnunet.py` ;
   - R8 : `connexite` = 26 et la connexité utilisée par `morceaux()` de 05i.
4. **Découpage** : SHA-256 de `splits/mmwhs_split.json` = `25881b214823758ddb708b6b46616c2422671c872035683852b1f045f4c4a05c` ; lot de test = exactement 1004, 1010, 1011, 1013, 1014.
5. **Versions** : MONAI 1.6.1 ; nnU-Net v2 2.8.1 pour les prédictions A et N0 ; archive des modèles Auto3DSeg `21ed8e5` (SHA-256 de D37) pour S0.
6. **Modèles terminés** : `checkpoint_final.pth` des 5 plis de nnU-Net, `postprocessing.pkl` de 05i, `model_final.pt` de `entrainement_segresnet_1250/pli_0` présents ; leurs empreintes sont relevées.

## 10. Règles d'ouverture et déroulé dans l'ordre

### 10.1 Règles

- **R-O1.** `labelsTs` est lu par **une seule exécution** d'un **seul script** d'évaluation, qui calcule d'un coup toutes les métriques des trois comparaisons. Aucune cellule, aucun autre script, aucun affichage ne lit `labelsTs` avant ni à côté.
- **R-O2.** Toutes les prédictions et tous les résultats du contrôle de plausibilité sont **produits et figés avant** l'ouverture (manifeste d'empreintes).
- **R-O3.** Après l'ouverture, **rien ne change** : ni modèle, ni réglage d'inférence, ni post-traitement, ni seuil, ni métrique, ni agrégat, ni règle de classement des alertes.
- **R-O4.** Toute anomalie est **rapportée telle quelle** (section 11). On ne refait pas une prédiction parce que son score est mauvais.

### 10.2 Déroulé

**Phase 0 — Gel du protocole.** Questions P1 à P10 tranchées par l'utilisateur ; document passé au statut « figé » ; commit ; SHA-256 du fichier inscrite au journal (nouvelle décision).

**Phase A — Écriture et essai à blanc du code (sans le test).** Scripts de l'étape 6 écrits, relus par l'agent qa-reviewer. Le script d'évaluation est d'abord exécuté sur des **cas d'entraînement** : prédictions hors pli de nnU-Net contre `labelsTr` (P9). Les Dice doivent concorder avec ceux de nnU-Net (`summary.json`, D31, D33) et ceux de 05m pour S0 sur 1001, 1019 et 1020. C'est la qualification opérationnelle (*OQ*) de l'outil de mesure : on vérifie que la règle mesure juste avant de mesurer la pièce.

**Phase B — Prédictions et contrôle (lecture de `imagesTs` seulement).**
1. Vérifications de la section 9.
2. Prédictions A (puis post-traitement D33), N0, S0 sur les 5 scanners de test. T existe déjà (étape 4).
3. Contrôle de forme, d'affine et d'étiquettes de chaque prédiction (R0). Un défaut technique se corrige **ici**, puisqu'aucune vérité n'a été vue ; la correction est consignée.
4. Contrôle de plausibilité (section 7.1) sur A, N0, S0 et T.
5. **Manifeste** (`manifeste_etape6.json`, dans Drive) : SHA-256 de chaque prédiction (20 fichiers), de chaque résultat du contrôle, des poids des modèles, du fichier de seuils, du protocole ; versions ; date. Son empreinte est inscrite au journal **avant** la phase C.

**Phase C — Ouverture (une seule fois).** Le script d'évaluation, au commit noté dans le journal :
1. vérifie le manifeste (toute prédiction modifiée depuis la phase B = arrêt) ;
2. relève l'empreinte des 5 masques de `labelsTs` ;
3. calcule toutes les métriques (sections 5 et 6), le classement des alertes (section 7.3) et les échecs silencieux ;
4. écrit tous les tableaux (section 8) dans Drive, en une fois, avec la traçabilité (commit, versions, empreintes, heures UTC et Paris).

**Phase D — Vérification QA, puis rapport.** L'agent qa-reviewer vérifie la plausibilité (Dice entre 0 et 1, volumes cardiaques réalistes, 5 cas, n corrects) et la concordance des chiffres annoncés avec les fichiers. Le verdict (section 8.4) est appliqué mécaniquement. Le rapport (étape 7) reprend les résultats sans les retoucher.

## 11. Anomalies et déviations

Une **anomalie** est un fait inattendu dans les données ou les résultats (vérité vide, géométrie incohérente, valeur impossible). Une **déviation** est un écart au protocole. Les deux sont **consignées** dans le journal des décisions, avec leur cause et leur effet possible sur le résultat.

| Situation | Conduite |
|---|---|
| Le calcul **plante avant** l'ouverture (phases A ou B) | corriger, relire (QA), relancer : ce n'est pas une déviation, aucune vérité n'a été vue |
| Le calcul **plante après** la lecture de `labelsTs`, pour une cause **extérieure au code** (session Colab coupée, Drive indisponible, mémoire) | **relance du même code, inchangé** (même commit), autorisée ; chaque tentative est consignée (heure, cause, sortie) |
| Le calcul plante après l'ouverture à cause d'un **défaut du code** | **déviation**. Correction **minimale**, limitée à ce qui empêche le calcul, **sans toucher** aux définitions des métriques, aux agrégats, aux seuils ni aux modèles ; diff relu par QA ; nouvelle ligne au journal ; les éventuels résultats partiels de la première tentative sont conservés et publiés à côté |
| Un résultat semble **faux** après l'ouverture (Dice incohérent, volume aberrant) | on vérifie d'abord le calcul (QA). Si le calcul est juste, le résultat est **publié tel quel**. Si un défaut du code est prouvé, même conduite que la ligne précédente |
| Vérité vide ou anomalie de la vérité | règles D2 / D3 (section 6) ; anomalie rapportée, vérité jamais corrigée |
| Seuil du contrôle mal adapté au test | noté comme **limite**, jamais corrigé (spécification, § 7.2) |

Interdits absolus : relancer une prédiction pour améliorer un score ; changer de modèle, de point de contrôle ou de post-traitement ; retirer un cas ; ajouter une métrique ; ouvrir `labelsTs` en dehors du script d'évaluation.

## 12. Limites (ce que ce protocole ne permet pas de conclure)

- **5 cas, 5 patients** : aucune conclusion statistique ; un seul cas difficile change fortement une médiane. Le résultat est une **performance interne** à une cohorte (MM-WHS), à confirmer sur des données externes (fiche projet, phase 2).
- **Un seul cas à 0,45 mm** (1014) : rien ne peut être conclu sur cette famille d'examens (D20).
- **(b) un seul modèle par méthode**, un seul entraînement : on ne sait pas quelle part de l'écart vient de la méthode et quelle part du hasard de l'entraînement ; même temps de calcul, mais environ 8 fois moins de morceaux d'image vus par SegResNet (D40, D41). Les réglages d'inférence (par exemple l'augmentation au moment de la prédiction) peuvent aussi différer (P7).
- **(c) définitions différentes** : TotalSegmentator n'est comparé que sur 5 structures, avec prudence sur l'OD et le VD (D23) ; un écart peut venir de la définition, pas de la qualité.
- **Contrôle de plausibilité** : sur 5 cas, l'estimation des fausses alertes est très imprécise ; si aucun cas de test n'est hors domaine, le test ne dit rien de la **sensibilité** (part des échecs attrapés) (spécification, § 7.2).
- **Qualité de la référence inconnue** : une seule annotation par cas, sans mesure de la variabilité entre annotateurs ; une partie de l'écart sur les frontières floues (valve pulmonaire, étendue de l'AP) est du bruit d'annotation (D31).
- **Un seul passage** par modèle sur le test : la variabilité due au caractère non déterministe des calculs n'est pas mesurée.
- **Prédictions de TotalSegmentator** : l'étape 4 a enregistré les empreintes des scanners, pas celles des prédictions. Leur intégrité depuis l'étape 4 repose sur les dates de Drive ; elles sont figées par empreinte à la phase B seulement.

## 13. Questions ouvertes (à trancher avant le gel)

**P1. Seuils d'acceptation et verdict.** La fiche projet exige un verdict « acceptable ou non », avec des seuils fixés avant le test (D10 : tirés de la littérature et de Metrics Reloaded). Les consignes de rédaction ne parlent que de description. Sans seuils fixés maintenant, le verdict de l'étape 7 serait posé après avoir vu les chiffres : c'est le biais que tout le projet cherche à éviter.
*Recommandation* : **fixer un verdict maintenant**, sur le pipeline A seulement, avec les critères C1 et C2 de la section 8.4. Ce n'est pas un test statistique : c'est un critère d'acceptation, comme en qualification. Les valeurs proposées (0,85 ; 0,75 pour l'AP ; 0,50) viennent des repères internes de D30 et des résultats hors pli de l'étape 5. Elles ne sont **pas encore justifiées par la littérature** comme le demande D10. Avant le gel, il faudrait les comparer aux résultats publiés du défi MM-WHS en CT (Zhuang et al., *Medical Image Analysis* 2019), à vérifier, ou les assumer explicitement comme repères internes. HD95 et volumes restent sans seuil, faute de référence.

**P2. Ajouter la NSD ?** La *NSD* (distance de surface normalisée) mesure la part du contour prédit située à moins d'une tolérance τ (en mm) du contour de référence. Elle figure dans la famille des métriques de contour de Metrics Reloaded (S7), et MONAI la calcule (`compute_surface_dice`). Mais τ doit refléter la variabilité entre annotateurs, inconnue ici.
*Recommandation* : l'ajouter **maintenant**, comme métrique **secondaire, hors verdict**, avec **τ = 2 mm** (repère « erreur proche ≤ 2 mm » de D30), mêmes règles de cas dégénérés que le Dice (D1 : 0). Après l'ouverture, ce ne sera plus possible. Si l'utilisateur préfère rester simple, ne pas l'ajouter est défendable : HD95 et Dice couvrent déjà contour et recouvrement.

**P3. Valeur plafond de la HD95 (structure oubliée).**
*Recommandation* : **diagonale physique du volume de l'image** (section 6, D1), marquée « plafond » dans les tableaux. Alternative écartée : exclure le cas, ce qui embellirait le résultat.

**P4. Contrôle de plausibilité sur (b).** Les seuils ont été vérifiés sur des prédictions post-traitées (D33 à D36). Sur des prédictions brutes, R8 sonnerait souvent à cause des fragments que le post-traitement aurait retirés (D32).
*Recommandation* : l'appliquer **à titre informatif**, rapporté à part, **hors** de l'estimation des fausses alertes et hors verdict. Cela montre, pour la phase 2, comment le contrôle se comporte sur une autre méthode, sans brouiller le résultat officiel.

**P5. Contre quoi comparer TotalSegmentator ?**
*Recommandation* : contre **A seulement** (le pipeline livré face à la référence externe), sur les structures 1 à 5. Les valeurs de N0 et S0 sur ces structures restent visibles dans les tableaux de (b), mais ne sont pas commentées face à TotalSegmentator : trois comparaisons distinctes, trois questions distinctes.

**P6. Désigner une « meilleure approche » en (b) ?** La fiche projet demande « la meilleure approche parmi les candidates, sur le même lot de test ».
*Recommandation* : une **règle descriptive** fixée maintenant, sans valeur statistique : une méthode est dite « meilleure sur la structure *k* » si sa médiane de Dice est supérieure de plus de 0,02 **et** si elle fait mieux sur au moins 4 cas sur 5 ; sinon « pas de différence établie sur 5 cas ». Le rapport dit explicitement qu'il ne s'agit pas d'une preuve.

**P7. Réglages d'inférence.** nnU-Net applique par défaut une augmentation au moment de la prédiction (il moyenne l'image et ses symétriques, *mirroring*). Le réglage équivalent de SegResNet n'est pas documenté dans nos scripts.
*Recommandation* : **réglages par défaut de chaque outil**, identiques à ceux de l'évaluation hors pli de l'étape 5 (05c pour nnU-Net, 05m pour SegResNet), **relevés et écrits** dans le manifeste avant l'ouverture. Si l'un utilise l'augmentation et l'autre non, le signaler comme limite de (b), sans l'aligner après coup.

**P8. Règle de classement des alertes (section 7.3).**
*Recommandation* : adopter la règle telle quelle, avec le repère 0,85 (D30) et 30 % d'écart de volume pour R2. Un examen visuel reste permis, documenté, sans changer le classement.

**P9. Essai à blanc du code d'évaluation sur l'entraînement (phase A).**
*Recommandation* : **oui, obligatoire** avant l'ouverture. C'est le meilleur moyen d'éviter la déviation « défaut du code après l'ouverture » (section 11). Critère : Dice identiques à ceux de nnU-Net et de 05m (à l'arrondi près) ; HD95 et volumes plausibles ; volumes identiques à ceux de 05h sur la vérité.

**P10. Gel du protocole.**
*Recommandation* : même procédure que D36 : statut « figé », date, commit, SHA-256 du fichier au journal ; le fichier exempté de conversion des fins de ligne dans `.gitattributes`, pour que l'empreinte soit la même sous Windows et dans Colab. La vérification de cette empreinte est la première du script (section 9, point 1).

## 14. Références

- L. Maier-Hein, A. Reinke et al., « Metrics reloaded: recommendations for image analysis validation », *Nature Methods* 21(2):195-212, 2024. doi:[10.1038/s41592-023-02151-z](https://doi.org/10.1038/s41592-023-02151-z) (texte consulté via PubMed Central, PMC11182665 : section « Semantic segmentation », Extended Data Tab. 1, Extended Data Fig. 6 et 7).
- A. Reinke, M. D. Tizabi et al., « Understanding metric-related pitfalls in image analysis validation », *Nature Methods* 21(2):182-194, 2024. doi:[10.1038/s41592-023-02150-0](https://doi.org/10.1038/s41592-023-02150-0).
- X. Zhuang et al., « Evaluation of algorithms for multi-modality whole heart segmentation: an open-access grand challenge », *Medical Image Analysis*, 2019 (défi MM-WHS) : **à vérifier** avant de s'en servir pour P1.
- MONAI 1.6.1, `monai.metrics` : `compute_dice`, `compute_hausdorff_distance`, `compute_surface_dice`.
- Projet : `docs/fiche_projet.pdf` (« Critères de réussite », « Livrables ») ; `docs/journal_decisions.md` (D16 à D41) ; `docs/specification_controle_plausibilite.md` (v1.1, figée) ; `docs/correspondance_etiquettes.md` ; `config/seuils_plausibilite.json` ; `README.md` (« À faire avant l'étape 6 »).
