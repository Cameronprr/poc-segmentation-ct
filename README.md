# POC segmentation CT cardiaque (projet personnel)

Pipeline de bout en bout : un scanner CT entre, une segmentation du cœur sort, et un rapport la compare à la vérité terrain d'experts.
Projet personnel, données publiques uniquement, aucun lien avec un employeur.

## Les trois lieux de travail

| Lieu | Rôle |
|---|---|
| Projet Claude.ai | Cadrer, décider, documenter (fiche projet) |
| Ce dépôt + Claude Code | Les agents écrivent et vérifient les scripts |
| Google Colab + Google Drive | Exécution sur GPU ; les données et les résultats restent dans Drive |

Boucle à chaque étape : l'agent écrit le script ici → le dépôt est mis à jour sur GitHub → Colab récupère le dépôt et exécute → les résultats arrivent dans Drive → l'agent QA vérifie.

## Feuille de route — phase 1 (MM-WHS, 20 cas)

| Étape | Contenu | Outils | État |
|---|---|---|---|
| 0. Cadrage | But, périmètre, décisions | Projet Claude.ai | ✅ |
| 1. Installer les lieux | Ce dépôt, agents, Colab relié à Drive | GitHub, Claude Code, Colab | ✅ |
| 2. Données | Télécharger MM-WHS dans Drive, vérifier ; découpage figé (`splits/mmwhs_split.json`) | Colab, `scripts/02_verifier_donnees.py` | ✅ 20 cas conformes |
| 3. Préparation | Contrôle qualité détaillé, conversion au format nnU-Net | MONAI, `scripts/03_preparer.py` | ✅ `Dataset101_MMWHS` reconstruit avec le découpage par patient (D20) : CONFORME, 15 cas en entraînement, 5 en test (1004, 1010, 1011, 1013, 1014) ; aucune paire test/entraînement au-dessus de 0,95 (corrélation maximale 0,638) |
| 4. Ligne de base | TotalSegmentator sur les 5 cas de test | TotalSegmentator, agent ml-engineer | ✅ `heartchambers_highres` : 5 cas prédits en 576 s (GPU T4), contrôle CONFORME (même forme et même affine que les scanners, étiquettes 1 à 7) ; aucun score calculé (D19). Correspondance des étiquettes : `docs/correspondance_etiquettes.md` |
| 4 bis. Définitions | Avant l'étape 6 : TotalSegmentator sur 2 cas d'**entraînement** (1003 à 0,45 mm, 1002 à 0,625 mm), contours superposés à la vérité terrain, sans score, pour fixer les règles de comparaison sans regarder le test | `scripts/04b_comparer_definitions.py`, notebook section 11 | ✅ exécutée : même définition pour myocarde VG, OG, cavité VG ; OD et VD comparables avec prudence ; aorte et artère pulmonaire définies différemment par TotalSegmentator. Règles de l'étape 6 : D23 |
| 5. Entraînement | nnU-Net v2 puis Auto3DSeg sur les 15 cas | nnU-Net, MONAI Auto3DSeg, MLflow, `scripts/05a_preparer_nnunet.py`, `scripts/05b_essai_nnunet.py`, `scripts/05c_entrainer_nnunet.py`, `scripts/05d_analyser_1019.py`, `scripts/05e_densites_vaisseaux.py`, `scripts/05f_bilan_validation_croisee.py`, `scripts/05g_examiner_cas.py`, `scripts/05h_volumes_verite.py`, `scripts/05i_postraitement_nnunet.py`, `scripts/05j_calibrer_controle.py`, `scripts/controle_plausibilite.py`, notebook sections 12 à 23 | 🟡 nnU-Net : préparation faite, essai de 5 époques fait (L4 : 114,5 s par époque) ; entraînement 250 époques par pli (D25). **Pli 0 terminé** : Dice moyen de validation 0,850 sur 1001, 1019 et 1020 (1001 et 1020 = même patient). Cavités et myocarde entre 0,84 et 0,96 sur les 3 cas, aorte 0,97-0,98 sur 1001 et 1020, mais **échec sur les vaisseaux de 1019** : aorte 0,00 (378 000 voxels dans la vérité, 15 000 prédits), artère pulmonaire 0,46. Cause trouvée (section 16, D27) : **phase d'injection atypique** (aorte moins dense que l'artère pulmonaire). 1019 est gardé dans les données ; cette limite sera inscrite au rapport. Incident de reprise clos (section 17) : 2 époques refaites, aucun défaut. 05c recopie maintenant sa sortie dans Drive et affiche l'heure de Paris. Contrôle de plausibilité (D28) : spécification rédigée (`docs/specification_controle_plausibilite.md`), seuils de densité (R4 à R7) à calibrer avec la section 18 (densités des vaisseaux sur les cas d'entraînement) ; mesure des volumes pour R2 et R3 à prévoir (D29). **Pli 1 terminé** (D30) : Dice moyen de validation 0,919 sur 1005, 1012 et 1015 (1005 et 1012 = même patient). 1005 et 1012 très bons ; **1015 plus faible** sur le VD (0,83) et l'artère pulmonaire (0,65), avec à la fois des oublis et des ajouts : limite VD/AP ou étendue des branches différente ? Phase d'injection classique. **Plis 2, 3 et 4 terminés** (entraînés sans analyse préalable, D30) : Dice moyens de validation 0,908 (pli 2), 0,933 (pli 3) et 0,920 (pli 4). **Validation croisée nnU-Net complète (5 plis × 250 époques).** Bilan des 15 cas de validation exécuté (05f, section 19) : Dice moyen hors pli **0,906 sur les cas**, **0,903 par patient** (13 patients, les paires 1001/1020 et 1005/1012 comptées une fois), médiane 0,924. Examen des cas faibles exécuté (05g, section 20) sur 1002, 1015 et 1018. Classement (D31) : hors domaine = vaisseaux de 1019 ; annotation atypique = OD de 1002 ; définition floue = AP et VD autour de la valve pulmonaire (1015) ; frontière ambiguë = myocarde VG de 1018 ; erreur du modèle = fragment isolé de l'AP de 1018 (à détecter par R8). **Aucun échec majeur ne relève d'une erreur pure du modèle.** Décalages des limites AP / VD (05f, section 19) et volumes de la vérité (05h, section 21) exécutés, contrôles conformes (D32) : **l'AP prédite est en plusieurs morceaux dans 9 cas sur 15** (jusqu'à 7 pour 1008), ce qui fausse les décalages (1008 : −97 mm avec un Dice AP de 0,915) et ferait sonner R8 sur la majorité des cas (avant post-traitement ; après, 3 cas gardent plus d'un morceau de l'AP de plus de 0,5 mL, D33) ; volumes de la vérité très variables (cavité VG de 30 à 150 mL, phase du cycle cardiaque), donc R2 et R3 ne détecteront que les erreurs grossières. **Post-traitement décidé (D33)** : nnU-Net (05i, section 22) retient « garder le plus grand morceau » pour OG, OD, VD et aorte, pas pour le myocarde, la cavité VG ni l'AP (Dice de l'AP 0,8271 → 0,8224 avec ce nettoyage) ; Dice moyen inchangé (0,906). **Appliqué tel quel, sans l'étendre à l'AP** : ses fragments sont confiés à R8. Sur le plus grand morceau, le bas de l'AP varie de −29 à +46 mm, écart-type 21 mm (48 mm tous morceaux ; 05f, partie 5 bis). **Préparé, pas encore exécuté** : calibrage du contrôle de plausibilité (R2 à R8 sur la vérité des 14 cas sans 1019, D34) et application aux 15 prédictions post-traitées, avec les règles déclenchées par cas (05j, section 23) ; seuils **proposés**, à valider avant d'être figés. **Reste pour l'étape 5** : validation et gel des seuils, puis Auto3DSeg |
| 6. Évaluation | Dice, HD95, volumes sur les 5 cas de test | MONAI (métriques), agent qa-reviewer | ⬜ |
| 7. Rapport | Comparaison des trois approches, verdict | — | ⬜ |

### Vérifications à faire au début de l'étape 5
- **Géométrie image / masque vue par nnU-Net** (relecture QA de l'étape 3, point A3). Le script 03 compare les affines avec MONAI et nibabel, mais nnU-Net lit les fichiers avec SimpleITK. Or l'image est copiée telle quelle, alors que le masque est réenregistré : son en-tête d'origine est recopié, seul le type de données change (uint8). Avant tout entraînement, définir les variables d'environnement `nnUNet_raw`, `nnUNet_preprocessed` et `nnUNet_results` vers des dossiers de Drive, puis lancer `nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity`. Cette commande vérifie, cas par cas, que l'image et son masque ont le même espacement, la même origine, la même orientation et la même taille.
- **Listes de données** : elles sont construites uniquement depuis `imagesTr` et `labelsTr`, jamais depuis `labelsTs` (décision D18).
- **Validation croisée par patient (D20)** : les deux paires `{1001, 1020}` et `{1005, 1012}` (clé `groupes_patients` du découpage) sont en entraînement. Chaque paire doit rester dans la même part de validation croisée (part = *fold* : sous-ensemble mis de côté à tour de rôle pour valider), sinon le modèle serait validé sur un patient qu'il a vu à l'entraînement :
  - **nnU-Net** répartit par défaut les cas au hasard, cas par cas, entre ses 5 parts. Il faut lui fournir un fichier `splits_final.json` construit par patient ;
  - **Auto3DSeg** (MONAI) a le même comportement : il faut fixer les parts (champ `fold` de la liste de données) par patient.

Phase 2 (plus tard) : jeu ImageCAS annoté (plus de cas), scanners d'autres hôpitaux, cœurs pathologiques, scanners sans contraste.

## Stratégie face aux cas difficiles

Un modèle peut avoir une bonne moyenne et échouer complètement sur un type d'examen : au pli 0, le Dice moyen de 0,850 cachait une aorte à 0,00 sur 1019 (D26). La stratégie du projet tient en quatre couches successives. Chacune rattrape ce que la précédente a laissé passer. Pour chaque couche : ce que le POC a fait (avec les décisions du journal) et ce qui reste pour la phase 2.

Un **sous-groupe** est une famille d'examens qui partagent une caractéristique susceptible de changer le comportement du modèle (épaisseur de coupe, moment de l'injection, etc.).

### 1. Connaître ses données : cartographier les sous-groupes

**Ce que le POC a fait**
- **Épaisseur de coupe** : deux familles repérées dans l'en-tête des fichiers, 0,45 mm (5 cas, probablement un autre scanner) et 0,625 mm (15 cas) (D20).
- **Patients en double** : deux paires de cas qui sont très probablement le même patient (1001/1020 et 1005/1012), trouvées par la géométrie des fichiers et un examen visuel (D20).
- **Phase d'injection** (moment où les images sont prises après l'injection du produit de contraste) : 1019 a une phase inversée, avec l'aorte moins dense que l'artère pulmonaire (D27). Un script mesure les densités des vaisseaux sur les cas d'entraînement (05e, section 18).
- **Phase du cycle cardiaque** (systole : cœur contracté ; diastole : cœur relâché) : volumes de la vérité très variables, cavité VG de 30 à 150 mL (D32). 1001 et 1020 sont probablement le même patient à deux phases (D20).
- **Annotations** : définitions comparées à celles de TotalSegmentator sur des cas d'entraînement (D22, D23) ; annotation atypique de l'OD (1002), définition floue de la valve pulmonaire et de l'étendue de l'AP (D31).

**Ce qui reste pour la phase 2**
- Ajouter les sous-groupes absents du POC : **scanners sans contraste**, **autres centres** (hôpitaux, constructeurs, protocoles), **cœurs pathologiques**.
- Relever la phase d'injection et la phase cardiaque **pour chaque examen** (en-tête DICOM, mesure automatique), au lieu de les découvrir après un échec.
- Écrire un protocole d'annotation avec des définitions précises des frontières (plans valvulaires, étendue des branches) (D31).

### 2. Bien apprendre : chaque sous-groupe assez représenté

**Ce que le POC a fait**
- **Découpage par patient** : un même patient n'est jamais à la fois en test et en entraînement, ni dans deux plis de validation croisée (D20, D24).
- **Découpage stratifié** par épaisseur de coupe (chaque famille a sa part, au lieu d'un tirage au hasard qui peut tout mettre du même côté) : dans le test, 1 cas à 0,45 mm et 4 à 0,625 mm (D20) ; dans les plis, les 3 patients d'entraînement à 0,45 mm répartis dans 3 plis différents (D24).
- **Aucune donnée retirée ni corrigée d'après un résultat** : 1019, 1015 et les autres cas faibles sont gardés (D27, D30, D31).
- Données d'entraînement prises uniquement dans le lot d'entraînement (D18). Jamais d'apprentissage sur des masques produits par TotalSegmentator (règle de licence, CLAUDE.md).

**Limites constatées** : 1019 est le seul cas à phase d'injection inversée ; le modèle ne peut pas apprendre cette situation (D27). Seulement 3 patients d'entraînement à 0,45 mm (D24).

**Ce qui reste pour la phase 2**
- Assez de cas dans **chaque** sous-groupe : plusieurs phases d'injection (D27), avec et sans contraste, plusieurs centres, des cœurs pathologiques (jeu ImageCAS et autres sources).
- Stratifier le découpage sur **tous** les sous-groupes connus, pas seulement l'épaisseur de coupe.
- Étude de variabilité entre annotateurs, sur le principe d'un **Gage R&R** (mesure de la part de variation due à l'instrument et à l'opérateur), pour connaître le bruit d'annotation que le modèle ne peut pas réduire (D31).

### 3. Bien évaluer : par cas et par sous-groupe, jamais seulement la moyenne

**Ce que le POC a fait**
- **Score par cas et par structure**, pas seulement la moyenne : l'échec de 1019 a été vu ainsi, et le pli 1 n'a été lancé qu'après son analyse (D26).
- **Moyenne par patient** en plus de la moyenne par cas, pour ne pas compter deux fois les paires (D30) : Dice hors pli 0,906 sur les cas, 0,903 par patient, médiane 0,924 (D31).
- **Examen des cas faibles** avec figures et indicateurs (05d, 05g) et **classement des causes d'échec** (D31) :
  - hors domaine : vaisseaux de 1019 ;
  - annotation atypique : OD de 1002 ;
  - définition floue : AP et VD autour de la valve pulmonaire (1015) ;
  - frontière ambiguë : myocarde VG de 1018 ;
  - erreur du modèle : fragment isolé de l'AP de 1018.

  Aucun échec majeur ne relève d'une erreur pure du modèle.
- **Mesures en plus du Dice** : décalage en mm des limites de l'AP et du VD, et nombre de morceaux, qui ont révélé les fragments de l'AP (D32).
- **Règles du verdict fixées avant de regarder le test** ; lot de test utilisé une seule fois, à l'étape 6 (D16, D18, D19, D22, D23 ; CLAUDE.md, règle 4) ; métriques principales sur tous les cas, alertes comprises (D29, Q5).

**Ce qui reste**
- **Étape 6 (phase 1)** : Dice, HD95 et volumes sur les 5 cas de test. Avec un seul cas à 0,45 mm, aucune conclusion propre à cette famille n'est possible (D20).
- **Phase 2** : résultats rapportés **par sous-groupe**, avec assez de cas pour conclure ; erreur sur les **mesures dérivées** utiles au clinicien, comme la masse du VG, en plus du Dice (D31) ; comparaison à la variabilité entre annotateurs (D31).

### 4. Se protéger en service : ne jamais rendre un résultat faux en silence

**Ce que le POC a fait**
- **Post-traitement** de nnU-Net (ne garder que le plus grand morceau) déterminé sur les 15 prédictions hors pli (05i, section 22) et appliqué tel que retenu : OG, OD, VD et aorte. Pas sur l'AP, qui peut légitimement être en plusieurs morceaux ; ses fragments sont confiés au contrôle de plausibilité (D33).
- **Contrôle de plausibilité en sortie** (D28) : présence, volume, rapports de volumes, densités et nombre de morceaux de chaque structure, règles R0 à R8 (D29 ; spécification dans `docs/specification_controle_plausibilite.md`). Mesures de calibrage : volumes et rapports mesurés sur la vérité (05h, section 21, D32) ; densités mesurées par 05e (section 18 : 1019 seul cas à phase d'injection inversée ; résultats dans la spécification). 1019 est exclu du calibrage, et chaque règle est vérifiée en laissant chaque patient de côté à tour de rôle, pour éviter un raisonnement circulaire (D28). Calibrage sur la vérité, R8 comprise (D34), puis application aux prédictions post-traitées (D32, D33) : préparé (05j, section 23), seuils proposés à valider ; ils seront figés avant l'étape 6 (D28 ; D29, Q9).
- **Deux types d'alertes**, présentés séparément : « hors domaine », qui signale un examen inhabituel (limite du modèle), et « erreur probable » (D29, Q8).
- **Revue humaine des alertes** : le POC signale et rapporte chaque alerte (nombre d'alertes publié à côté des métriques, D29, Q5) ; c'est un humain qui lit et tranche.

**Ce qui reste pour la phase 2**
- **Contrôle d'entrée**, avant la segmentation : lecture de l'en-tête DICOM, estimation de la phase d'injection, détection des scanners sans contraste (D28).
- Règle **R9** (positions et contacts entre structures), reportée à une version 2 (D29, Q1).
- **Circuit de revue humaine** organisé : qui revoit les alertes, dans quel délai, et ce que l'on fait d'un examen signalé.
- Recalibrer le contrôle de plausibilité sur des données plus variées : avec 13 patients (12 pour le calibrage, 1019 exclu), les plages seront larges (D29, Q2) et R2/R3 ne détecteront que les erreurs grossières (D32).

## Organisation du dépôt

```
CLAUDE.md                     Instructions pour Claude Code
.claude/agents/               Les agents (data-engineer, ml-engineer, qa-reviewer)
docs/journal_decisions.md     Décisions tracées
docs/correspondance_etiquettes.md  Étiquettes TotalSegmentator ↔ MM-WHS (avant l'étape 6)
splits/mmwhs_split.json       Découpage figé entraînement / test
scripts/                      Scripts du pipeline, numérotés par étape
notebooks/colab_runner.ipynb  Notebook qui exécute le pipeline dans Colab
```

La fiche projet (cadrage complet, décisions D1 à D15) est un document local, non publié dans ce dépôt.

## Démarrer

Dans Google Colab : Fichier → Importer un notebook → onglet GitHub → ce dépôt → `notebooks/colab_runner.ipynb`. Activer le GPU, exécuter les cellules dans l'ordre.

## Données

MM-WHS (Zhuang et al.), site officiel : https://zmiclab.github.io/zxh/0/mmwhs/ — usage personnel non commercial, source citée. Les données ne sont jamais versionnées dans ce dépôt.
