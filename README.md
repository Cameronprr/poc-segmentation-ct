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
| 5. Entraînement | nnU-Net v2 puis Auto3DSeg sur les 15 cas | nnU-Net, MONAI Auto3DSeg, MLflow, `scripts/05a_preparer_nnunet.py`, `scripts/05b_essai_nnunet.py`, `scripts/05c_entrainer_nnunet.py`, `scripts/05d_analyser_1019.py`, `scripts/05e_densites_vaisseaux.py`, `scripts/05f_bilan_validation_croisee.py`, `scripts/05g_examiner_cas.py`, `scripts/05h_volumes_verite.py`, `scripts/05i_postraitement_nnunet.py`, notebook sections 12 à 22 | 🟡 nnU-Net : préparation faite, essai de 5 époques fait (L4 : 114,5 s par époque) ; entraînement 250 époques par pli (D25). **Pli 0 terminé** : Dice moyen de validation 0,850 sur 1001, 1019 et 1020 (1001 et 1020 = même patient). Cavités et myocarde entre 0,84 et 0,96 sur les 3 cas, aorte 0,97-0,98 sur 1001 et 1020, mais **échec sur les vaisseaux de 1019** : aorte 0,00 (378 000 voxels dans la vérité, 15 000 prédits), artère pulmonaire 0,46. Cause trouvée (section 16, D27) : **phase d'injection atypique** (aorte moins dense que l'artère pulmonaire). 1019 est gardé dans les données ; cette limite sera inscrite au rapport. Incident de reprise clos (section 17) : 2 époques refaites, aucun défaut. 05c recopie maintenant sa sortie dans Drive et affiche l'heure de Paris. Contrôle de plausibilité (D28) : spécification rédigée (`docs/specification_controle_plausibilite.md`), seuils de densité (R4 à R7) à calibrer avec la section 18 (densités des vaisseaux sur les cas d'entraînement) ; mesure des volumes pour R2 et R3 à prévoir (D29). **Pli 1 terminé** (D30) : Dice moyen de validation 0,919 sur 1005, 1012 et 1015 (1005 et 1012 = même patient). 1005 et 1012 très bons ; **1015 plus faible** sur le VD (0,83) et l'artère pulmonaire (0,65), avec à la fois des oublis et des ajouts : limite VD/AP ou étendue des branches différente ? Phase d'injection classique. **Plis 2, 3 et 4 terminés** (entraînés sans analyse préalable, D30) : Dice moyens de validation 0,908 (pli 2), 0,933 (pli 3) et 0,920 (pli 4). **Validation croisée nnU-Net complète (5 plis × 250 époques).** Bilan des 15 cas de validation exécuté (05f, section 19) : Dice moyen hors pli **0,906 sur les cas**, **0,903 par patient** (13 patients, les paires 1001/1020 et 1005/1012 comptées une fois), médiane 0,924. Examen des cas faibles exécuté (05g, section 20) sur 1002, 1015 et 1018. Classement (D31) : hors domaine = vaisseaux de 1019 ; annotation atypique = OD de 1002 ; définition floue = AP et VD autour de la valve pulmonaire (1015) ; frontière ambiguë = myocarde VG de 1018 ; erreur du modèle = fragment isolé de l'AP de 1018 (à détecter par R8). **Aucun échec majeur ne relève d'une erreur pure du modèle.** Décalages des limites AP / VD (05f, section 19) et volumes de la vérité (05h, section 21) exécutés, contrôles conformes (D32) : **l'AP prédite est en plusieurs morceaux dans 9 cas sur 15** (jusqu'à 7 pour 1008), ce qui fausse les décalages (1008 : −97 mm avec un Dice AP de 0,915) et ferait sonner R8 sur la majorité des cas ; volumes de la vérité très variables (cavité VG de 30 à 150 mL, phase du cycle cardiaque), donc R2 et R3 ne détecteront que les erreurs grossières. **Préparés, pas encore exécutés** : post-traitement officiel de nnU-Net (`nnUNetv2_find_best_configuration` : ne garder que le plus grand morceau, d'abord pour tout le cœur puis structure par structure, retenu seulement s'il améliore le Dice) déterminé sur les 15 prédictions hors pli, Dice avant / après et nombre de morceaux (05i, section 22 ; cette commande écrit dans `nnunet_results`, sans toucher aux prédictions d'origine) ; décalage du bas de l'AP sur le plus grand morceau seulement, à côté du calcul actuel (05f, section 19). **Reste pour l'étape 5** : décision sur le post-traitement, puis calibrage du contrôle de plausibilité (R2 à R7 sur la vérité, R8 et fausses alertes sur les prédictions retenues, D32), puis Auto3DSeg |
| 6. Évaluation | Dice, HD95, volumes sur les 5 cas de test | MONAI (métriques), agent qa-reviewer | ⬜ |
| 7. Rapport | Comparaison des trois approches, verdict | — | ⬜ |

### Vérifications à faire au début de l'étape 5
- **Géométrie image / masque vue par nnU-Net** (relecture QA de l'étape 3, point A3). Le script 03 compare les affines avec MONAI et nibabel, mais nnU-Net lit les fichiers avec SimpleITK. Or l'image est copiée telle quelle, alors que le masque est réenregistré : son en-tête d'origine est recopié, seul le type de données change (uint8). Avant tout entraînement, définir les variables d'environnement `nnUNet_raw`, `nnUNet_preprocessed` et `nnUNet_results` vers des dossiers de Drive, puis lancer `nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity`. Cette commande vérifie, cas par cas, que l'image et son masque ont le même espacement, la même origine, la même orientation et la même taille.
- **Listes de données** : elles sont construites uniquement depuis `imagesTr` et `labelsTr`, jamais depuis `labelsTs` (décision D18).
- **Validation croisée par patient (D20)** : les deux paires `{1001, 1020}` et `{1005, 1012}` (clé `groupes_patients` du découpage) sont en entraînement. Chaque paire doit rester dans la même part de validation croisée (part = *fold* : sous-ensemble mis de côté à tour de rôle pour valider), sinon le modèle serait validé sur un patient qu'il a vu à l'entraînement :
  - **nnU-Net** répartit par défaut les cas au hasard, cas par cas, entre ses 5 parts. Il faut lui fournir un fichier `splits_final.json` construit par patient ;
  - **Auto3DSeg** (MONAI) a le même comportement : il faut fixer les parts (champ `fold` de la liste de données) par patient.

Phase 2 (plus tard) : jeu ImageCAS annoté (plus de cas), scanners d'autres hôpitaux, cœurs pathologiques, scanners sans contraste.

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
