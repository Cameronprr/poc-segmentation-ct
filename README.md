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
| 4. Ligne de base | TotalSegmentator sur les 5 cas de test | TotalSegmentator, agent ml-engineer | 🟡 `scripts/04_totalsegmentator.py` relu par QA (tâche `heartchambers_highres`, prédictions seulement, D19) ; à exécuter dans Colab (section 10) une fois la clé de licence obtenue |
| 5. Entraînement | nnU-Net v2 puis Auto3DSeg sur les 15 cas | nnU-Net, MONAI Auto3DSeg, MLflow | ⬜ |
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
splits/mmwhs_split.json       Découpage figé entraînement / test
scripts/                      Scripts du pipeline, numérotés par étape
notebooks/colab_runner.ipynb  Notebook qui exécute le pipeline dans Colab
```

La fiche projet (cadrage complet, décisions D1 à D15) est un document local, non publié dans ce dépôt.

## Démarrer

Dans Google Colab : Fichier → Importer un notebook → onglet GitHub → ce dépôt → `notebooks/colab_runner.ipynb`. Activer le GPU, exécuter les cellules dans l'ordre.

## Données

MM-WHS (Zhuang et al.), site officiel : https://zmiclab.github.io/zxh/0/mmwhs/ — usage personnel non commercial, source citée. Les données ne sont jamais versionnées dans ce dépôt.
