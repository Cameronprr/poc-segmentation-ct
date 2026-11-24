# Instructions pour Claude Code — POC segmentation CT cardiaque

## Contexte
Projet personnel d'un product manager medtech qui débute en développement IA. But : un pipeline de segmentation du cœur sur scanners CT publics, de bout en bout, avec un verdict mesurable. Le cadrage complet est dans `docs/fiche_projet.pdf`, les décisions dans `docs/journal_decisions.md`. En cas de doute, ces deux documents font foi.

## Où s'exécute quoi
- Ce dépôt contient le code. Il ne contient JAMAIS de données (voir `.gitignore`).
- Le code s'exécute dans Google Colab (GPU T4). Les données sont dans Google Drive, sous `/content/drive/MyDrive/poc_ct/`.
- Chaque script accepte donc les chemins en paramètres (`--donnees`, `--sortie`) et fonctionne dans Colab.

## Règles
Réponds toujours en français, avec des explications accessibles à un débutant.

1. Une étape de la feuille de route (`docs/historique_etapes.md`) à la fois. Ne pas anticiper les étapes suivantes.
2. Scripts dans `scripts/`, numérotés par étape (`03_preparer.py`, `04_totalsegmentator.py`...), courts, commentés en français, avec un bloc d'explication en tête : ce que fait le script, entrées, sorties.
3. Utiliser les briques existantes plutôt que réécrire : MONAI (chargement, transformations, métriques Dice/HD95), nnU-Net v2, TotalSegmentator.
4. Le découpage `splits/mmwhs_split.json` est figé. Ne jamais le modifier (découpage remplacé une fois, voir D20). Le lot de test ne sert qu'à l'étape 6, une seule fois.
5. Ne jamais entraîner sur des masques produits par TotalSegmentator (licence).
6. Expliquer chaque terme technique la première fois : l'utilisateur apprend.
7. Après chaque étape : mettre à jour l'état dans `docs/historique_etapes.md` (feuille de route détaillée) ; mettre à jour `README.md` (page d'accueil d'une page) seulement si les résultats clés, les limites ou les liens changent ; proposer une ligne pour `docs/journal_decisions.md` si une décision a été prise.

## Agents disponibles
- `data-engineer` : téléchargement, contrôle qualité, conversion des données.
- `ml-engineer` : TotalSegmentator, nnU-Net, Auto3DSeg, suivi MLflow.
- `qa-reviewer` : relit chaque script avant exécution et vérifie les résultats.

## Délégation aux agents (règle permanente)
1. Toute modification du journal (`docs/journal_decisions.md`), d'un fichier figé (tout fichier au statut « figé » ou cité comme figé dans le journal, notamment `docs/protocole_etape6.md`, `docs/specification_controle_plausibilite.md`, `docs/correspondance_etiquettes.md`, `config/seuils_plausibilite.json`, `config/empreintes_totalseg_etape4.json`, `config/empreinte_manifeste_etape6.json`, `splits/mmwhs_split.json`) ou d'une empreinte (y compris les empreintes de référence écrites en constantes dans les scripts) est relue par l'agent `qa-reviewer`, en lecture seule, **avant** de montrer le diff à l'utilisateur.
2. Toute analyse de résultats de modèles est faite par l'agent `ml-engineer`, puis relue par l'agent `qa-reviewer`.
3. Toute manipulation de données, de découpage ou de format passe par l'agent `data-engineer` (la règle 4 reste prioritaire : le découpage figé n'est jamais modifié).
4. Chaque réponse indique quel agent a fait quoi (et ce qui a été fait directement, sans agent).

Ces règles s'adressent à l'agent principal ; un agent n'en délègue pas un autre.
