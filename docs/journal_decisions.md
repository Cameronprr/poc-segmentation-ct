# Journal des décisions

Les décisions D1 à D15 sont détaillées dans la fiche projet, document local non publié. Nouvelles décisions ci-dessous.

| N° | Date | Décision | Raison |
|---|---|---|---|
| D16 | 2026-11-08 | Phase 1 sur MM-WHS (20 cas annotés) : 15 en entraînement / validation croisée, 5 en test sous clé (graine 42, `splits/mmwhs_split.json`) | Démarrer sans attendre l'accès au jeu ImageCAS ; 20 cas suffisent pour valider le pipeline de bout en bout |
| D17 | 2026-11-08 | Dépôt GitHub = code uniquement ; données dans Google Drive ; exécution dans Colab | Licences des données, poids des fichiers, GPU disponible seulement dans Colab |
| D18 | 2026-11-08 | Étape 3 : les masques de test sont rangés dans `Dataset101_MMWHS/labelsTs` mais ne sont lus qu'à l'étape 6. À l'étape 5, les listes de données (nnU-Net, Auto3DSeg) sont construites uniquement depuis `imagesTr`/`labelsTr` | Ranger n'est pas utiliser ; nnU-Net ignore `labelsTs`, mais Auto3DSeg pourrait les inclure par une recherche de fichiers trop large |
| D19 | 2026-11-08 | Étape 4 : TotalSegmentator produit ses prédictions sur les 5 cas de test, mais le Dice n'est calculé qu'à l'étape 6 | Une prédiction n'apprend rien du test ; calculer le score plus tôt reviendrait à regarder le résultat avant le verdict final |
