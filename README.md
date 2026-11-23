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
| 5. Entraînement | nnU-Net v2 puis Auto3DSeg (SegResNet seul, D37) sur les 15 cas | nnU-Net, MONAI Auto3DSeg, MLflow, `scripts/05a_preparer_nnunet.py`, `scripts/05b_essai_nnunet.py`, `scripts/05c_entrainer_nnunet.py`, `scripts/05d_analyser_1019.py`, `scripts/05e_densites_vaisseaux.py`, `scripts/05f_bilan_validation_croisee.py`, `scripts/05g_examiner_cas.py`, `scripts/05h_volumes_verite.py`, `scripts/05i_postraitement_nnunet.py`, `scripts/05j_calibrer_controle.py`, `scripts/controle_plausibilite.py`, `scripts/05k_preparer_auto3dseg.py`, `scripts/05l_essai_segresnet.py`, `scripts/05m_entrainer_segresnet.py`, `scripts/reprise_segresnet.py`, notebook sections 12 à 25 | ✅ nnU-Net : préparation faite, essai de 5 époques fait (L4 : 114,5 s par époque) ; entraînement 250 époques par pli (D25). **Pli 0 terminé** : Dice moyen de validation 0,850 sur 1001, 1019 et 1020 (1001 et 1020 = même patient). Cavités et myocarde entre 0,84 et 0,96 sur les 3 cas, aorte 0,97-0,98 sur 1001 et 1020, mais **échec sur les vaisseaux de 1019** : aorte 0,00 (378 000 voxels dans la vérité, 15 000 prédits), artère pulmonaire 0,46. Cause trouvée (section 16, D27) : **phase d'injection atypique** (aorte moins dense que l'artère pulmonaire). 1019 est gardé dans les données ; cette limite sera inscrite au rapport. Incident de reprise clos (section 17) : 2 époques refaites, aucun défaut. 05c recopie maintenant sa sortie dans Drive et affiche l'heure de Paris. Contrôle de plausibilité (D28) : spécification rédigée (`docs/specification_controle_plausibilite.md`), seuils de densité (R4 à R7) à calibrer avec la section 18 (densités des vaisseaux sur les cas d'entraînement) ; mesure des volumes pour R2 et R3 à prévoir (D29). **Pli 1 terminé** (D30) : Dice moyen de validation 0,919 sur 1005, 1012 et 1015 (1005 et 1012 = même patient). 1005 et 1012 très bons ; **1015 plus faible** sur le VD (0,83) et l'artère pulmonaire (0,65), avec à la fois des oublis et des ajouts : limite VD/AP ou étendue des branches différente ? Phase d'injection classique. **Plis 2, 3 et 4 terminés** (entraînés sans analyse préalable, D30) : Dice moyens de validation 0,908 (pli 2), 0,933 (pli 3) et 0,920 (pli 4). **Validation croisée nnU-Net complète (5 plis × 250 époques).** Bilan des 15 cas de validation exécuté (05f, section 19) : Dice moyen hors pli **0,906 sur les cas**, **0,903 par patient** (13 patients, les paires 1001/1020 et 1005/1012 comptées une fois), médiane 0,924. Examen des cas faibles exécuté (05g, section 20) sur 1002, 1015 et 1018. Classement (D31) : hors domaine = vaisseaux de 1019 ; annotation atypique = OD de 1002 ; définition floue = AP et VD autour de la valve pulmonaire (1015) ; frontière ambiguë = myocarde VG de 1018 ; erreur du modèle = fragment isolé de l'AP de 1018 (à détecter par R8). **Aucun échec majeur ne relève d'une erreur pure du modèle.** Décalages des limites AP / VD (05f, section 19) et volumes de la vérité (05h, section 21) exécutés, contrôles conformes (D32) : **l'AP prédite est en plusieurs morceaux dans 9 cas sur 15** (jusqu'à 7 pour 1008), ce qui fausse les décalages (1008 : −97 mm avec un Dice AP de 0,915) et ferait sonner R8 sur la majorité des cas (avant post-traitement ; après, 3 cas gardent plus d'un morceau de l'AP de plus de 0,5 mL, D33) ; volumes de la vérité très variables (cavité VG de 30 à 150 mL, phase du cycle cardiaque), donc R2 et R3 ne détecteront que les erreurs grossières. **Post-traitement décidé (D33)** : nnU-Net (05i, section 22) retient « garder le plus grand morceau » pour OG, OD, VD et aorte, pas pour le myocarde, la cavité VG ni l'AP (Dice de l'AP 0,8271 → 0,8224 avec ce nettoyage) ; Dice moyen inchangé (0,906). **Appliqué tel quel, sans l'étendre à l'AP** : ses fragments sont confiés à R8. Sur le plus grand morceau, le bas de l'AP varie de −29 à +46 mm, écart-type 21 mm (48 mm tous morceaux ; 05f, partie 5 bis). Calibrage du contrôle de plausibilité (05j, section 23) **exécuté une première fois** avec les seuils de D34 : 1019 en alerte, concordance parfaite avec 05e et 05h, mais **critère Q3 non atteint** (causes de conception). **Révision unique (D35)** : R8 réduite au nombre de morceaux de plus de 0,5 mL, R6 à seuil fixe 0 HU, R3 limitée au rapport AP / aorte, marge de R2 à 50 %, l'alerte R8 sur l'AP de 1018 comptée comme vraie alerte. Section 23 relancée : critère **atteint sous réserve** (1019 en alerte, 1001 et 1020 OK, fragment de 1018 détecté, une fausse alerte consignée : 1002, second morceau de l'AP de 0,65 mL). **Seuils figés le 2026-11-14 (D36)** dans `config/seuils_plausibilite.json` (empreinte SHA-256 dans D36) ; fausses alertes attendues sur des cas nouveaux : environ 2 sur 14 (indicatif). **Auto3DSeg en version réduite (D37)** : SegResNet seul, sur les mêmes 5 plis que nnU-Net ; liste de données et essai court exécutés (section 24, pli 0, 5 époques, L4 : 21 s d'entraînement et 164 s de validation par époque, pic de mémoire 12,2 Go). Autres architectures d'Auto3DSeg reportées en phase 2. **D38** : 300 époques par pli, validation toutes les 10 époques (et à la dernière), évaluation hors pli avec le modèle final (`model_final.pt`, comme nnU-Net), prédictions des 3 cas de validation de chaque pli dans la géométrie d'origine, reprise complète après coupure. **Pli 0 de SegResNet à 300 époques exécuté** (05m, section 25 ; 3,24 h, pic de mémoire GPU 21 Go sur 22,5) : Dice moyen hors pli **0,691** avec le modèle final, contre 0,850 pour nnU-Net ; l'aorte de 1019 échoue aussi (cause probable dans les données, D27) ; le Dice de validation montait encore à la fin : **modèle sous-entraîné**, conservé comme essai. **D40 corrige D38** : une époque n'a pas le même contenu dans les deux outils (nnU-Net 250 époques ≈ 125 000 morceaux d'image vus ; SegResNet 300 époques = 3 600, environ 35 fois moins). **Pli 0 à refaire à temps de calcul égal** : 1 250 époques (réglage par défaut), validation toutes les 50 époques et à la dernière, en repartant de zéro dans `entrainement_segresnet_1250/pli_0` ; 05m et section 25 adaptés, non exécutés. **Pli 0 de SegResNet à 1 250 époques exécuté** (section 25 ; 8,6 h, sans coupure) : Dice moyen hors pli **0,829** avec le modèle final (1001 = 0,852 ; 1019 = 0,762 ; 1020 = 0,873), contre 0,850 pour nnU-Net ; **plateau atteint** (0,827 à l'époque 949, 0,829 à l'époque 1 249), donc l'écart ne vient pas du budget ; écart concentré sur l'AP (0,566) et l'OD (0,859) ; aorte de 1019 partiellement trouvée (0,32 contre 0,00 pour nnU-Net), piste sur un seul cas (D41). **D41 : plis 1 à 4 de SegResNet non entraînés** (environ 34 h et 68 unités pour une comparaison secondaire) ; la comparaison des méthodes se fera à l'étape 6, à armes égales : SegResNet du pli 0 contre nnU-Net du pli 0 seul (mêmes 12 cas, même temps de calcul, sans post-traitement) ; révision de D4. **Étape 5 terminée** : pipeline principal = ensemble des 5 plis nnU-Net, post-traitement D33, contrôle de plausibilité figé (D36) |
| 6. Évaluation | Dice, HD95, volumes sur les 5 cas de test ; trois comparaisons (D41) : pipeline principal, nnU-Net pli 0 contre SegResNet pli 0, TotalSegmentator en référence externe (D23) | MONAI (métriques), agent qa-reviewer, `docs/protocole_etape6.md` | ✅ **Verdict : ACCEPTABLE (indicatif, 5 patients ; D46)**. Phase C faite le 2026-11-23, tentative n° 1, commit `1ab4828` : médianes du Dice de A de 0,881 (OD, marge la plus faible) à 0,982 (aorte), AP 0,896 ; aucun échec silencieux ; 1 fausse alerte sur 5 (R5 sur 1010 ; R5 sur 1004 vraie) ; (b) N0 contre S0 : pas de différence démontrée (P6) ; (c) TotalSegmentator, structures 1 à 5 : écarts de médiane avec A ≤ 0,015, pire cas plus bas (1013, cavité VG 0,863) ; description, aucune règle de comparaison prévue pour (c). Résultats dans `resultats/etape6/` (`resultats_etape6.json`, SHA-256 `b69a24c3…5bec110`). Historique : **protocole figé le 2026-11-16 (D42)** : commit `c78bde4`, SHA-256 `ade98f69…dab5d3` ; verdict sur le pipeline principal (médiane du Dice ≥ 0,85 pour les structures 1 à 6, ≥ 0,80 pour l'AP, aucun échec silencieux), NSD à 2 mm secondaire. **Code écrit et relu (D43)**, non exécuté : `06a` (0 bis, sans GPU, moins d'une minute), `06b` (A, sans GPU, 15 à 30 min), `06c` (B, GPU L4 pour nnU-Net puis SegResNet en deux sessions, puis contrôle sans GPU ; environ 2 h 30 à 3 h et 3 à 4 unités), `06d` (C, sans GPU, 10 à 20 min), sections 26 à 29 du notebook ; 15 déviations connues avant l'ouverture consignées dans D43. **Phase 0 bis arrêtée, puis prédictions refaites (D44)** : les prédictions de TotalSegmentator de l'étape 4 ont été perdues (incident du Drive plein ; seul `mmwhs_1004.nii.gz` reste dans `predictions/totalsegmentator`, conservé tel quel). **Prédictions à refaire** (mêmes version 2.18.0, tâche et réglages, GPU L4, jamais `labelsTs`) dans `predictions/totalsegmentator_d44`, avec relecture de chaque fichier écrit et comparaison voxel par voxel du nouveau 1004 avec l'ancien : `scripts/04c_refaire_totalseg_d44.py`, section 26 bis du notebook, **exécutée le 2026-11-22** (GPU L4) : contrôle CONFORME, 6 fichiers sur 6 écrits, relus et vérifiés dans l'interface web de Drive ; 1004 : 139 voxels différents sur 52 428 800 (0,0003 %), Dice entre ancienne et nouvelle prédiction > 0,9999, reproductible à un bruit numérique près ; 06a, 06c et 06d lisent désormais ce nouveau dossier. **Phase 0 bis terminée le 2026-11-22** : 6 empreintes relevées sur le nouveau dossier, recopiées dans `config/empreintes_totalseg_etape4.json` et au journal (D44). Phase A faite. **Phase B terminée le 2026-11-23** : 20 prédictions conformes à R0, contrôle de plausibilité (A : 2 alertes sur 5, R5 sur 1004 et 1010), manifeste figé, SHA-256 `75a8ce43…e65d`, référence dans `config/empreinte_manifeste_etape6.json` et au journal (complément de D42) ; observations consignées avant l'ouverture (D45). **Phase C terminée le 2026-11-23** (D46). **Suite** : rapport de l'étape 7 |
| 7. Rapport | Comparaison des trois approches, verdict | — | ⬜ prochaine étape, à partir de `resultats/etape6/` et de D46 |

### Vérifications à faire au début de l'étape 5
- **Géométrie image / masque vue par nnU-Net** (relecture QA de l'étape 3, point A3). Le script 03 compare les affines avec MONAI et nibabel, mais nnU-Net lit les fichiers avec SimpleITK. Or l'image est copiée telle quelle, alors que le masque est réenregistré : son en-tête d'origine est recopié, seul le type de données change (uint8). Avant tout entraînement, définir les variables d'environnement `nnUNet_raw`, `nnUNet_preprocessed` et `nnUNet_results` vers des dossiers de Drive, puis lancer `nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity`. Cette commande vérifie, cas par cas, que l'image et son masque ont le même espacement, la même origine, la même orientation et la même taille.
- **Listes de données** : elles sont construites uniquement depuis `imagesTr` et `labelsTr`, jamais depuis `labelsTs` (décision D18).
- **Validation croisée par patient (D20)** : les deux paires `{1001, 1020}` et `{1005, 1012}` (clé `groupes_patients` du découpage) sont en entraînement. Chaque paire doit rester dans la même part de validation croisée (part = *fold* : sous-ensemble mis de côté à tour de rôle pour valider), sinon le modèle serait validé sur un patient qu'il a vu à l'entraînement :
  - **nnU-Net** répartit par défaut les cas au hasard, cas par cas, entre ses 5 parts. Il faut lui fournir un fichier `splits_final.json` construit par patient ;
  - **Auto3DSeg** (MONAI) a le même comportement : il faut fixer les parts (champ `fold` de la liste de données) par patient.

### À faire avant l'étape 6 (fait, D42 à D46)
- **Valeurs du fichier de seuils figé non relues par le code** (D36). Le code utilise ses propres constantes pour quatre valeurs de `config/seuils_plausibilite.json` (`scripts/controle_plausibilite.py` pour R0 ; `MORCEAU_ML` et `morceaux()` de `scripts/05i_postraitement_nnunet.py`, repris par le module de contrôle, pour R8) : la tolérance (0,01 mm) et les étiquettes autorisées (0 à 7) de R0, la taille minimale d'un morceau (0,5 mL) et la connexité (26) de R8. Elles sont identiques aujourd'hui. **Au démarrage de l'étape 6**, avant toute lecture du test : vérifier que ces quatre valeurs du fichier figé sont identiques à celles du code, et **s'arrêter sinon**. Vérifier aussi l'empreinte SHA-256 du fichier (D36).
- **Protocole figé (D42)** : `docs/protocole_etape6.md` fait foi pour toute l'étape 6. **Première opération** : empreintes SHA-256 des 5 prédictions de TotalSegmentator (lecture de ces seuls fichiers, jamais de la vérité), inscrites au journal (phase 0 bis). Le script de l'étape 6 vérifie au démarrage l'empreinte du protocole (`ade98f69…dab5d3`) et celles des prédictions de TotalSegmentator. **D44** : les fichiers de l'étape 4 étant perdus, les 5 prédictions sont d'abord **refaites** à l'identique (section 26 bis, `scripts/04c_refaire_totalseg_d44.py`, exécutée le 2026-11-22 : écart sur 1004 de 0,0003 % des voxels, sans effet pratique) dans `predictions/totalsegmentator_d44`, et le nouveau 1004 est comparé à l'ancien ; la phase 0 bis a relevé les empreintes de **ce nouveau dossier** (faite le 2026-11-22, `config/empreintes_totalseg_etape4.json`). Le protocole figé n'est pas modifié.

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
- **Étape 6 (phase 1), faite (D46)** : Dice, HD95 et volumes sur les 5 cas de test, selon le protocole figé avant l'ouverture (`docs/protocole_etape6.md`) ; verdict ACCEPTABLE, indicatif. Avec un seul cas à 0,45 mm (1014), aucune conclusion propre à cette famille n'est possible (D20).
- **Phase 2** : résultats rapportés **par sous-groupe**, avec assez de cas pour conclure ; erreur sur les **mesures dérivées** utiles au clinicien, comme la masse du VG, en plus du Dice (D31) ; comparaison à la variabilité entre annotateurs (D31).

### 4. Se protéger en service : ne jamais rendre un résultat faux en silence

**Ce que le POC a fait**
- **Post-traitement** de nnU-Net (ne garder que le plus grand morceau) déterminé sur les 15 prédictions hors pli (05i, section 22) et appliqué tel que retenu : OG, OD, VD et aorte. Pas sur l'AP, qui peut légitimement être en plusieurs morceaux ; ses fragments sont confiés au contrôle de plausibilité (D33).
- **Contrôle de plausibilité en sortie** (D28) : présence, volume, rapports de volumes, densités et nombre de morceaux de chaque structure, règles R0 à R8 (D29 ; spécification dans `docs/specification_controle_plausibilite.md`). Mesures de calibrage : volumes et rapports mesurés sur la vérité (05h, section 21, D32) ; densités mesurées par 05e (section 18 : 1019 seul cas à phase d'injection inversée ; résultats dans la spécification). 1019 est exclu du calibrage, et chaque règle est vérifiée en laissant chaque patient de côté à tour de rôle, pour éviter un raisonnement circulaire (D28). Calibrage sur la vérité, R8 comprise (D34), puis application aux prédictions post-traitées (D32, D33) : exécuté une fois (05j, section 23), puis révisé une seule fois (D35 : R3 réduite à AP / aorte, R6 fixée à 0 HU, R8 sans critère de part, marge de R2 à 50 %) ; seuils figés le 2026-11-14, avant l'étape 6 (D36 : `config/seuils_plausibilite.json`, empreinte SHA-256 inscrite au journal). Fausses alertes attendues : environ 2 sur 14, estimation indicative ; sensibilité non mesurable avec un seul échec connu.
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
