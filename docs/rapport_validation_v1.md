# Rapport technique de validation du POC — segmentation cardiaque sur CT (version 1)

> **Statut** : version 1, rédigée à l'étape 7, après la clôture de l'étape 6 (D46). Projet personnel, données publiques uniquement.
> **Règle de rédaction** : chaque chiffre vient d'un fichier du dépôt, indiqué entre crochets : `[D46]` renvoie à une décision du journal (`docs/journal_decisions.md`) ; `[resultats/etape6/…]` à un fichier de résultats ; `[protocole § x]` à `docs/protocole_etape6.md` (figé). Rien n'a été recalculé sur le test pour ce rapport.
> **Plan** inspiré de la liste CLAIM 2024 (liste de contrôle de ce qu'un article d'IA en imagerie médicale doit rapporter) [réf. 3], mise à jour de CLAIM 2020 [réf. 8] et des recommandations Metrics Reloaded [réf. 2] (guide du choix des métriques de validation en analyse d'images). Ce rapport ne prétend pas remplir toute la liste CLAIM : c'est un POC sur 20 cas.

## 1. Résumé

**Objectif.** Construire, de bout en bout, un pipeline de segmentation des cavités cardiaques et des gros vaisseaux sur scanner (CT), et l'évaluer avec un verdict mesurable, fixé avant de regarder les résultats du test.

**Méthode.** 20 scanners annotés de MM-WHS [réf. 1], découpés par patient : 15 pour l'entraînement et la validation croisée (chaque modèle est validé sur des cas qu'il n'a pas vus), 5 pour le test, gardés sous clé jusqu'à l'étape 6 [D16, D20]. Pipeline principal (« A ») : nnU-Net v2 [réf. 4], ensemble des 5 modèles de la validation croisée, post-traitement, et un contrôle de plausibilité (R0 à R8) qui signale les résultats suspects sans connaître la vérité [D33, D36]. Deux comparaisons secondaires : nnU-Net pli 0 contre SegResNet pli 0 [réf. 6, 7], à temps de calcul égal [D41] ; TotalSegmentator [réf. 5] comme référence externe [D23]. Critères du verdict et protocole figés avant l'ouverture du test [D42] ; le test n'a été ouvert qu'une fois [D46].

**Verdict : ACCEPTABLE, à titre indicatif** [D46]. Les 7 médianes du Dice (recouvrement entre prédiction et vérité, de 0 à 1) dépassent leur seuil (0,85, et 0,80 pour l'artère pulmonaire) ; la marge la plus faible est celle de l'oreillette droite (+0,031). Aucun échec silencieux. Le contrôle de plausibilité a émis 2 alertes sur 5 cas : 1 vraie, 1 fausse.

**Limites principales** [D42, D46] : 5 patients de test seulement ; seuils fixés après la validation croisée ; un seul jeu de données, une seule source ; sensibilité du contrôle de plausibilité non mesurable ; comparaisons secondaires à un seul modèle par méthode. Le verdict dit que, sur ces 5 patients, le pipeline remplit les critères fixés d'avance ; il ne dit pas que le modèle est fiable en général.

## 2. Objectif et question posée

La phase 1 du projet vise une **chaîne de mesure fiable**, de bout en bout, pas à battre l'état de l'art ni TotalSegmentator [protocole § 4 (c) ; D41]. Trois questions distinctes, fixées avant l'ouverture du test [protocole § 4, D41, D42] :

- **(a) Pipeline principal** : le pipeline livré (A) est-il acceptable selon les critères C1 et C2, et le contrôle de plausibilité sonne-t-il à bon escient ? C'est la seule question qui porte le verdict.
- **(b) Comparaison de méthodes** : nnU-Net pli 0 (N0) et SegResNet pli 0 (S0), entraînés sur les mêmes 12 cas et avec le même temps de calcul, diffèrent-ils ? Question descriptive, hors verdict.
- **(c) Référence externe** : où se place TotalSegmentator (T), outil public déjà entraîné, face à A, sur les 5 structures retenues (même définition pour le myocarde VG, l'OG et la cavité VG ; définitions comparables, avec prudence, pour l'OD et le VD) [D23] ? Question descriptive, hors verdict.

Structures segmentées (étiquettes de MM-WHS) [docs/correspondance_etiquettes.md, protocole § 4] : 1 myocarde du ventricule gauche (VG), 2 oreillette gauche (OG), 3 cavité du VG, 4 oreillette droite (OD), 5 ventricule droit (VD), 6 aorte, 7 artère pulmonaire (AP).

## 3. Données

- **Source** : MM-WHS (*Multi-Modality Whole Heart Segmentation*), partie CT, 20 scanners annotés par des experts [réf. 1 ; README, section « Données »]. Usage personnel non commercial, source citée. Les données ne sont jamais dans ce dépôt, seulement dans Google Drive [D17].
- **Deux familles d'examens**, repérées dans l'en-tête des fichiers : espacement entre coupes de 0,45 mm (5 cas : 1001, 1003, 1014, 1016, 1020, probablement un autre scanner) et de 0,625 mm (15 cas) [D20].
- **Patients en double** : 1001/1020 et 1005/1012 sont très probablement le même patient à deux phases cardiaques (géométrie identique au 0,01 mm, corrélation des miniatures de 0,974 et 0,962, examen visuel) [D20].
- **Découpage par patient, stratifié par épaisseur de coupe** (tiré séparément dans chaque famille d'examens) [D20] : test = 1004, 1010, 1011, 1013 (0,625 mm) et 1014 (0,45 mm), 5 patients à un seul examen, tirés avec la graine 42 (nombre qui fixe le tirage au hasard, pour le reproduire) ; entraînement = les 15 autres, paires comprises, chaque paire restant dans la même part de validation croisée [D24]. Ce découpage a remplacé une fois celui de D16 (déviation volontaire et unique), puis a été figé : `splits/mmwhs_split.json`, empreinte (64 caractères calculés sur tous les octets du fichier ; un seul octet changé la change) **SHA-256 `25881b214823758ddb708b6b46616c2422671c872035683852b1f045f4c4a05c`**, vérifiée au démarrage de chaque phase de l'étape 6 [D43 ; resultats/etape6/tracabilite_06d.json pour la phase C]. Aucune paire test/entraînement au-dessus d'une corrélation de 0,95 (maximum 0,638) [README, étape 3].
- **Masques de test** (`labelsTs`) : rangés à l'étape 3 mais jamais lus avant la phase C de l'étape 6 [D18, D19, D43] ; leurs empreintes sont relevées à l'ouverture [D46].
- **Cas particuliers de l'entraînement**, gardés sans correction ni retrait :
  - **1019, phase d'injection inversée** : aorte à 253 HU et AP à 416 HU, contre 562 et 192 HU sur 1001 (HU, unités Hounsfield : échelle de densité du scanner). Seul cas de ce type sur 15 : le modèle ne peut pas apprendre cette situation [D27] ;
  - **1002, annotation atypique de l'OD** : zone sanguine de l'OD non annotée ; le modèle prédit une OD 61 % plus grande que l'annotation, dans le même sens que TotalSegmentator (OD +46 %, VD −18 %) [D31, D23].

## 4. Méthodes

Cette section décrit les quatre modèles évalués à l'étape 6 et leurs réglages. Repères utilisés dans tout le rapport [docs/protocole_etape6.md § 3.3] :

- **A** : pipeline principal, ensemble des 5 plis nnU-Net, post-traité, suivi du contrôle de plausibilité ; c'est le seul qui porte le verdict ;
- **N0** : nnU-Net, pli 0 seul, sans post-traitement ;
- **S0** : SegResNet (MONAI Auto3DSeg), pli 0 seul, sans post-traitement ;
- **T** : TotalSegmentator, référence externe.

Numérotation des structures [docs/protocole_etape6.md § 3.2] : 1 myocarde VG (paroi musculaire du ventricule gauche), 2 OG (oreillette gauche), 3 cavité VG, 4 OD (oreillette droite), 5 VD (ventricule droit), 6 aorte ascendante, 7 AP (artère pulmonaire, tronc).

### 4.1 TotalSegmentator (référence externe)

TotalSegmentator est un outil de segmentation déjà entraîné par ses auteurs sur un autre jeu de données. Nous ne l'entraînons pas : nous l'appliquons tel quel, comme point de comparaison extérieur.

- **Version et tâche** : TotalSegmentator 2.18.0, tâche `heartchambers_highres` [docs/correspondance_etiquettes.md ; D44]. Cette tâche demande une licence. Les statistiques d'usage, qui transmettraient le numéro de licence, sont désactivées [D21].
- **Correspondance des étiquettes** : l'étiquette *k* de TotalSegmentator correspond à notre étiquette *k*, pour *k* = 1 à 7, sans renumérotation. Les numéros coïncident, mais pas toujours les définitions [docs/correspondance_etiquettes.md].
- **Prédictions** : à l'étape 4, les 5 cas de test ont été prédits sur GPU T4, sans calcul de score [README ; D19]. Ces fichiers ont ensuite été perdus, probablement lors de l'incident du Drive plein (cause non démontrée), sauf `mmwhs_1004.nii.gz`. Les 5 prédictions ont donc été **refaites** avant l'ouverture du test : même version, même tâche, mêmes réglages, sur GPU L4, en ne lisant que les scanners [D44]. Il s'agit d'une déviation connue avant l'ouverture.
- **Reproductibilité** : l'ancienne et la nouvelle prédiction de 1004 diffèrent sur **139 voxels sur 52 428 800** (0,0003 %), répartis sur toutes les étiquettes. Le Dice entre les deux prédictions dépasse 0,9999 [D44]. Ce chiffre compare deux prédictions entre elles, pas une prédiction à la vérité. Cause probable : changement de matériel (L4 au lieu de T4) ou des bibliothèques de Colab, non démontrée. Limite : on ne peut pas prouver que les 4 autres prédictions refaites (1010, 1011, 1013, 1014) sont identiques à celles de l'étape 4 [D44].
- **Structures comparées : 1 à 5 seulement** [D23]. L'examen de l'étape 4 bis, fait sur deux cas d'entraînement (1003 et 1002), donc sans toucher au test, a montré [D23] :
  - même définition pour le myocarde VG, l'OG et la cavité VG (volumes à moins de 7 %, niveaux à 2 mm) ;
  - OD et VD comparables, mais avec des écarts variables (1002 : OD +46 %, VD −18 %) et des frontières moins nettes, d'où une **mention de prudence** obligatoire ;
  - aorte : TotalSegmentator inclut l'aorte descendante (volume doublé, 66 à 72 mm plus bas) ;
  - AP : TotalSegmentator inclut une partie des branches (+8 % et +29 %).

  Comparer l'aorte et l'AP pénaliserait TotalSegmentator pour une différence de définition, pas pour sa qualité : ces deux structures sont exclues pour T.
- **Aucun entraînement sur les masques de TotalSegmentator** (CLAUDE.md, règle 5 ; licence). Ses prédictions ne servent qu'à la comparaison (c) et ne sont pas post-traitées [docs/protocole_etape6.md § 3.3].

### 4.2 nnU-Net v2 (version 2.8.1)

nnU-Net est un outil qui règle lui-même son réseau (un U-Net, réseau de neurones en forme de « U » courant en imagerie médicale) d'après les données. Nous avons gardé ses réglages par défaut, sauf le nombre d'époques et le nombre de configurations, contraints par le budget de calcul.

- **Configuration** : `3d_fullres` seule, entraîneur `nnUNetTrainer_250epochs_reprise10`, plans `nnUNetPlans` [D24 ; docs/protocole_etape6.md § 3.3]. Le réglage par défaut (3 configurations × 5 plis × 1 000 époques) n'était pas réalisable sur Colab ; `3d_fullres` est la configuration recommandée par les auteurs quand on n'en entraîne qu'une [D24].
- **Validation croisée en 5 plis, par patient** [D24]. Un *pli* est un sous-ensemble de cas mis de côté à tour de rôle : le modèle du pli *i* est entraîné sur 12 cas et validé sur les 3 autres. Les plis sont construits par `scripts/05a_preparer_nnunet.py` (graine 42, 5 plis de 3 cas). Les deux paires de cas qui sont probablement le même patient (1001/1020 et 1005/1012) restent dans le même pli. Les plis sont **stratifiés par épaisseur de coupe** : les 3 patients d'entraînement à 0,45 mm sont dans 3 plis différents.
- **250 époques par pli** [D25]. Une *époque* est un passage d'entraînement d'une longueur fixée par l'outil. Sur GPU L4 : 114,5 s par époque, soit environ 8 h par pli. Sur T4 (Colab gratuit), 420 s par époque, soit environ 29 h, ce qui était impossible avec des sessions de quelques heures [D25]. 250 époques au lieu de 1 000 : compromis entre durée et qualité. Poids utilisés : `checkpoint_final`, c'est-à-dire ceux de la dernière époque [docs/protocole_etape6.md § 3.3].
- **Ensemble des 5 plis (A)** : un *ensemble* combine les prédictions des 5 modèles. Le protocole parlait d'une moyenne des probabilités. nnU-Net 2.8.1 fait par défaut la moyenne des **logits**, c'est-à-dire des scores bruts avant leur conversion en probabilités. Ce réglage par défaut a été gardé (règle P7) et consigné comme **déviation 1 de D43**, acceptée avant l'ouverture. L'écart attendu sur le masque final est marginal [D43].
- **Post-traitement « plus grand morceau »** [D33]. Ce nettoyage ne garde que le plus gros bloc de voxels connectés d'une structure et efface les fragments isolés. nnU-Net l'a évalué structure par structure sur les prédictions *hors pli* (chaque cas prédit par le modèle qui ne l'a pas vu à l'entraînement). Il l'a retenu pour **OG, OD, VD et aorte**, pas pour le myocarde VG, la cavité VG ni l'AP. Sur l'AP, il faisait baisser le Dice de 0,8271 à 0,8224 [D33]. Décision : appliquer le post-traitement tel que nnU-Net l'a retenu, sans l'étendre à l'AP. Les fragments de l'AP sont confiés à la règle R8 du contrôle de plausibilité [D33].
- **Pipeline principal A** = ensemble des 5 plis, post-traitement D33, puis contrôle de plausibilité figé (D36) [D41 ; D42].
- **N0** = modèle du **pli 0 seul** (`checkpoint_final`), **sans post-traitement**. Il sert uniquement à la comparaison à armes égales avec SegResNet [D41].

### 4.3 SegResNet (MONAI Auto3DSeg, MONAI 1.6.1)

Auto3DSeg est la chaîne automatique de MONAI. Par défaut, elle entraîne plusieurs architectures sur 5 plis puis les combine, ce qui dépassait le budget. Elle a été réduite à **une seule architecture, SegResNet** (réseau convolutif), sur **exactement les mêmes plis que nnU-Net** [D37]. Versions épinglées : MONAI 1.6.1 et modèles d'algorithmes de l'archive `21ed8e5`, SHA-256 `b4f24826bb974d4627d28c7dfe3b64bb8bcae455ec7a7edbeeaadd3168eebfc2` vérifiée à chaque lancement [D37]. La liste de données est construite depuis les seuls cas d'entraînement et ne contient aucun cas de test [D37 ; D18].

- **Pli 0 seulement, 1 250 époques** [D40 ; D41]. Un premier entraînement à 300 époques (D38) est resté sous-entraîné : Dice moyen hors pli 0,691 [D40]. Il est conservé comme essai, sans servir à la comparaison. Le pli 0 a été refait à **temps de calcul égal à nnU-Net**, avec le réglage par défaut de 1 250 époques, une validation toutes les 50 époques, en repartant de zéro [D40]. Durée : 8,6 h sur L4, sans coupure [D41]. Les plis 1 à 4 n'ont pas été entraînés : environ 34 h et 68 unités de calcul pour une comparaison secondaire [D41].
- **Modèle évalué** : `model_final.pt`, poids de la dernière époque, comme `checkpoint_final` pour nnU-Net. Le modèle « meilleur Dice de validation » (`model.pt`) n'est pas utilisé : il serait choisi sur les 3 cas qui servent ensuite à le juger [D38]. L'arrêt anticipé est désactivé [D39]. Ce modèle est appelé **S0**, sans post-traitement [D41].
- **Limite de la comparaison** : à temps de calcul égal, SegResNet voit environ 15 000 morceaux d'image, environ 8 fois moins que nnU-Net, mais des morceaux plus grands (224 × 224 × 144 voxels) [D40].
- **Correctif de `segmenter.py`** [D39]. MONAI ne reprend ni l'état de l'optimiseur ni le calendrier du taux d'apprentissage après une coupure de session. Pour permettre une reprise complète, le fichier `segmenter.py` du modèle a été modifié : 5 changements, empreintes SHA-256 contrôlées avant et après (`0477831…1d43` → `8ee7331…7c81`). Le correctif touche **uniquement** à l'enregistrement, à la reprise et aux écritures atomiques, sans modifier le calcul de l'apprentissage. L'agent qa-reviewer l'a vérifié ligne à ligne [D39]. Limite : l'état des tirages au hasard n'est pas repris, et l'entraînement n'était de toute façon pas reproductible à l'identique [D39].
- **Repli mémoire pendant la prédiction du test** [D45 a)]. En phase B, SegResNet a affiché des avertissements de manque de mémoire GPU : jusqu'à 6 Go demandés pour environ 2 Go libres. Chaque cas est pourtant allé au bout, et la règle R0 est conforme pour les 20 prédictions. Mécanisme vérifié dans le code de MONAI 1.6.1 : quand la mémoire manque, `SlidingWindowInfererAdapt` assemble les fenêtres par blocs, puis sur le processeur. Les prédictions des fenêtres restent calculées sur le GPU : c'est le même calcul. Seul l'ordre des additions peut changer, avec un écart d'arrondi possible vers le 7e chiffre, sans effet attendu sur l'étiquette retenue. Limites : le lien entre l'avertissement et le repli est une déduction ; on n'a pas vérifié lequel des replis a eu lieu [D45 a)].

### 4.4 Réglages d'inférence et leurs différences (N0 contre S0)

*L'inférence* est l'étape où le modèle entraîné produit un masque sur un scanner nouveau. Deux réglages comptent ici :

- la *fenêtre glissante* : le réseau ne voit qu'un bloc du scanner à la fois, puis on fait glisser ce bloc avec un recouvrement et on combine les résultats avec un poids plus fort au centre (pondération « gaussienne ») ;
- l'*augmentation au moment de la prédiction* (TTA, ou « miroir ») : on prédit aussi les images retournées et on fait la moyenne [scripts/06c_predire_test.py, en-tête].

Le protocole (P7) impose les **réglages par défaut de chaque outil**, identiques à ceux de l'évaluation hors pli de l'étape 5, relevés et écrits dans le manifeste avant l'ouverture [docs/protocole_etape6.md § 13, P7]. Le *manifeste* est le fichier qui fige les empreintes de toutes les prédictions et de tous les contrôles avant l'ouverture. Il est stocké dans Drive. Le dépôt ne contient que son empreinte (SHA-256 `75a8ce43…e65d`) [config/empreinte_manifeste_etape6.json ; D42]. Les valeurs effectivement relevées par 06c en phase B ont été affichées à la fin de la phase B, sous le titre « LIMITE de la comparaison (b) (P7, § 12) : réglages d'inférence différents entre N0 et S0, NON alignés ». Elles sont écrites dans le manifeste (sur Drive) et ont été recopiées au journal depuis la sortie de 06c, conservée dans Drive [D47].

| Réglage | N0 (nnU-Net, pli 0) | S0 (SegResNet, pli 0) | Source |
|---|---|---|---|
| Augmentation au moment de la prédiction (miroir) | oui | non | [D47] |
| Passages par fenêtre | 8 | 1 | [D47] |
| Taille de fenêtre (voxels) | 80 × 192 × 160 | 224 × 224 × 144 | [D47] ; pour S0, même valeur qu'à l'entraînement [D38] |
| Recouvrement des fenêtres | 0,5 | 0,625 | [D47] |
| Pondération des fenêtres | gaussienne | gaussienne (réglage par défaut de `segmenter.py`) | [scripts/06c_predire_test.py] |
| Assemblage des fenêtres | réglage par défaut de nnU-Net | `SlidingWindowInfererAdapt`, avec repli par blocs ou sur processeur en cas de manque de mémoire | [D45 a)] |
| Nombre de modèles | 1 (pli 0) | 1 (pli 0) | [D41] |

**Pourquoi ces réglages ne sont pas alignés** : la règle P7 demande de garder les réglages par défaut de chaque outil, ceux qui ont servi à l'étape 5. Si l'un utilise l'augmentation et l'autre non, c'est signalé comme une **limite de la comparaison (b)**, sans aligner après coup [docs/protocole_etape6.md P7 ; § 12]. Aligner les réglages après avoir vu les chiffres reviendrait à choisir le réglage qui arrange le résultat. Avec les 4 réglages imposés par l'absence de vérité (déviation 2 de D43), le chemin de prédiction de S0 est le même que celui de l'évaluation de l'étape 5 [D43].

---

## 5. Contrôle de plausibilité (R0 à R8)

### 5.1 Principe

Pendant la validation croisée, le Dice moyen de 0,850 du pli 0 cachait un **échec complet** sur l'aorte du cas 1019 [D26 ; D28]. D28 en tire une règle : **le pipeline ne rend jamais en silence un résultat hors du domaine validé**. Le *domaine validé* est l'ensemble des examens qui ressemblent à ceux sur lesquels le modèle a été vérifié [D28].

Le **contrôle de plausibilité** s'exécute après la segmentation. Il ne lit **aucune vérité terrain**, seulement l'image et le masque prédit. Il répond à la question : « ce masque ressemble-t-il à un cœur segmenté correctement, sur un examen semblable à ceux qu'on connaît ? » Si une règle se déclenche, le cas reçoit le statut **ALERTE**. Le contrôle ne corrige rien et ne remplace pas une relecture humaine [docs/specification_controle_plausibilite.md § 1-3].

Il existe **deux types d'alertes** [D29, Q8 ; docs/specification_controle_plausibilite.md § 4] :

- **« erreur probable »** (R0 à R5, R8) : le masque est probablement faux ;
- **« hors domaine »** (R6, R7) : le masque peut être juste, mais l'examen ne ressemble pas aux examens de validation. Ces alertes sont présentées comme des **limites du modèle**, séparément des erreurs.

Règles de la version 1 [docs/specification_controle_plausibilite.md § 5 ; seuils : config/seuils_plausibilite.json]. Le HU (unité Hounsfield) est l'échelle des densités du scanner, où l'eau vaut 0. Les densités sont mesurées au cœur des structures, à plus de 3 mm du bord. Un *morceau* est un bloc de voxels qui se touchent (connexité 26 : par une face, une arête ou un coin).

| Règle | Ce qu'elle mesure | Seuil figé | Catégorie |
|---|---|---|---|
| R0 | Cohérence technique : même forme et même affine (position, taille et orientation des voxels) que l'image (tolérance 0,01 mm), étiquettes entre 0 et 7 | fixé | erreur probable (bloquante) |
| R1 | Présence de chaque structure 1 à 7 | volume > 1 mL | erreur probable |
| R2 | Volume de chaque structure dans la plage observée | [min × 0,5 ; max × 1,5] sur la vérité, soit une marge de 50 % ; par exemple aorte 14,459 à 177,179 mL | erreur probable |
| R3 | Rapport des volumes AP / aorte | 0,4705 à 2,7905 | erreur probable |
| R4 | Écart de densité entre l'aorte et la cavité VG (le sang de l'aorte sort du VG) | ≤ 207 HU | erreur probable |
| R5 | Écart de densité entre l'AP et le VD (le sang de l'AP sort du VD) | ≤ 176 HU | erreur probable |
| R6 | Aorte au moins aussi dense que l'AP (phase d'injection habituelle) | aorte − AP ≥ 0 HU | hors domaine |
| R7 | Présence du produit de contraste (densité de la cavité VG) | ≥ 245 HU | hors domaine |
| R8 | Nombre de morceaux de plus de 0,5 mL par structure | au plus 1 pour chaque structure 1 à 7 | erreur probable |

R9 (positions relatives et contacts entre structures) est **reportée à une version 2**, en phase 2 du projet [D29, Q1]. Ces mesures sont plus délicates, et R2, R3 et R4 suffisaient pour 1019.

### 5.2 Calibration

*Calibrer* une règle, c'est fixer son seuil à partir de données connues.

- **Sur la vérité terrain des cas d'entraînement** : les seuils de R2 à R5, R7 et R8 sont tirés des annotations des cas d'entraînement, pas des prédictions du modèle. La vérité décrit l'anatomie et les densités plausibles, indépendamment du modèle [D29, Q9 ; D34 ; docs/specification_controle_plausibilite.md § 6.2]. R8 est calibrée sur la vérité elle aussi, car le nombre de morceaux qu'une structure peut avoir est une question anatomique [D34].
- **1019 exclu** : les règles ont été imaginées après avoir vu l'échec de 1019. S'en servir pour fixer les seuils, puis pour « prouver » que le contrôle marche, serait un raisonnement circulaire [D28 ; docs/specification_controle_plausibilite.md § 6.4]. Le calibrage porte donc sur **14 cas, soit 12 patients** [config/seuils_plausibilite.json].
- **Marges fixées avant de voir les mesures** : 30 % pour les volumes et les rapports, 50 HU pour les densités [D29, Q2].
- **Validation en laissant chaque patient de côté** (*leave-one-patient-out*) : pour chaque patient, les seuils sont recalculés sans lui (sans les deux cas pour une paire), puis appliqués à ce patient. On mesure ainsi combien de cas réussis seraient signalés à tort par des seuils qui ne les ont pas vus [D28 ; docs/specification_controle_plausibilite.md § 6.4].
- **Application aux prédictions post-traitées** : le contrôle est vérifié sur les 15 prédictions hors pli **après** le post-traitement D33, car c'est le masque livré [D32 ; D33].
- **Critère de validation** : 1019 en ALERTE ; 1001 et 1020 sans alerte ; au plus une fausse alerte, expliquée, sur les 14 autres cas [D29, Q3 ; docs/specification_controle_plausibilite.md § 7].

### 5.3 Révision unique (D35)

La première vérification n'a pas atteint le critère des fausses alertes (Q3). Une **révision unique** a été décidée **après** cette vérification, et assumée comme telle [D35] :

- **R3** limitée au rapport **AP / aorte**. Les 4 rapports divisés par la cavité VG sont supprimés, car le volume de la cavité VG suit la phase du cycle cardiaque ;
- **R6** à seuil **fixe de 0 HU** (alerte si aorte − AP < 0). C'est la définition « inversé » écrite avant les mesures. L'ancien seuil calibré reposait sur un seul cas (1006) ;
- **R8** réduite au nombre de morceaux de plus de 0,5 mL. Le critère « part du plus gros morceau » est supprimé : sans marge, il signalait des fragments de quelques centièmes de mL ;
- **marge de R2** portée de 30 % à **50 %**. Avec 13 patients, 30 % produisait des fausses alertes en validation par patient ;
- une alerte qui correspond à une erreur du modèle **classée en D31 avant tout calibrage** compte comme vraie : le fragment de l'AP de 1018.

Il n'y aura plus de révision ensuite. **Biais assumé** : les seuils révisés ne sont plus indépendants des cas d'entraînement. Seul le test mesure les fausses alertes sans ce biais [D35].

### 5.4 Gel (D36)

Après la révision, le critère est **atteint sous réserve** [D36] :

- 1019 en ALERTE (R2 aorte, R3 AP / aorte, R8 AP ; R6 hors domaine) ;
- 1001 et 1020 sans alerte ;
- fragment de l'AP de 1018 détecté, ce qui est une vraie alerte ;
- **une fausse alerte consignée** : 1002, R8 sur l'AP, second morceau de 0,65 mL, à peine au-dessus du seuil de 0,5 mL ;
- aucune alerte sur la vérité en validation par patient.

Les seuils sont **figés le 2026-11-14** dans `config/seuils_plausibilite.json`. Les 45 valeurs de seuil et de marge sont identiques à celles calculées par 05j. Empreinte SHA-256 : `c1b247070765d193a46e90b538cc655787dbb65f8fdd13f113a39efd3ef90672` [D36], identique à l'empreinte du fichier présent dans le dépôt (vérifiée avec sha256sum). Aucune modification n'est permise après le gel : un seuil mal adapté au test est noté comme limite, pas corrigé [D36 ; docs/specification_controle_plausibilite.md § 7.2].

- **Fausses alertes attendues sur des cas nouveaux : environ 2 sur 14**, estimation indicative tirée de la validation par patient sur les prédictions (1001 : R2 sur l'AP ; 1002 : R8 sur l'AP) [D36].
- **Sensibilité non mesurable** : la *sensibilité* est la part des vrais échecs que le contrôle attrape. Avec un seul échec connu (1019), elle ne peut pas être mesurée. Signaler 1019 est une vérification de cohérence, pas une preuve d'efficacité [D36].

### 5.5 Classement des alertes à l'étape 6

Le contrôle ne lit pas la vérité : il a donc été exécuté **avant l'ouverture** du test, et ses résultats ont été figés dans le manifeste [docs/protocole_etape6.md § 7.1]. Après l'ouverture, chaque alerte de A est classée vraie ou fausse selon une règle fixée d'avance [docs/protocole_etape6.md § 7.3]. Une alerte « erreur probable » est **vraie** si la structure en cause est réellement mal segmentée. Le repère est un Dice < 0,85, le seuil « Dice faible » de D30 :

| Règle | Vraie si… |
|---|---|
| R0 | toujours vraie |
| R1 | la vérité de la structure fait plus de 1 mL |
| R2 | Dice < 0,85 **ou** \|écart de volume\| > 30 % |
| R3 | Dice(AP) < 0,85 **ou** Dice(aorte) < 0,85 |
| R4 | Dice(aorte) < 0,85 **ou** Dice(cavité VG) < 0,85 |
| R5 | Dice(AP) < 0,85 **ou** Dice(VD) < 0,85 |
| R8 | un morceau de plus de 0,5 mL recouvre moins de la moitié de ses voxels avec la vérité (morceau égaré), **ou** Dice < 0,85 |
| R6, R7 | la même mesure, faite sur la vérité, déclenche aussi la règle (l'examen est réellement hors domaine) |

- **Règle « ou »** : l'alerte est vraie dès qu'une condition définie est vraie [D43, déviation 3].
- **Catégorie « non classable »** [D43, déviation 3] : une alerte est non classable quand la grandeur à comparer n'est pas définie, par exemple une vérité vide ou une densité impossible à mesurer sur la vérité. Elle n'est ni vraie ni fausse, et elle est rapportée à part, hors de l'estimation des fausses alertes.
- **Estimation officielle des fausses alertes** : nombre de cas, sur 5, ayant au moins une alerte « erreur probable » classée fausse, écrit « x sur 5 ». Elle porte sur A seulement [docs/protocole_etape6.md § 7.3].
- **Échec silencieux** : un couple (cas, structure) avec un Dice < 0,50 **sans aucune alerte** sur ce cas. C'est exactement ce que D28 veut éviter [docs/protocole_etape6.md § 7.3].
- **Statut des alertes selon le modèle** : sur N0 et S0, les alertes sont **informatives**, hors estimation, car les seuils ont été vérifiés sur des prédictions post-traitées (P4). Sur **T**, le contrôle est **restreint** à R0 et R7, et à R1, R2 et R8 sur les structures 1 à 5. Les règles qui portent sur l'aorte ou l'AP sont écartées à la lecture, car leurs définitions diffèrent [docs/protocole_etape6.md § 4 (c), § 7.1 ; D29, Q7].

---


## 6. Protocole de l'étape 6

**Gel avant l'ouverture.** Le protocole (`docs/protocole_etape6.md`) a été figé le 2026-11-16, avant tout code et toute lecture du test : commit `c78bde4`, **SHA-256 `ade98f69f6587035fc17f5ab96bc208bbdcfab50812e5fa3317f8b4958dab5d3`** [D42]. Un écart découvert ensuite n'a jamais modifié le protocole : il a été consigné comme déviation (section 8).

**Critères du verdict, fixés avant l'ouverture** [D42, protocole § 8.4], sur le pipeline principal A seulement :
- **C1, performance** : médiane du Dice (valeurs non arrondies) ≥ 0,85 pour les structures 1 à 6 et ≥ 0,80 pour l'AP. Le Dice mesure le recouvrement entre la prédiction et la vérité (0 = aucun, 1 = parfait). Seuil de l'AP relevé de 0,75 (proposé) à 0,80 par l'utilisateur ;
- **C2, sécurité** : aucun **échec silencieux**, c'est-à-dire aucun Dice < 0,50 sur un cas où le contrôle de plausibilité n'a émis aucune alerte ;
- HD95 (distance au pire endroit du contour, sans les 5 % de points les plus éloignés), volumes et NSD à 2 mm (part des contours à moins de 2 mm l'un de l'autre) sont décrits sans seuil ; la NSD est une métrique secondaire, hors verdict [D42, P2]. Choix des métriques justifié par Metrics Reloaded [protocole § 5 ; réf. 2].

**Les quatre phases d'exécution** (après le gel, phase 0 ; la phase D est la vérification QA) [protocole § 10.2, D43] :

| Phase | Ce qu'elle fait | Lit la vérité du test ? | Trace |
|---|---|---|---|
| 0 bis | Empreintes SHA-256 des 5 prédictions de TotalSegmentator, avant toute autre opération (prédictions refaites selon D44) | non | `config/empreintes_totalseg_etape4.json` [D44] |
| A | Essai à blanc de tout le code de mesure sur des cas d'**entraînement** | non | faite ; son résultat n'est pas consigné au journal [README] |
| B | Prédictions des 4 modèles sur le test (lecture des scanners seulement), contrôle de forme (R0) et de plausibilité, **manifeste** d'empreintes figé | non | manifeste `75a8ce43c4f3eb4539aa9ef254c24c262d09afa2251698690da20adc0fe5e65d`, `config/empreinte_manifeste_etape6.json` [D42, complément] ; observations écrites avant l'ouverture [D45] |
| C | Ouverture unique : lecture de la vérité, calcul de toutes les métriques, verdict | oui, une seule fois | tentative n° 1, commit `1ab4828` [D46] |

**Scellés et empreintes.** Une empreinte SHA-256 est une suite de 64 caractères calculée sur tous les octets d'un fichier : un seul octet changé la change. Chaque phase vérifie au démarrage, et s'arrête sinon [protocole § 9, D43] : l'empreinte du protocole, celle du fichier de seuils figé (`c1b24707…ef90672`, D36), quatre valeurs du fichier de seuils que le code n'avait pas relues (D36), le découpage, et à partir de la phase B les empreintes de TotalSegmentator. La phase C refuse de démarrer si une seule prédiction ou un seul contrôle diffère du manifeste, si le dépôt n'est pas propre ou si `scripts/` diffère du commit de la phase B ; un registre d'ouverture interdit toute relance après une tentative terminée [D43, déviations 6 et 7]. Avant la phase C, aucun chemin vers `labelsTs` ne peut être lu : le script s'arrête (code 18) [D43 ; scripts/etape6_commun.py].

## 7. Résultats

*Rappel des métriques* [docs/protocole_etape6.md § 5.2] :

- **Dice** : recouvrement entre le masque prédit et la vérité, de 0 (aucun) à 1 (parfait) ;
- **HD95** (distance de Hausdorff à 95 %) : distance en mm entre les deux contours, au 95e percentile, dans sa version symétrique de MONAI ; plus bas = mieux ;
- **NSD à 2 mm** (distance de surface normalisée) : part des contours situés à moins de 2 mm l'un de l'autre ; métrique secondaire, hors verdict ;
- **écart de volume signé** : volume prédit − volume de la vérité ; positif = sur-segmentation (la prédiction est trop grande).

Les chiffres sont arrondis à l'affichage seulement. Une différence de Dice de 0,005 n'est pas une différence [docs/protocole_etape6.md § 8.1].

### 7.1 Validation croisée (étape 5)

**Les chiffres de cette sous-section sont des chiffres de validation, pas de test.** Ils viennent des 15 cas d'entraînement, chacun prédit par le modèle du pli qui ne l'avait pas vu (prédictions *hors pli*).

- **nnU-Net, 5 plis × 250 époques** : Dice moyens de validation de 0,850, 0,919, 0,908, 0,933 et 0,920 pour les plis 0 à 4 [D31]. Sur l'ensemble des 15 cas, le Dice moyen hors pli est de **0,906 par cas** et de **0,903 par patient** (paires comptées une fois), avec une **médiane de 0,924** [D31]. Dice moyen de l'**AP : 0,827** (0,8271 sans le nettoyage, qui n'est pas appliqué à l'AP) [D33 ; D42].
- **Échec de 1019** : aorte Dice **0,00**, AP 0,46, alors que les cavités et le myocarde y sont entre 0,89 et 0,96 [D26]. Cause : une **phase d'injection atypique**. Sur 1019, l'aorte est à 253 HU et l'AP à 416 HU ; sur 1001, l'aorte est à 562 HU et l'AP à 192 HU [D27]. Le modèle a laissé l'aorte, plus grise, dans le fond. 1019 est gardé dans les données, sans correction. Avec un seul cas de ce type sur 15, le modèle ne peut pas apprendre cette situation [D27].
- **Classement des causes des cas faibles** [D31] :
  - hors domaine : aorte et AP de 1019 ;
  - annotation atypique : OD de 1002 ;
  - définition floue : AP et VD autour de la valve pulmonaire (1015) ;
  - frontière ambiguë : myocarde VG de 1018 ;
  - erreur du modèle : fragment isolé de l'AP de 1018.

  **Aucun échec majeur ne relève d'une erreur pure du modèle** [D31].
- **Effet du post-traitement** : le Dice moyen hors pli reste **inchangé (0,906)**. Après post-traitement, OG, OD, VD et aorte sont d'un seul tenant sur les 15 cas ; l'AP garde plusieurs morceaux dans 9 cas [D33].
- **SegResNet pli 0 contre nnU-Net pli 0** (mêmes 3 cas de validation : 1001, 1019, 1020) [D41] :
  - Dice moyen hors pli de **0,829** pour SegResNet (1 250 époques, `model_final.pt`) contre **0,850** pour nnU-Net ;
  - par structure, pour SegResNet : myocarde VG 0,900 ; OG 0,949 ; cavité VG 0,918 ; OD 0,859 ; VD 0,859 ; aorte 0,752 ; AP 0,566 ;
  - l'écart se concentre sur l'AP et l'OD. Un plateau est atteint (0,827 à l'époque 949, 0,829 à l'époque 1 249) : l'écart ne vient donc pas du budget d'entraînement ;
  - l'aorte de 1019 est partiellement trouvée par SegResNet (Dice 0,32, contre 0,00 pour nnU-Net). C'est une piste, pas une preuve : un seul cas, un seul entraînement [D41]. À 300 époques, le modèle sous-entraîné donnait 0,691 [D40].

### 7.2 Test : verdict (C1, C2)

Le verdict porte sur le **pipeline A seulement**. Les critères ont été fixés avant l'ouverture [D42 ; docs/protocole_etape6.md § 8.4] :

- **C1** : médiane du Dice ≥ 0,85 pour les structures 1 à 6 et ≥ 0,80 pour l'AP, comparée sur les valeurs non arrondies ;
- **C2** : aucun échec silencieux.

| Structure | Médiane du Dice (non arrondie) | Arrondie (4 déc.) | n | Seuil | Marge | C1 |
|---|---|---|---|---|---|---|
| myocarde VG | 0,9215323328971863 | 0,9215 | 5 | 0,85 | +0,0715 | rempli |
| OG | 0,9282763600349426 | 0,9283 | 5 | 0,85 | +0,0783 | rempli |
| cavité VG | 0,9642567038536072 | 0,9643 | 5 | 0,85 | +0,1143 | rempli |
| OD | 0,8813151717185974 | 0,8813 | 5 | 0,85 | **+0,0313** (la plus faible) | rempli |
| VD | 0,9024218916893005 | 0,9024 | 5 | 0,85 | +0,0524 | rempli |
| aorte | 0,9819454550743103 | 0,9819 | 5 | 0,85 | +0,1319 | rempli |
| AP | 0,8961717486381531 | 0,8962 | 5 | 0,80 | +0,0962 | rempli |

Sources : médianes [resultats/etape6/agregats.csv ; resultats/etape6/rapport_etape6.md ; D46] ; seuils [D42] ; marges calculées depuis resultats/etape6/agregats.csv.

- **Minimum global** : le plus petit Dice de A sur les 35 couples (cas, structure) est **0,838**, sur l'AP de 1004 [D46 ; resultats/etape6/metriques_par_cas.csv].
- **Échecs silencieux** : **aucun**. `echecs_silencieux.csv` ne contient que son en-tête : aucun Dice de A n'est inférieur à 0,50 [resultats/etape6/echecs_silencieux.csv ; D46]. C2 est rempli.
- **Cas dégénérés et anomalies** : aucun (§ 7.3).

**Verdict : ACCEPTABLE, à titre indicatif** [D46 ; resultats/etape6/resultats_etape6.json]. Il découle mécaniquement des critères figés. Ses limites sont écrites depuis D42 :

- les seuils ont été fixés après la validation croisée et ne sont donc pas totalement indépendants de nos résultats ;
- avec 5 patients, un seul cas peut faire basculer une médiane ;
- les chiffres de Zhuang et al. 2019 ne sont pas directement comparables.

S'y ajoutent un seul cas à 0,45 mm (1014) et une sensibilité du contrôle non mesurable [D46].

### 7.3 Test : résultats par cas et par structure (A)

**Dice moyen par cas** (moyenne des 7 structures) [resultats/etape6/dice_moyen_par_cas.csv] :

| Cas | 1004 | 1010 | 1011 | 1013 | 1014 (seul cas à 0,45 mm) |
|---|---|---|---|---|---|
| Dice moyen | 0,915 | 0,938 | 0,926 | 0,919 | 0,923 |

**Par structure, sur les 5 cas, alertes comprises** [resultats/etape6/agregats.csv ; resultats/etape6/rapport_etape6.md] :

| Structure | Dice médian | HD95 médiane (mm) | HD95 max (mm) | NSD 2 mm médiane | Écart de volume médian signé |
|---|---|---|---|---|---|
| myocarde VG | 0,922 | 1,9 | 2,0 | 0,966 | +0,2 mL (+0 %) |
| OG | 0,928 | 6,3 | 11,0 | 0,842 | +7,1 mL (+10 %) |
| cavité VG | 0,964 | 1,6 | 2,9 | 0,974 | −2,6 mL (−3 %) |
| OD | 0,881 | 8,4 | 11,4 | 0,782 | +9,1 mL (+15 %) |
| VD | 0,902 | 6,0 | 11,1 | 0,800 | −9,2 mL (−8 %) |
| aorte | 0,982 | 0,9 | 1,3 | 0,993 | +0,6 mL (+2 %) |
| AP | 0,896 | 14,5 | 17,6 | 0,866 | +0,1 mL (+0 %) |

Les médianes en mL et en % sont calculées séparément : elles peuvent venir de cas différents.

Lecture :

- Les structures les mieux segmentées sont l'aorte et la cavité VG : Dice médian ≥ 0,96, HD95 médiane ≤ 1,6 mm.
- L'AP a la plus grande HD95 (médiane 14,5 mm, maximum 17,6 mm), alors que son volume médian est presque exact. L'erreur porte plutôt sur l'étendue ou la position que sur la taille, comme à l'étape 5 [D31 ; D33 ; D46].
- **Aucun cas dégénéré** (structure absente de la prédiction ou de la vérité), aucune valeur plafond de HD95, aucune anomalie [resultats/etape6/metriques_par_cas.csv ; resultats/etape6/tracabilite_06d.json ; D46].
- En complément, et **pas comme résultat principal** [D29, Q5] : sans les 2 cas en alerte (n = 3), les médianes du Dice vont de 0,880 (OD) à 0,980 (aorte) [resultats/etape6/agregats.csv].

### 7.4 Alertes

**Contrôle sur A (alertes officielles)** : **2 cas en alerte sur 5**, par la seule règle R5 (écart de densité AP − VD, seuil 176 HU) [resultats/etape6/alertes.csv ; config/seuils_plausibilite.json] :

| Cas | Règle | Mesure : AP / VD (écart) | Grandeurs décisives (règle de la section 5.5) | Classement |
|---|---|---|---|---|
| 1004 | R5 | 536 / 252 HU (284 HU) | Dice AP **0,838** < 0,85 ; Dice VD 0,924 | **vraie** |
| 1010 | R5 | 379 / 197 HU (182 HU) | Dice AP 0,947 ; Dice VD 0,915 (tous deux ≥ 0,85) | **fausse** |

- Aucune alerte « hors domaine » et aucune alerte non classable [resultats/etape6/rapport_etape6.md ; D46].
- **Fausses alertes : 1 sur 5** (1010), contre **environ 2 sur 14** attendues [D36 ; D46]. Sur 5 cas, cette estimation reste très imprécise [docs/protocole_etape6.md § 12].
- L'alerte de 1010 dépasse le seuil de 6 HU seulement (calculé depuis resultats/etape6/alertes.csv).

**Hypothèse de D45 b) : ni confirmée ni infirmée** [D46]. Avant l'ouverture, D45 b) avait écrit que R5 réagissait probablement à une **propriété des images** (par exemple le moment de l'injection) plutôt qu'à une erreur de segmentation. Selon la règle figée (protocole § 7.3 ; section 5.5), R5 se classe par le Dice, jamais par une densité mesurée sur la vérité : l'hypothèse n'est donc pas testée [D46]. Les mesures sont seulement *compatibles* avec elle, sans mesure sur la vérité [D46] :

- sur **1004**, les densités sont presque identiques avec **3 des 4 masques prédits** : A 536/252 HU, S0 520/253, T 540/253 (AP/VD). Seule exception : **N0**, qui ne déclenche pas R5 mais dont l'AP est la plus fausse (Dice 0,462) ;
- sur **1010**, elles sont presque identiques avec tous les masques : A et N0 379/197, S0 376/197, T 378/197.

Sur 1004, l'AP est aussi la structure la plus difficile : Dice de 0,838 pour A, 0,462 pour N0 et 0,750 pour S0. Son volume est juste (+0,5 %) malgré une HD95 de 17,6 mm : c'est une erreur d'étendue ou de position, pas de volume [D46].

**Repère 0,85 de l'AP** : la règle de classement (protocole § 7.3, P8) utilise le repère « Dice faible » de 0,85, plus strict que le seuil de verdict de l'AP (0,80). Une alerte qui met en cause l'AP est donc facilement classée vraie : c'est le cas de 1004 (0,838, entre 0,80 et 0,85). Ce repère est **voulu par le protocole** (protocole § 7.3, P8, D30). Il est noté comme limite, sans toucher aux seuils [D46 ; protocole § 11 ; D36].

**Contrôle sur les autres modèles (informatif, hors estimation)** [resultats/etape6/alertes.csv ; resultats/etape6/rapport_etape6.md] :

- N0 : 3 cas en alerte (1004, 1010, 1013). Le seul Dice < 0,50 des trois comparaisons (AP de N0 sur 1004, 0,462) est couvert par trois alertes (R2, R3 et R8 sur l'AP) : ce n'est pas un échec silencieux ;
- S0 : 4 cas en alerte (1004, 1010, 1011, 1014) ;
- T : aucune alerte retenue après restriction (§ 7.6).

### 7.5 Comparaison (b) : N0 contre S0 (règle P6)

N0 et S0 sont entraînés sur les **mêmes 12 cas** pendant le **même temps de calcul**, et évalués **sans post-traitement** [D41]. La règle descriptive P6, fixée avant l'ouverture, dit qu'une méthode est « meilleure » sur une structure si deux conditions sont remplies : un écart de médiane **> 0,02** (valeurs non arrondies) **et** de meilleurs résultats sur **au moins 4 cas sur 5**. Sinon, la conclusion est « pas de différence démontrée » [docs/protocole_etape6.md P6 ; D43, déviation 11]. Ce n'est pas une preuve statistique.

| Structure | Médiane N0 | Médiane S0 | Écart N0 − S0 | Cas où N0 fait mieux | Cas où S0 fait mieux | Conclusion |
|---|---|---|---|---|---|---|
| myocarde VG | 0,922 | 0,918 | +0,004 | 3 | 2 | pas de différence démontrée |
| OG | 0,932 | 0,910 | **+0,021** | 3 | 2 | pas de différence démontrée |
| cavité VG | 0,964 | 0,956 | +0,008 | **5** | 0 | pas de différence démontrée |
| OD | 0,879 | 0,874 | +0,005 | 3 | 2 | pas de différence démontrée |
| VD | 0,895 | 0,899 | −0,004 | 3 | 2 | pas de différence démontrée |
| aorte | 0,981 | 0,976 | +0,005 | **5** | 0 | pas de différence démontrée |
| AP | 0,878 | 0,869 | +0,009 | 3 | 2 | pas de différence démontrée |

Sources : médianes, nombres de cas et conclusions [resultats/etape6/comparaison_b_p6.csv] ; écarts calculés depuis resultats/etape6/comparaison_b_p6.csv.

- **Pas de différence démontrée sur les 7 structures** [D46]. Pour l'OG, l'écart de médiane dépasse 0,02 (+0,0215), mais N0 ne fait mieux que sur 3 cas sur 5. Pour la cavité VG et l'aorte, N0 fait mieux sur 5 cas sur 5, mais l'écart reste < 0,02 [D46].
- Dice moyen sur les 5 cas : **N0 0,910, S0 0,904** [D46 ; calculé depuis resultats/etape6/dice_moyen_par_cas.csv].
- Pires cas : AP de N0 sur 1004 (0,462) ; OD de S0 sur 1014 (0,755, HD95 56,3 mm) [resultats/etape6/rapport_etape6.md].
- **Limites** [docs/protocole_etape6.md § 12 ; D41 ; D46] :
  - un seul modèle par méthode, donc la variabilité d'un entraînement à l'autre n'est pas mesurée ;
  - même temps de calcul, mais environ 8 fois moins de morceaux d'image vus par SegResNet ;
  - réglages d'inférence non alignés (P7, § 4.4) ;
  - 5 cas seulement.

### 7.6 Comparaison (c) : TotalSegmentator contre A, structures 1 à 5

| Structure | Dice médian T | Dice médian A | Écart T − A | Dice min T | Dice min A | HD95 médiane T / A (mm) |
|---|---|---|---|---|---|---|
| myocarde VG | 0,916 | 0,922 | −0,005 | 0,895 | 0,911 | 1,7 / 1,9 |
| OG | 0,943 | 0,928 | +0,015 | 0,857 | 0,857 | 3,8 / 6,3 |
| cavité VG | 0,967 | 0,964 | +0,003 | 0,863 | 0,927 | 1,7 / 1,6 |
| OD (prudence, D23) | 0,884 | 0,881 | +0,003 | 0,863 | 0,864 | 6,9 / 8,4 |
| VD (prudence, D23) | 0,914 | 0,902 | +0,011 | 0,877 | 0,893 | 5,0 / 6,0 |

Sources : [resultats/etape6/agregats.csv ; resultats/etape6/rapport_etape6.md ; D46] ; écarts calculés depuis resultats/etape6/agregats.csv.

- **Description, pas conclusion** : les écarts de médiane sont ≤ 0,015 en valeur absolue. Le protocole ne fixe **aucune règle de comparaison** pour (c), car P6 ne vaut que pour (b) [D46].
- **Pire cas** : le pire cas de T est plus bas que celui de A. Sur **1013**, T obtient un Dice de 0,863 sur la cavité VG, avec un volume de +22 %. Dice moyen sur les structures 1 à 5 pour 1013 : 0,885 pour T, contre 0,910 pour A [D46 ; resultats/etape6/dice_moyen_par_cas.csv].
- **Prudence sur l'OD et le VD** [D23] : sur deux cas d'entraînement, les volumes de TotalSegmentator s'écartaient de la vérité (1002 : OD +46 %, VD −18 %), avec des frontières moins nettes. Un écart sur ces deux structures peut venir de la définition, pas de la qualité de l'outil.
- **Aucune alerte retenue pour T** [D45 c) ; D46 ; resultats/etape6/rapport_etape6.md]. Avant restriction, le module avait signalé R8 sur l'aorte pour les 5 cas et R3 sur 1004, 1010 et 1014 ; R5 s'y déclenchait aussi, à titre informatif [D45 b), c)]. Ces règles portent sur l'aorte ou l'AP et sont écartées à la lecture (protocole § 7.1, ligne (c) ; D29, Q7). L'aorte de TotalSegmentator est définie autrement (D23), et les seuils n'ont pas été calibrés pour T [D45 c)].
- Rappel : la phase 1 ne vise pas à battre TotalSegmentator, mais à construire une chaîne de mesure fiable [docs/protocole_etape6.md § 4 (c)].

### 7.7 Écart entre test et validation croisée

| Grandeur | Test (A, 5 cas) | Validation croisée (hors pli, 15 cas) |
|---|---|---|
| Dice moyen par cas, médiane | 0,923 | 0,924 |
| Dice moyen par cas, moyenne | 0,924 | 0,906 |
| Dice de l'AP, moyenne | 0,893 | 0,827 |

Sources : [D46] ; test calculé depuis resultats/etape6/dice_moyen_par_cas.csv (médiane 0,9229, moyenne 0,9242) et resultats/etape6/agregats.csv (AP, moyenne 0,8929) ; validation croisée [D31 ; D33].

- **Les médianes sont quasi identiques** (0,923 contre 0,924). La moyenne plus haute sur le test s'explique **probablement surtout par l'absence, dans le test, d'un cas du type de 1019**, dont l'échec sur l'aorte tirait la moyenne hors pli vers le bas. Cet effet n'est **pas chiffré** [D46].
- **L'effet de l'ensemble des 5 modèles est possible mais non mesuré** : sur le test, chaque cas est prédit par l'ensemble des 5 plis, alors qu'en validation croisée chaque cas était prédit par un seul modèle [D46].
- **Le post-traitement n'y est pour rien** : le Dice moyen hors pli est inchangé avec lui [D33 ; D46].
- **Aucune dégradation n'est observée entre validation et test, sur 5 cas** [D46]. Cela ne prouve pas la généralisation à d'autres données. Le résultat reste une performance interne à la cohorte MM-WHS [docs/protocole_etape6.md § 12].

### 7.8 Biais de volume de l'OD et du VD

Un *biais* est ici une erreur qui va toujours dans le même sens (toujours trop grand, ou toujours trop petit).

- **OD sur-segmentée dans 5 cas sur 5** pour A : écarts de +0,6 à +11,5 mL, médiane signée **+9,1 mL (+15 %)**. Calculé depuis resultats/etape6/metriques_par_cas.csv ; médiane [resultats/etape6/agregats.csv].
- **VD sous-segmenté dans 4 cas sur 5** pour A : seul 1014 est au-dessus (+5,4 mL). Médiane signée **−9,2 mL (−8 %)**. Calculé depuis resultats/etape6/metriques_par_cas.csv ; médiane [resultats/etape6/agregats.csv].
- **Même sens chez les autres modèles du test** (calculé depuis resultats/etape6/metriques_par_cas.csv) :
  - OD sur-segmentée dans 5 cas sur 5 pour N0, S0 et T ;
  - VD sous-segmenté dans 4 cas sur 5 pour N0, et dans 3 cas sur 5 pour S0 et T ;
  - pour T, médianes signées OD +8,6 mL (+13 %) et VD −3,5 mL (−3 %) [resultats/etape6/agregats.csv].
- **Même sens qu'à l'étape 5 sur 1002** : OD prédite 61 % plus grande que l'annotation ; 42 % des oublis du VD prédits comme OD, donc une frontière OD/VD déplacée [D31]. Et même sens que **TotalSegmentator** sur 1002 (OD +46 %, VD −18 %) [D23].
- **Hypothèse** : une **convention d'annotation de la frontière OD/VD** différente de celle que les modèles reproduisent. Elle est cohérente avec la répétition du même sens entre trois familles de méthodes, dont un outil entraîné sur d'autres données (T). Formulée après l'ouverture (contrairement à l'hypothèse de D45 b) [D47, point 3], elle n'est **pas démontrée** : aucune analyse de cette frontière sur le test n'est consignée, et la variabilité entre annotateurs de MM-WHS est inconnue [docs/protocole_etape6.md § 12 ; D31].


## 8. Déviations

Une **déviation** est un écart au protocole figé. Toutes celles qui suivent ont été consignées au journal, et les deux premières séries **avant** l'ouverture du test, pour que ce rapport les distingue de tout écart survenu après.

**8.1 Déviations connues avant l'ouverture, écrites en codant (D43, 15 déviations, acceptées le 2026-11-17).** Résumé, texte complet dans D43 :

| N° | Objet | Effet |
|---|---|---|
| 1 | Ensemble nnU-Net : moyenne des *logits* (scores bruts) par défaut, et non des probabilités | imposé par l'outil (P7) ; écart marginal |
| 2 | SegResNet sur le test : 4 réglages de plus que l'évaluation de 05m ; contrôle de géométrie plus strict que R0 | imposé par l'absence de vérité |
| 3 | Catégorie « non classable » pour une alerte dont la grandeur de classement n'est pas définie ; règle « ou » | cas non prévu ; aucune alerte non classable n'est survenue [D46] |
| 4 | Morceaux de R8 recomptés avec `ndimage` | imposé par l'outil ; recoupé |
| 5 | Géométrie : arrêt si une prédiction n'a pas la géométrie du scanner sans alerte R0 ; vérité hors géométrie traitée comme cas dégénéré | cas non prévu ; aucun cas dégénéré [D46] |
| 6, 7, 13, 15 | Ouverture unique plus stricte, statut `arret_defaut`, même commit pour la phase B, mesures de 05j reprises seulement si identiques | règles plus strictes |
| 8 | Arrondis : fichiers non arrondis, seul le rapport lisible arrondit | présentation |
| 9, 10 | Empreintes de référence versionnées ; empreinte de la traçabilité de l'étape 4 relevée aussi | imposé par l'absence de vérité |
| 11 | Règle P6 : inégalité stricte, « pas de différence démontrée » | règle plus stricte |
| 12 | Phase A : scores de TotalSegmentator sur 1002 et 1003 pour vérifier le code seulement | présentation |
| 14 | Correction d'une prédiction de T non implémentée : arrêt si R0 échoue | cas non survenu (aucune correction en phase B [resultats/etape6/tracabilite_06d.json, champ totalseg_corriges_en_phase_B vide]) |

**8.2 Prédictions de TotalSegmentator refaites (D44, 2026-11-22).** Les prédictions de l'étape 4 ont été perdues, probablement lors de l'incident du Drive plein (seule celle de 1004 subsistait). Elles ont été refaites avec la même version, la même tâche, les mêmes réglages et la même licence, sur un GPU L4 au lieu du T4 d'origine, en lisant seulement les scanners. Comparaison voxel par voxel de 1004 : **139 voxels différents sur 52 428 800 (0,0003 %)**, dispersés sur toutes les étiquettes ; reproductible à un bruit numérique près ; cause probable (matériel ou bibliothèques) non démontrée. Limite : l'identité des 4 autres prédictions avec celles de l'étape 4 ne peut pas être prouvée [D44].

**8.3 Anomalie de traçabilité de la phase C (D46).** `tracabilite_06d.json` porte le commit `1ab48283c215bc2e2c52d89915813b0cc0a305eb+modifié`, alors que le registre de la tentative porte le même commit sans suffixe [resultats/etape6/tracabilite_06d.json, D46].

**Conclusion de l'examen en lecture seule** (agent qa-reviewer, 2026-11-23). Le suffixe « +modifié » est ajouté par `commit_git()` (`scripts/05a_preparer_nnunet.py`) quand `git status --porcelain` liste au moins un fichier suivi modifié, ou un fichier non suivi et non ignoré, dans le clone de Colab. 06d appelle cette fonction deux fois : au démarrage (`verifier_commit`, qui refuse tout « + » ; c'est la valeur écrite au registre), puis à la fin du calcul, dans `tracabilite()`, juste avant l'écriture des résultats. Le clone est donc devenu « modifié » pendant l'exécution (session `20261123_215037`, entre son début et 21:56:37 UTC). Le dépôt permet d'exclure trois explications. (1) Le code lu (06d, `etape6_commun.py`, `etape6_metriques.py` et les modules qu'ils chargent) n'écrit rien dans le dépôt : le registre, le verrou, la copie de l'affichage et les résultats vont dans `--sortie`, sur Drive, que `verifier_hors_depot` interdit dans le dépôt. (2) Au commit `1ab4828`, le `.gitignore` ignorait déjà `__pycache__/` (fichiers compilés que Python crée en chargeant les modules) et `resultats/`. (3) Tous les fichiers suivis sont enregistrés avec des fins de ligne LF, ce qui écarte un faux « modifié » dû à une conversion des fins de ligne. La cause reste **non identifiable depuis le dépôt** : il faudrait une action extérieure au script dans la session Colab (terminal, explorateur ou éditeur de fichiers, autre processus) ou une bibliothèque écrivant dans le dossier courant (`/content/poc-segmentation-ct`, d'où la cellule 29 lance le script), sans qu'aucun élément du dépôt ne permette de choisir. **Effet : traçabilité seulement, pas le calcul.** Les modules du dépôt ont été chargés au démarrage, sur un dépôt vérifié propre ; les fichiers du dépôt utilisés (seuils, découpage, références du manifeste et de TotalSegmentator) ont été lus et vérifiés par empreinte avant l'ouverture, et le calcul n'en lit plus aucun ensuite ; HEAD est resté `1ab4828`. **Pour trancher, en lecture seule** : si la session Colab existe encore, `git -C /content/poc-segmentation-ct status --porcelain --ignored` et `git -C /content/poc-segmentation-ct diff --stat` nommeraient le fichier en cause. Sinon, relire dans Drive `poc_ct/etape6/phaseC/sortie_06d_20261123_215037.txt` (message inattendu pendant le calcul) et `poc_ct/etape6/phaseC/registre_ouverture.json` (statut final « terminée », heure de fin, `sha256_resultats_json` égal à `b69a24c3…5bec110`), et consulter l'historique d'exécution du carnet pour savoir si une autre action a eu lieu entre 21:50 et 21:57 UTC. Le clone de Colab est effacé à la fin de la session : si elle est close, la cause ne pourra probablement plus être établie.

## 9. Limites

**Rappelées de D42 (écrites avant l'ouverture)** :
- les seuils du verdict ont été fixés **après** la validation croisée (Dice hors pli 0,906, AP 0,827) : ils ne sont pas totalement indépendants de nos résultats ;
- avec **5 patients**, le verdict est **indicatif** ;
- les chiffres de Zhuang et al. [réf. 1] ne sont pas directement comparables (autre lot de test, moyenne et non médiane, Dice du cœur entier).

**Apparues ensuite** :
- **un seul cas à 0,45 mm** dans le test (1014) : aucune conclusion propre à cette famille d'examens [D20, D46] ;
- **sensibilité du contrôle de plausibilité non mesurable** : aucun échec réel dans le test, un seul échec connu à l'entraînement (1019) [D36, D46] ;
- **R5**, classée « erreur probable », a donné la seule fausse alerte (1010) ; l'hypothèse d'une réaction à une propriété des images (D45 b) n'est ni confirmée ni infirmée : les mesures y sont seulement compatibles ; le repère de classement de l'AP (0,85) est plus strict que son seuil de verdict (0,80), si bien qu'une alerte mettant en cause l'AP est facilement classée vraie. Noté, sans toucher aux seuils [D46, erratum D47 ; protocole § 11] ;
- **comparaison (b)** : un seul modèle par méthode (pli 0, 12 cas), réglages de prédiction non alignés [P7, D45, D46] ; **comparaison (c)** : aucune règle de comparaison prévue, écarts de définition possibles sur l'OD et le VD [D23, D46] ; elle porte sur des données publiques de recherche, selon la convention d'annotation de MM-WHS, sur les structures 1 à 5 et sur 5 patients : elle ne dit pas si un modèle entraîné sur une autre population (par exemple des cœurs pathologiques) ou selon une autre convention d'annotation ferait mieux qu'un modèle générique comme TotalSegmentator [D47] ;
- **prédictions de TotalSegmentator refaites** (D44) ;
- **une seule source de données** (MM-WHS), sans scanner sans contraste, sans cœur pathologique identifié, sans autre centre ; une seule phase d'injection atypique à l'entraînement (1019) [D27, README, stratégie face aux cas difficiles] ;
- **annotations** : définitions de frontières floues (valve pulmonaire, étendue de l'AP, frontière OD/VD), sans étude de variabilité entre annotateurs [D31] ;
- **traçabilité** : résultat de la phase A non consigné au journal (le code n'a été qualifié que sur données synthétiques, puis par cette phase A) [D43, README] ; anomalie « +modifié » de la phase C, cause non identifiable depuis le dépôt, sans effet sur le calcul (section 8.3) ; statut final « terminée » du registre d'ouverture non vérifiable depuis le dépôt [D46] ; réglages d'inférence de la phase B recopiés au journal après coup depuis la sortie de 06c conservée dans Drive [D47].

**Contrôle de plausibilité** :
- seuils révisés une fois après vérification (D35), donc fausses alertes mesurées à l'entraînement optimistes ; règles conçues après avoir vu l'échec de 1019 [D35, D36 ; spécification § 9] ;
- il juge la vraisemblance, pas l'exactitude : R3 et R8 réduites, et sans R9 une permutation cohérente de deux structures peut passer [spécification § 9] ;
- l'estimation « 1 fausse alerte sur 5 » est très imprécise avec 5 cas [protocole § 12 ; D36] ;
- non calibré pour TotalSegmentator : R3, R5 et R8 sur l'aorte écartées pour T [D45 c), protocole § 7.1].

**Modèles et prédictions** :
- budget d'entraînement réduit : 250 époques au lieu de 1 000 et configuration `3d_fullres` seule pour nnU-Net [D24, D25] ; SegResNet a vu environ 8 fois moins de morceaux d'image à temps de calcul égal [D40, D41] ;
- reproductibilité de l'entraînement limitée : état des tirages au hasard non repris à la reprise, `determ` = false [D39] ; un seul passage par modèle sur le test, non-déterminisme non mesuré [protocole § 12] ;
- ensemble nnU-Net par moyenne des *logits* (scores bruts) et non des probabilités, écart « marginal » non mesuré [D43, déviation 1] ;
- S0 : repli de mémoire pendant la prédiction ; le lien entre l'avertissement et le repli est une déduction, et le repli effectif n'est pas connu [D45 a)] ;
- TotalSegmentator refait sur GPU L4 au lieu de T4 ; seules les versions de TotalSegmentator et de nnU-Net contrôlées (2.8.1 épinglée par hypothèse) ; seul 1004 comparé à l'ancienne prédiction [D44].

**Mesures et interprétation** :
- HD95 calculée de façon symétrique par MONAI : non comparable aux HD95 d'autres logiciels [protocole § 5.2 ; resultats/etape6/tracabilite_06d.json] ;
- performance interne à la cohorte MM-WHS [protocole § 12] ;
- écart entre test et validation croisée non expliqué quantitativement : effet de l'ensemble non mesuré, effet de l'absence d'un cas du type de 1019 non chiffré [D46] ;
- biais de volume systématique de l'OD et du VD (section 7.8) : hypothèse d'une convention d'annotation, formulée après l'ouverture, non démontrée.

## 10. Conclusion et suites

**Conclusion.** Sur les 5 patients du test de MM-WHS, le pipeline principal (nnU-Net, ensemble de 5 modèles, post-traitement, contrôle de plausibilité) remplit les deux critères fixés avant l'ouverture : verdict **ACCEPTABLE, indicatif** [D46]. La chaîne de mesure a fonctionné de bout en bout sous scellés : protocole figé, empreintes vérifiées, ouverture unique, déviations consignées avant l'ouverture ; principales réserves : une anomalie de traçabilité sans effet sur le calcul (section 8.3) et le statut final du registre d'ouverture, non vérifiable depuis le dépôt. Le résultat ne dit rien de la fiabilité sur d'autres scanners, d'autres centres ou d'autres pathologies. De même, la comparaison avec TotalSegmentator, faite sur des données publiques de recherche, selon la convention d'annotation de MM-WHS, sur les structures 1 à 5 et sur 5 patients, ne dit pas si un modèle entraîné sur une autre population (par exemple des cœurs pathologiques) ou selon une autre convention d'annotation ferait mieux qu'un modèle générique : c'est une question à tester avec des données et une vérité terrain propres à l'usage visé [D47].

**Suites, phase 2** [README, « Stratégie face aux cas difficiles » ; D27, D28, D29, D31] :
- **données** : jeu ImageCAS annoté (plus de cas), autres centres et constructeurs, cœurs pathologiques, scanners sans contraste, plusieurs phases d'injection ;
- **sous-groupes** : relever la phase d'injection et la phase cardiaque pour chaque examen ; stratifier le découpage sur tous les sous-groupes connus ; rapporter les résultats par sous-groupe ;
- **annotations** : protocole d'annotation écrit (plans valvulaires, étendue des branches) ; étude de variabilité entre annotateurs (principe du Gage R&R : part de la variation due à l'instrument et à l'opérateur) ;
- **contrôle** : contrôle d'entrée avant la segmentation (en-tête DICOM, phase d'injection, absence de contraste) ; règle R9 (positions et contacts) ; recalibrage du contrôle sur des données plus variées ; circuit de revue humaine des alertes ;
- **mesures dérivées utiles au clinicien** (par exemple la masse du VG) en plus du Dice ;
- **étape 7, à poursuivre** : examen de l'anomalie de traçabilité (section 8.3) et du statut final du registre d'ouverture, en lecture seule.

## 11. Références

DOI vérifiés auprès de Crossref (arXiv pour la réf. 7), le 2026-11-23.

1. Zhuang X, Li L, Payer C, et al. Evaluation of algorithms for Multi-Modality Whole Heart Segmentation: An open-access grand challenge. *Medical Image Analysis*. 2019;58:101537. doi:10.1016/j.media.2019.101537
2. Maier-Hein L, Reinke A, Godau P, et al. Metrics reloaded: recommendations for image analysis validation. *Nature Methods*. 2024;21:195-212. doi:10.1038/s41592-023-02151-z
3. Tejani AS, Klontzas ME, Gatti AA, et al. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update. *Radiology: Artificial Intelligence*. 2024;6:e240300. doi:10.1148/ryai.240300
4. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*. 2021;18:203-211. doi:10.1038/s41592-020-01008-z
5. Wasserthal J, Breit HC, Meyer MT, et al. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. *Radiology: Artificial Intelligence*. 2023;5:e230024. doi:10.1148/ryai.230024
6. Myronenko A. 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization. In: *Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (BrainLes 2018)*. Lecture Notes in Computer Science. Springer; 2019:311-320. doi:10.1007/978-3-030-11726-9_28
7. Cardoso MJ, Li W, Brown R, et al. MONAI: An open-source framework for deep learning in healthcare. arXiv:2211.02701. 2022. doi:10.48550/arXiv.2211.02701
8. Mongan J, Moy L, Kahn CE Jr. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): A Guide for Authors and Reviewers. *Radiology: Artificial Intelligence*. 2020;2:e200029. doi:10.1148/ryai.2020200029
