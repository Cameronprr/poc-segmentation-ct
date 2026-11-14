# Spécification : contrôle de plausibilité des segmentations (décision D28)

***Statut : FIGÉ le 2026-11-14 (D36).** Seuils dans `config/seuils_plausibilite.json`, empreinte SHA-256 `c1b247070765d193a46e90b538cc655787dbb65f8fdd13f113a39efd3ef90672`. Aucune modification après le gel (section 7).*
*Version 1.1 du 2026-11-14 (D35 : révision unique avant le gel, décidée après la première exécution de la section 23 ; voir section 7.1). Version 1.0 du 2026-11-11. Questions Q1 à Q9 tranchées par l'utilisateur (section 10, décision D29). Le contrôle est codé dans `scripts/controle_plausibilite.py` ; calibrage et vérification par `scripts/05j_calibrer_controle.py` (section 23, seuils proposés, figés depuis par D36). Scripts de mesure : `05e_densites_vaisseaux.py` (densités, R4 à R7, section 18 du notebook) et `05h_volumes_verite.py` (volumes et rapports, R2 et R3, section 21, D32), tous deux exécutés ; résultats en fin de section 5. Seuils proposés par 05j, validés et figés (D36, section 7.2). Calendrier du calibrage (section 6.2) modifié par D32 : tout le calibrage attend la décision sur le post-traitement de nnU-Net (05i, section 22) ; R2 à R7 restent calibrés sur la vérité, sauf R6, fixée à 0 HU (D35) ; R8 aussi, sur la vérité (D34), puis toutes les règles sont appliquées aux prédictions hors pli post-traitées (D33).*
*Rédigée sans ouvrir le lot de test. Les seuls chiffres cités viennent du pli 0 (cas d'entraînement 1001, 1019 et 1020), de D27, et des mesures 05e et 05h sur la vérité des 15 cas d'entraînement (fin de la section 5).*

---

## 1. But

Le Dice moyen de 0,850 du pli 0 cachait un échec complet sur l'aorte de `mmwhs_1019` (D26, D27). Le modèle a laissé l'aorte dans le fond et a appelé « aorte » un morceau de l'artère pulmonaire (AP). La cause est une **phase d'injection atypique**. La *phase d'injection* est le moment où les images sont prises après l'injection du produit de contraste. Sur 1019, l'aorte est à 253 HU et l'AP à 416 HU. Sur 1001, c'est l'inverse : aorte 562 HU, AP 192 HU. (HU, unités Hounsfield : l'échelle des densités du scanner, où l'eau vaut 0.)

D28 fixe la règle : **le pipeline ne rend jamais en silence un résultat hors du domaine validé.** Le *domaine validé* est l'ensemble des examens qui ressemblent à ceux sur lesquels le modèle a été vérifié.

Ce document définit le **contrôle de plausibilité**. Il s'exécute après la segmentation et répond à une question : « ce masque ressemble-t-il à un cœur segmenté correctement, sur un examen semblable à ceux qu'on connaît ? » Si la réponse est non, le cas reçoit le statut **ALERTE**. Le contrôle ne corrige rien.

## 2. Périmètre

| Inclus (phase 1) | Exclu (phase 2, en perspective) |
|---|---|
| Contrôle **après** la segmentation, à partir de l'image et du masque prédit | Contrôle **avant** la segmentation : en-tête DICOM, estimation de la phase d'injection, scanners sans contraste (section 11) |
| Masques nnU-Net et Auto3DSeg (définitions MM-WHS, étiquettes 1 à 7) | Masques TotalSegmentator : contrôle **restreint** aux structures 1 à 5, plus R0 et R7, aucune règle sur l'aorte ni l'AP (D23, Q7) ; alertes rapportées à part, à titre d'information |
| Calibrage sur les cas d'entraînement uniquement, 1019 exclu (section 6.4) | Toute donnée du lot de test avant l'étape 6 |

Le contrôle **ne remplace pas** une relecture humaine. Il ne prouve pas qu'un masque est bon. Il repère seulement les masques manifestement invraisemblables.

## 3. Entrées

| Entrée | Contenu | Utilisation |
|---|---|---|
| Image CT (NIfTI) | Intensités en HU | Densités médianes dans chaque structure prédite |
| Masque prédit (NIfTI) | Étiquettes 0 à 7 (`dataset.json` : 1 myocarde VG, 2 OG, 3 cavité VG, 4 OD, 5 VD, 6 aorte ascendante, 7 AP) | Présence, volumes, morceaux, positions |
| Géométrie | Forme, espacement des voxels (mm) et affine, lus dans l'en-tête de l'image | Conversion des voxels en mL (volume = nombre de voxels × produit des 3 espacements / 1000) et des positions en mm |
| Fichier de seuils | Seuils figés, version et date, empreinte SHA-256 | Comparaisons. **Le contrôle refuse de tourner sans fichier de seuils figé** |

Aucune vérité terrain n'est lue : le contrôle doit fonctionner sur un examen nouveau, sans annotation.

## 4. Sorties

Pour chaque cas, un **statut** :
- **OK** : aucune règle bloquante n'est déclenchée ;
- **ALERTE** : au moins une règle est déclenchée. Les règles et leurs chiffres sont listés.

Format JSON prévu (un fichier par cas, plus un récapitulatif par lot) :

```json
{
  "cas": "mmwhs_1019",
  "statut": "ALERTE",
  "modele": "nnUNet 3d_fullres, pli 0",
  "seuils": {"fichier": "config/seuils_plausibilite.json", "sha256": "…", "date_gel": "2026-11-…"},
  "regles_declenchees": [
    {"regle": "R4", "categorie": "erreur_probable",
     "libelle": "densité de l'aorte prédite éloignée de celle de la cavité VG prédite",
     "mesure": {"hu_aorte": 410, "hu_VG": 255, "ecart_hu": 155},
     "seuil": {"ecart_hu_max": 90},
     "origine_seuil": "vérité terrain, 14 cas d'entraînement (sans 1019), max + 50 HU"}
  ],
  "regles_verifiees": ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"],
  "mesures": {"volumes_ml": {"1": 0.0}, "hu_medianes": {"3": 0.0}, "morceaux": {"6": 0}},
  "masque": "chemin du masque, inchangé",
  "avertissement": "Résultat hors du domaine validé : ne pas utiliser sans relecture."
}
```

*(Les chiffres de l'exemple sont fictifs. Ils montrent seulement la forme du fichier.)*

Chaque règle porte une **catégorie** :
- `erreur_probable` : le masque est probablement faux ;
- `hors_domaine` : le masque peut être juste, mais l'examen ne ressemble pas à ceux qui ont servi à la validation (exemple : AP plus dense que l'aorte).

Les deux catégories donnent le statut ALERTE. Dans le rapport, elles sont présentées **séparément** (Q8) : une alerte `hors_domaine` (R6, R7) est présentée comme une **limite du modèle** (D27), pas comme une erreur de segmentation ; une alerte `erreur_probable` signale un masque suspect.

## 5. Règles

Notations : « structure *k* » = voxels du masque prédit portant l'étiquette *k*. La *densité médiane* d'une structure est la médiane des HU de l'image dans cette structure. On la mesure sur le **cœur** de la structure, c'est-à-dire les voxels à plus de 3 mm du bord, comme dans le script 05d. On évite ainsi l'*effet de volume partiel* : sur les bords, un voxel mélange le sang et la paroi, ce qui fausse sa densité. Si le cœur est vide (structure trop fine), on prend la médiane sur toute la structure et on le note.

Une *composante connexe* (ou « morceau ») est un bloc de voxels qui se touchent. On utilise la connexité 26, comme les scripts 04b et 05d : deux voxels se touchent s'ils partagent une face, une arête ou un coin.

| N° | Règle | Ce qu'elle détecte | Comment la mesurer | D'où vient le seuil | Catégorie |
|---|---|---|---|---|---|
| R0 | Cohérence technique | Masque inutilisable : forme ou affine différente de l'image, étiquette hors de 0 à 7 | Comparer forme et affine (tolérance 0,01 mm), lister les valeurs présentes | Fixé, sans calibrage | erreur_probable (bloquante) |
| R1 | Présence des étiquettes 1 à 7 | Structure oubliée | Volume de chaque étiquette > *V*min,abs = 1 mL, pour ignorer quelques voxels égarés | Fixé (1 mL). Toutes les structures sont présentes dans la vérité des 15 cas | erreur_probable |
| R2 | Volume de chaque structure dans la plage observée | Structure beaucoup trop petite ou trop grande, par exemple l'aorte de 1019 réduite à un fragment | Volume en mL, comparé à [min × (1 − *m*), max × (1 + *m*)] | Vérité des 14 cas d'entraînement sans 1019 (sections 6 et 6.4). Marge *m* = **50 %** (D35 ; 30 % en v1.0, Q2) | erreur_probable |
| R3 | Rapport de volumes **AP / aorte** (D35) | Même chose que R2, mais sans dépendre de la taille du patient ni de l'espacement des voxels, par exemple une « aorte » réduite à un fragment. Les rapports divisés par la cavité VG (aorte, OG, VD, myocarde) sont supprimés en v1.1 : le volume de la cavité VG dépend de la phase du cycle cardiaque (D32) | Rapport des nombres de voxels, dans une même image | Vérité des 14 cas sans 1019 : [min × (1 − *m*), max × (1 + *m*)], *m* = 30 % (Q2) | erreur_probable |
| R4 | Densité de l'aorte proche de celle de la cavité VG | L'« aorte » prédite n'est pas l'aorte. Le sang de l'aorte ascendante est celui qui sort du VG : leurs densités doivent être proches, quelle que soit la phase | \|HU médiane aorte prédite − HU médiane cavité VG prédite\| ≤ *S*4 | Vérité des 14 cas sans 1019 : écart maximal aorte/VG (sorties de `05e_densites_vaisseaux.py`), plus 50 HU (Q2) | erreur_probable |
| R5 | Densité de l'AP proche de celle du VD | Symétrique de R4 pour le circuit droit : le sang de l'AP est celui qui sort du VD | \|HU AP prédite − HU VD prédit\| ≤ *S*5 | Vérité des 14 cas sans 1019 (05e, qui mesure aussi le VD : colonne `AP_moins_VD_ref`), plus 50 HU (Q2) | erreur_probable |
| R6 | Aorte pas moins dense que l'AP | **Phase d'injection hors domaine** : AP plus dense que l'aorte, comme sur 1019. Cette règle peut se déclencher sur un masque **correct** : elle signale l'examen, pas le masque | HU aorte prédite − HU AP prédite ≥ 0 | **Fixé à 0 HU** (D35) : c'est la définition « INVERSÉ » de 05e (médiane aorte < médiane AP), écrite avant les résultats de la section 18. Non calibré (en v1.0 : minimum observé − 50 HU, qui reposait sur un seul cas, 1006) | hors_domaine |
| R7 | Contraste présent | Scanner sans injection, ou injection ratée : ce cas est hors domaine, même si les contours sont justes | HU médiane cavité VG prédite ≥ *S*7 | Vérité des 14 cas sans 1019 : minimum observé, moins 50 HU (Q2) | hors_domaine |
| R8 | Morceaux | Prédiction éclatée : plusieurs morceaux, ou une étiquette posée à deux endroits (un bout d'AP appelé « aorte ») | Pour chaque étiquette : nombre de morceaux de plus de 0,5 mL. La part du plus gros morceau reste mesurée, pour information, mais ne déclenche plus rien (D35) | **Vérité des 14 cas sans 1019** (D34, qui remplace « prédictions hors-pli ») : *N* maximum observé ; règle **appliquée** aux prédictions hors pli post-traitées (D33). Sous-règle « part du plus gros ≥ *p* » supprimée (D35) : sans marge, elle signalait des fragments de quelques centièmes de mL, plus petits que les morceaux négligeables (0,5 mL). Limite : le myocarde MM-WHS est en plusieurs morceaux dans la vérité (10 sur 1003, D23), donc *N*(myocarde) peut être élevé | erreur_probable |
| R9 *(version 2, hors version 1 : Q1)* | Position relative et contacts | Étiquettes permutées ou posées au mauvais endroit : l'aorte ascendante doit toucher la cavité VG, l'AP doit toucher le VD, et l'AP est à gauche de l'aorte, plus en avant | Contact : voxels voisins entre deux étiquettes (oui/non). Position : écart des centres de gravité en mm, dans le repère RAS (axe 1 vers la droite du patient, axe 2 vers l'avant, axe 3 vers la tête) | Vérité des 15 cas : signe de l'écart (règle anatomique), pas de seuil fin | erreur_probable |

Remarques :
- **R2 contre R3.** Les volumes en mL varient avec la taille du patient et la phase cardiaque. 1001 et 1020 sont probablement le même patient à deux phases cardiaques (D20). On attendait des rapports de R3 qu'ils varient moins ; c'est faux pour ceux qui divisent par la cavité VG, dont le volume suit la phase cardiaque (30 à 150 mL sur la vérité, D32). En v1.1, R3 ne garde que AP / aorte, deux vaisseaux (D35). Les deux règles sont gardées (version 1 = R0 à R8, Q1).
- **Illustration seulement, sans valeur de calibrage.** Rapports tirés des voxels **prédits** du pli 0 (`summary.json`, champ `n_pred`) :

  | Cas | aorte / cavité VG | AP / aorte |
  |---|---|---|
  | 1001 | 2,11 | 1,08 |
  | 1020 | 1,13 | 1,06 |
  | 1019 | 0,055 | 8,8 |

  Sur la **vérité** de 1019, le rapport aorte / cavité VG vaut 1,28 (378 433 / 295 457). R2, R3 et R4 auraient donc probablement signalé 1019. Ces chiffres ont servi à concevoir les règles : ils ne peuvent pas les valider (section 6.4).
- **R6 relève de D28 plus que d'une erreur.** Sur 1019, la vérité respecte R4 (aorte de même densité que le VG, D27) mais pas R6 (aorte moins dense que l'AP). R6 se déclencherait donc même si le modèle avait bien segmenté 1019. C'est voulu : D28 demande de signaler tout examen hors domaine. Il reste un détail de mesure. Quand le modèle se trompe comme sur 1019, l'« aorte » et l'« AP » prédites tombent toutes les deux sur l'AP réelle, et leurs densités sont proches. R6 peut alors ne pas se déclencher, alors que R4 le fait. Les deux règles se complètent.
- **Version 1 = R0 à R8** (Q1). R9 est reportée à une version 2 : contacts et positions sont plus délicats à mesurer (voisinage voxel à voxel, orientation) et risquent des fausses alertes, alors que R2, R3 et R4 suffisent à signaler 1019.
- Marges fixées **avant** de voir les résultats de 05e (Q2) : *m* = 30 % pour les volumes et les rapports, 50 HU pour les densités. **Révisées une fois par D35, après la vérification** : 50 % pour les volumes (R2) ; R6 fixée à 0 HU, sans marge. Inchangées : 30 % pour R3 (AP / aorte), 50 HU pour R4, R5 et R7.
- **Mesures sur la vérité des 15 cas d'entraînement, sans aucun seuil** (description seulement ; le calibrage se fera selon la section 6) :
  - **densités (05e, section 18)** : 15 cas lus. 1019 est le **seul cas INVERSÉ** (rapport aorte / AP de 0,61) ; 1006 est « proche » (1,17). Densité de la cavité VG de 295 à 489 HU (relevé de la section 18). Sans 1019 : écart aorte − AP minimal de +65 HU (1006) et écart aorte − VG minimal de +27 HU (1015), donc aorte toujours plus dense que la cavité VG sur ces 14 cas. Attention, ce sont des relevés, pas les grandeurs des formules : R4 se calibrera sur l'écart **maximal** \|aorte − VG\|, R7 sur la vérité **sans 1019**, et R5 (AP − VD) n'est pas reprise ici ; les valeurs complètes sont dans le tableau de 05e (Drive) ;
  - **volumes (05h, section 21, D32)** : 15 cas mesurés. Volumes très variables (cavité VG de 30 à 150 mL), probablement selon la phase du cycle cardiaque ; myocarde VG de 114 mL en médiane. Rapport aorte / cavité VG de 1019 : 1,281. Volume minimal de l'aorte observé : 29 mL, à comparer aux 2,2 mL de l'aorte prédite de 1019 au pli 0 : R2 et R3 ne détecteront que les erreurs grossières, ce qui est leur rôle (D32).

## 6. Méthode de calibrage

### 6.1 Données autorisées

Seuls les **15 cas d'entraînement** du découpage figé (`splits/mmwhs_split.json`) servent au calibrage ; 1019 est mesuré mais exclu du calcul des seuils (D28, section 6.4). **Aucun cas de test** (1004, 1010, 1011, 1013, 1014) n'est lu, ni pour les seuils, ni pour choisir les règles.

### 6.2 Vérité terrain ou prédictions hors-pli : lesquelles, et pourquoi

| Source | Avantages | Limites | Usage retenu (Q9) |
|---|---|---|---|
| **Vérité terrain** des 15 cas | Densités (R4 à R7) mesurées par 05e ; volumes et rapports (R2, R3) mesurés par 05h (`scripts/05h_volumes_verite.py`, section 21 du notebook), avec et sans 1019. Elle décrit l'anatomie et les densités **plausibles** | Elle ne contient **aucune erreur typique du modèle** : morceaux égarés, bords baveux. Le myocarde y est fragmenté (îlots d'annotation) | Volumes (R2), rapport AP / aorte (R3), densités (R4, R5, R7 ; R6 est fixée, D35), morceaux (R8, D34), positions (R9, version 2) : ce sont des propriétés de l'anatomie et de l'examen |
| **Prédictions hors-pli** (*out-of-fold* : chaque cas est prédit par le pli qui ne l'a pas vu à l'entraînement) | Elles reflètent le comportement réel du modèle, et donc ce qu'est un « masque normal du modèle » | Elles n'existent que pour le pli 0 (3 cas). Il faut les **5 plis** pour en avoir 15. Elles contiennent l'échec de 1019 | Vérification des faux positifs de toutes les règles (R8 : calibrée sur la vérité depuis D34) |

**Calendrier retenu (Q9) :**
1. **Juste après la section 18 du notebook** : calibrer R2 à R7 sur la **vérité** des cas d'entraînement (1019 exclu des seuils). R4 à R7 viennent de 05e ; R2 et R3 viennent de 05h (volumes des étiquettes 1 à 7 et 5 rapports, section 21), qui ne fixe aucun seuil. *Depuis D35 : R6 fixée à 0 HU, non calibrée ; R3 réduite au rapport AP / aorte.* Ces règles décrivent ce qui est anatomiquement plausible. La vérité est la bonne référence, et elle ne dépend pas du modèle.
2. **Après les 5 plis** : calibrer R8 sur la **vérité** des 14 cas sans 1019 (D34 ; remplace « sur les prédictions hors-pli »), puis vérifier toutes les règles sur les 15 prédictions hors pli **post-traitées** (D33 ; critère de la section 7).
3. **Geler** le fichier de seuils, puis le tracer dans le journal (date, empreinte SHA-256). Tout cela **avant** l'étape 6. **Fait le 2026-11-14 (D36)** : `config/seuils_plausibilite.json`, SHA-256 `c1b247070765d193a46e90b538cc655787dbb65f8fdd13f113a39efd3ef90672`.

**Rangement (Q6) :** `config/seuils_plausibilite.json`, **dans le dépôt**. Il ne contient que des valeurs agrégées (plages, marges, version, date de gel), **aucune valeur par cas** et aucun identifiant de cas (pas de champ « cas du min / cas du max »). Le tableau cas par cas de 05e reste dans Drive. Git date le fichier, ce qui prouve qu'il a été figé avant l'étape 6.

### 6.3 Marge et statistiques

Avec 14 cas de calibrage (sans 1019), soit 12 patients, les percentiles ne veulent pas dire grand-chose : le 5e percentile de 12 valeurs est presque le minimum. Retenu (Q2) :
- **volumes (R2)** : [min × (1 − *m*), max × (1 + *m*)], avec *m* = 50 % (D35 ; 30 % en v1.0) ;
- **rapport AP / aorte (R3)** : même formule, *m* = 30 % ;
- **écarts de densité** : maximum observé + 50 HU, qui couvre le bruit de mesure (pour R7 : minimum observé − 50 HU). R6 : seuil fixe de 0 HU (D35) ;
- si la validation en laissant un patient de côté produit malgré tout une fausse alerte, la marge est élargie et la modification est consignée dans le journal, **avant** le gel ;
- **morceaux** : « au plus *N* morceaux de plus de 0,5 mL ». *N* (maximum) est lu, étiquette par étiquette, sur la **vérité** des 14 cas sans 1019, sans marge (D34). La sous-règle « plus gros morceau ≥ *p* % du volume » est supprimée (D35).

**Paires d'un même patient** (1001/1020, 1005/1012) : elles comptent pour un seul patient. Un min/max n'en souffre pas, mais une moyenne ou un écart-type si. Dans la validation « en laissant un patient de côté » (section 6.4), on retire **les deux cas de la paire** ensemble.

**Familles d'épaisseur de coupe** (0,45 mm : 1001/1020, 1003, 1016 ; 0,625 mm : les autres) : les volumes en mL et les HU sont des grandeurs physiques. Ils ne devraient pas dépendre de l'épaisseur de coupe. On ne calibre **pas** de seuils par famille : 3 patients à 0,45 mm, c'est trop peu. On **vérifie** en revanche qu'aucune règle ne sépare les deux familles, par exemple un bruit plus fort qui décalerait les médianes. Si c'est le cas, on le signale.

### 6.4 Risque de circularité (à lire attentivement)

**Le problème.** Les règles ont été **imaginées après avoir vu l'échec de 1019**. Si 1019 sert aussi à fixer les seuils, puis à « prouver » que le contrôle marche, la démonstration tourne en rond. On trouvera toujours un seuil qui sépare un cas connu des autres.

**Méthode retenue (D28) :**
1. **Calibrer sans 1019.** Les seuils sont tirés des 14 autres cas d'entraînement, soit 12 patients. 1019 sert seulement à la vérification.
2. **Validation en laissant un patient de côté** (*leave-one-patient-out*). Pour chaque patient, on recalcule les seuils sans lui (sans les deux cas pour une paire), puis on applique le contrôle à ce patient. On mesure ainsi combien de cas réussis seraient signalés à tort par des seuils qui ne les ont pas vus.
3. **Le dire dans le rapport.** Signaler 1019 est une **vérification de cohérence**, pas une preuve d'efficacité. Avec un seul échec connu, on ne peut pas mesurer la *sensibilité*, c'est-à-dire la part des échecs que le contrôle attrape. Le choix des règles reste lui-même inspiré par 1019. Ce biais ne disparaît pas, il est seulement documenté.

**Cas particulier de R4 et R6.** La vérité de 1019 respecte R4 (aorte ≈ VG) : l'inclure ne change presque rien au seuil de R4. En revanche, elle **viole** R6 (aorte moins dense que l'AP). En v1.0, R6 était calibrée : inclure 1019 aurait élargi la plage jusqu'à accepter sa phase. C'est la raison principale pour laquelle 1019 est **exclu du calibrage de toutes les règles** (D28). Depuis D35, R6 n'est plus calibrée : son seuil fixe (0 HU) vient de la définition « INVERSÉ », écrite avant les mesures. Elle reste inspirée par 1019 : le domaine validé est défini comme « aorte au moins aussi dense que l'AP ».

**Révision après la vérification (D35).** Les règles R3, R6 et R8 et la marge de R2 ont été révisées **une fois**, après avoir vu la vérification sur les cas d'entraînement. C'est un biais de plus, assumé : les seuils v1.1 ne sont plus indépendants de ces cas. Il est borné par trois choix : une seule révision, motivée par des causes de conception (pas par un cas à faire passer), et aucune autre ensuite (section 7.1). Seul le test (étape 6) donnera une estimation indépendante des fausses alertes.

## 7. Critère de validation du contrôle

Le contrôle est **validé** si, avec des seuils calibrés comme en section 6 :

| Exigence | Cas | Quand |
|---|---|---|
| 1019 → ALERTE | prédiction du pli 0 | maintenant |
| 1001 et 1020 → OK | prédictions du pli 0 | maintenant |
| Fausses alertes sur les 14 cas autres que 1019 : objectif zéro, au plus une tolérée si elle est expliquée et consignée dans le journal (Q3). Une alerte qui correspond à une **erreur du modèle classée en D31 avant tout calibrage** est une **vraie** alerte (D35) : 1018, si R8 se déclenche sur son AP | prédictions hors-pli des 5 plis (1001 et 1020 du pli 0, les 12 autres des plis 1 à 4) | après les 5 plis |
| Aucune fausse alerte en *leave-one-patient-out* sur la vérité des 15 cas, règles R0 à R8 (R8 depuis D34), sauf 1019 pour R6 | vérité | maintenant |

### 7.1 Révision unique (D35)

La première exécution de 05j (seuils v1.0) n'a pas atteint le critère Q3, pour quatre causes de conception (D35) : la part minimale du plus gros morceau (R8) sans marge ; le seuil de R6 dépendant d'un seul cas ; des rapports R3 divisés par la cavité VG, qui suit la phase cardiaque ; une marge de 30 % trop étroite pour R2 avec 13 patients. S'y ajoutait le décompte de 1018 comme fausse alerte, alors que son fragment de l'AP est une erreur du modèle classée en D31.

Règle pour la suite : **si le critère n'est toujours pas atteint avec la version 1.1, on ne révise plus.** Les seuils sont figés tels quels et les limites restantes sont consignées dans le journal et au rapport. Cela remplace, pour ce contrôle, « au-delà, la règle en cause est revue ou retirée » (Q3).

Pour 1018 : une alerte par d'autres règles seulement reste une fausse alerte. Si R8 ne signale pas l'AP de 1018 (par exemple si *N*(AP) ≥ 2 sur la vérité), c'est une limite à consigner (D34), pas un échec du critère.

### 7.2 Résultat et gel (D36)

Section 23 relancée avec la version 1.1 : critère **ATTEINT SOUS RÉSERVE**.

| Exigence | Résultat |
|---|---|
| 1019 → ALERTE | atteint : R2 (aorte), R3 (AP / aorte), R8 (AP) ; R6 (hors domaine) |
| 1001 et 1020 → OK | atteint |
| Fausses alertes sur les 14 cas autres que 1019 | **une**, consignée (Q3) : 1002, R8 sur l'AP, second morceau de 0,65 mL, à peine au-dessus du seuil de 0,5 mL ; cliniquement mineur, reste de la segmentation correct. 1018 : fragment de l'AP détecté par R8, vraie alerte (D35) |
| Vérification par patient sur la vérité | atteint : aucune alerte ; vérité de 1019 : R6 seule |

Autres résultats : *N*(AP) = 1, donc R8 peut détecter le fragment de 1018 ; concordance avec 05e et 05h : 240 valeurs, 0 écart. Vérification par patient sur les **prédictions** (information) : 2 fausses alertes sur 14 (1001 : R2 sur l'AP ; 1002 : R8 sur l'AP). C'est l'**estimation indicative** des fausses alertes sur des cas nouveaux : **environ 2 sur 14**.

**Gel** : `config/seuils_plausibilite.json` est la copie de `seuils_proposes.json` (05j, code `f95bc7b`), avec trois changements seulement : statut « figé », date de gel 2026-11-14, champ `sha256_du_fichier_gele` retiré (un fichier ne peut pas contenir sa propre empreinte). Les 45 valeurs de seuil et de marge sont identiques. Empreinte SHA-256 du fichier : `c1b247070765d193a46e90b538cc655787dbb65f8fdd13f113a39efd3ef90672`. Le fichier est exempté de conversion des fins de ligne (`.gitattributes`), pour que l'empreinte soit la même partout.

Interdits :
- **jamais** de calibrage, de réglage ni d'essai sur le lot de test ;
- **jamais** de modification des seuils après le gel.

**À l'étape 6**, le contrôle est appliqué aux prédictions du test avec les seuils **figés avant**, sur le modèle de D23 (règles fixées avant tout calcul). Le statut de chaque cas est rapporté, quel qu'il soit. Si un seuil se révèle mauvais sur le test, on le note comme limite. On ne le corrige pas.

Le test ne contient que 5 cas. S'ils sont tous en phase « normale », il ne dira rien de la capacité du contrôle à attraper un échec. Il dira seulement s'il produit de fausses alertes.

## 8. Comportement en cas d'alerte

- **Jamais de résultat silencieux.** Le statut ALERTE s'affiche **en première ligne** de la sortie du script, **en tête** du rapport du cas, et dans **MLflow** (étiquette `controle_plausibilite = ALERTE`, nombre de règles déclenchées, JSON du contrôle joint comme fichier).
- **Le masque reste disponible**, inchangé, avec son fichier JSON de contrôle à côté. On ne le renomme pas, pour ne pas casser les outils nnU-Net, mais tout rapport qui l'affiche reprend l'avertissement. Rien n'est supprimé, rien n'est corrigé automatiquement.
- **Métriques de l'étape 6** (validé par l'utilisateur, Q5) :
  1. métriques principales (D23) calculées sur **tous** les cas du test, alertes comprises ;
  2. **nombre de cas en alerte**, et règles déclenchées, rapportés à côté ;
  3. en complément, métriques **sans** les cas en alerte, clairement libellées, jamais présentées comme le résultat principal.

  Exclure les alertes du chiffre principal embellirait le verdict. Les inclure seules masquerait l'intérêt du contrôle. Rapporter les deux, avec le décompte, reste transparent.

## 9. Limites

- **15 cas, dont 13 patients** : les plages observées sont étroites et les seuils fragiles. Un patient normal mais plus grand, ou un cœur pathologique, peut sortir de la plage (fausse alerte).
- **Un seul échec connu** (1019) : on ne peut pas estimer la sensibilité. Un autre type d'erreur (bord mal placé, structure de bon volume mais décalée) peut passer (faux négatif).
- **Les règles ont été conçues après avoir vu 1019** : c'est un biais de conception, documenté en section 6.4.
- **Les règles ont été révisées une fois après la vérification** (D35) : les fausses alertes mesurées sur les cas d'entraînement sont donc optimistes. Estimation indicative avant le test : environ 2 sur 14 (D36). Le test en donnera une estimation indépendante, sur 5 cas seulement.
- **R3 réduite à AP / aorte, R8 sans critère de part** (D35) : moins de façons de détecter une erreur. Par exemple, une OG ou un VD grossièrement faux en proportion de la cavité VG n'est plus repéré que par R2, dont la plage est large (marge de 50 %).
- **Le contrôle porte sur la vraisemblance, pas sur l'exactitude.** Un masque plausible peut être faux de quelques millimètres. Le contrôle ne remplace ni le Dice sur données annotées, ni la relecture d'un expert.
- **Mesures dépendantes du masque.** R4 à R7 mesurent les densités dans le masque prédit. Si le modèle se trompe de façon cohérente (deux structures permutées ensemble), certaines règles peuvent rester muettes. C'est la raison d'être de R9, prévue en version 2.
- **Hors domaine ≠ faux.** R6 et R7 signalent des examens, pas des erreurs : elles produisent des alertes sur des masques peut-être justes. C'est voulu (D28). Le rapport les présente comme des limites du modèle (Q8).
- **Version 1 sans R9** : une permutation cohérente de deux étiquettes peut passer inaperçue si volumes et densités restent plausibles.

## 10. Décisions de l'utilisateur sur les questions ouvertes (D29)

Tranchées le 2026-11-11 : toutes les recommandations ont été suivies, sans modification.

| Question | Décision |
|---|---|
| **Q1.** R9 (positions et contacts) dès la version 1 ? | **Non.** Version 1 = R0 à R8. R9 est reportée à une version 2 : plus délicate à mesurer, risque de fausses alertes, et R2, R3, R4 suffisent pour 1019 |
| **Q2.** Marges ? | **30 %** pour les volumes et les rapports, **50 HU** pour les densités, fixées avant de voir les résultats de 05e. Élargies seulement si la validation par patient laissé de côté produit une fausse alerte, avec une ligne dans le journal. *Modifié par D35 : 50 % pour R2 ; R6 fixée à 0 HU* |
| **Q3.** Fausses alertes acceptées sur les 14 cas réussis ? | **Objectif zéro, au plus une**, expliquée et consignée. Au-delà, la règle en cause est revue ou retirée. *Modifié par D35 : une seule révision (R3, R6, R8, marge de R2) ; une alerte sur une erreur du modèle classée en D31 (1018) est une vraie alerte ; ensuite, plus de révision : gel et limites consignées (section 7.1)* |
| **Q4.** Mesure des densités | *(Réglée avant.)* Même fonction `intensites()` que 05d, cœur à plus de 3 mm du bord, repli sur la région entière ; 05e mesure aussi le VD |
| **Q5.** Cas en alerte dans les métriques de l'étape 6 | Métriques principales sur **tous** les cas, alertes comprises ; nombre d'alertes et règles déclenchées rapportés à côté ; métriques sans les alertes en complément seulement, clairement libellées |
| **Q6.** Où ranger les seuils figés ? | **Dans le dépôt**, `config/seuils_plausibilite.json` : valeurs agrégées seulement, aucune valeur ni identifiant par cas. Le tableau cas par cas reste dans Drive |
| **Q7.** Contrôle des masques TotalSegmentator à l'étape 6 ? | **Oui, restreint** aux structures 1 à 5, plus R0 et R7 ; aucune règle sur l'aorte ni l'AP (définitions différentes, D23). Alertes rapportées à part, à titre d'information |
| **Q8.** R6 comme limite du modèle ? | **Oui.** Les alertes `hors_domaine` (R6, R7) sont présentées comme des limites du modèle (D27), séparément des alertes `erreur_probable` |
| **Q9.** Calendrier de calibrage | **En deux temps** : R2 à R7 sur la vérité juste après la section 18 (version 1) ; R8 et vérification des fausses alertes sur les prédictions hors-pli après les 5 plis. Tout est figé avant l'étape 6. *Modifié ensuite : calibrage reporté après la décision sur le post-traitement (D32) ; R8 calibrée sur la vérité, appliquée aux prédictions post-traitées (D34)* |

---

## 11. Phase 2 : estimation de la phase d'injection (contrôle d'entrée)

*Hors périmètre de la phase 1. Cette section est un état des lieux pour préparer la phase 2.*

### 11.1 Ce que TotalSegmentator propose

**Vérifié** (README du dépôt GitHub `wasserth/TotalSegmentator`, code de `totalseg_get_phase.py`, page `resources/contrast_phase_prediction.md`, consultés le 2026-11-11, branche `master`) :

- **Commande :** `totalseg_get_phase -i ct.nii.gz -o contrast_phase.json`. Elle demande le paquet `xgboost`.
- **Méthode, en deux temps :**
  1. TotalSegmentator lance sa tâche principale `total` en mode rapide (`fast=True`, résolution 3 mm). Il segmente 16 structures : foie, pancréas, vessie, vésicule biliaire, cœur, aorte, veine cave inférieure, veine porte et veine splénique, veines et artères iliaques droites et gauches, veines pulmonaires, cerveau, côlon, intestin grêle. Si le cerveau est visible (volume > 100), la tâche `headneck_bones_vessels` ajoute 4 vaisseaux du cou (carotides internes, jugulaires internes). Sinon, ces valeurs sont mises à 0.
  2. La **densité médiane (HU)** de chaque structure alimente un classifieur *XGBoost* (une méthode d'apprentissage à base d'arbres de décision). Il s'agit d'un ensemble de 5 modèles, fichier `contrast_phase_classifiers_2024_07_19.pkl` livré avec le paquet. Il prédit le **temps écoulé depuis l'injection** (`pi_time`, en secondes), puis le convertit en phase.
- **Sorties (JSON) :** `pi_time`, `phase`, `probability`, `pi_time_min`, `pi_time_max`, `stddev`. L'écart-type entre les 5 modèles sert d'indice de confiance : faible = les modèles sont d'accord.
- **Phases reconnues :** `native` (sans contraste, < 10 s), `arterial_early` (10 à 30 s), `arterial_late` (30 à 60 s), `portal_venous` (≥ 60 s). La phase « tardive » (*delayed*) est **désactivée** dans le code, faute de données d'entraînement suffisantes selon le commentaire de l'auteur.
- **Performances annoncées** (validation croisée en 5 plis, page du dépôt) : erreur absolue moyenne de 5,55 s. Score F1 : 0,980 en natif, 0,915 en artériel (précoce et tardif regroupés), 0,940 en portal. (Le F1 combine la part des cas détectés et la part des détections justes : 1 = parfait.)
- **Données d'entraînement :** « entraîné sur le jeu TotalSegmentator ». La page ne donne ni le nombre d'examens, ni les régions du corps, ni l'origine des temps de référence.
- **Licence :** la tâche `total` et la tâche `headneck_bones_vessels` figurent dans la liste « *Openly available for any usage (Apache-2.0 license)* ». Le script `totalseg_get_phase.py` ne contient **aucune vérification de licence**. En revanche, il **envoie des statistiques d'usage** à la fin (`send_usage_stats_application("get_phase")`). Le README indique qu'on les désactive avec `send_usage_stats: false` dans `~/.totalsegmentator/config.json`. Il faut les désactiver comme le prévoit D21.

**Supposé (non vérifié) :**
- Le jeu TotalSegmentator (1 228 examens CT de routine clinique, hôpital universitaire de Bâle, selon l'article et le README) couvre surtout des scanners **thoraco-abdominaux et du corps entier**. On ne sait pas combien de **scanners cardiaques** (champ de vue serré sur le cœur) il contient. Il y en a probablement peu.
- Sur un CT cardiaque, la plupart des 16 structures (foie entier, pancréas, vessie, iliaques, côlon, cerveau) sont **absentes ou coupées par le bord du champ**. Les caractéristiques d'entrée du classifieur seraient alors manquantes ou faussées : l'examen serait hors du domaine du classifieur lui-même. Ce comportement n'a pas été testé.
- La liste des 16 structures ne contient pas l'**artère pulmonaire**. Elle mesure le cœur entier, l'aorte entière (ascendante et descendante) et les veines pulmonaires.

### 11.2 Pertinence pour notre problème

La distinction utile pour nous n'est **pas** « artériel précoce / tardif / portal ». C'est **« sang le plus dense dans l'aorte (circuit systémique, celui du corps) ou dans l'AP (circuit pulmonaire) »**. 1019 est probablement un examen pris trop tôt, quand le produit de contraste est encore dans le cœur droit et l'AP. Les phases de `totalseg_get_phase` ont été définies pour l'abdomen. Une phase « artérielle précoce » n'implique pas forcément « AP plus dense que l'aorte », et l'outil ne mesure pas l'AP.

**Conclusion (avis, à confirmer par un essai) :** `totalseg_get_phase` est **peu pertinent** tel quel pour le contrôle d'entrée d'un CT cardiaque. Il pourrait servir de repère grossier, par exemple pour détecter un examen **natif** (sans contraste), où son score F1 est le meilleur. Mais son comportement sur un champ cardiaque serré est inconnu. Pour la phase 2, une piste plus directe : mesurer soi-même les densités de l'aorte, de l'AP, du VG et du VD dans des régions simples, puis comparer aux plages des cas d'entraînement. C'est la logique des règles R4, R6 et R7, appliquée avant la segmentation.

**Essai possible, sans danger pour le protocole** (à décider en phase 2) : lancer `totalseg_get_phase` sur quelques cas d'**entraînement** (1001, 1019 et d'autres), jamais sur le test avant l'étape 6, pour voir s'il distingue 1019. Ce serait un essai, pas une validation.

### 11.3 Règle du projet et licence

- **Règle 5 de CLAUDE.md :** ne jamais entraîner sur des masques produits par TotalSegmentator. Utiliser une **phase estimée** comme contrôle d'entrée n'est pas entraîner : la règle n'est pas enfreinte, tant qu'aucun masque TotalSegmentator n'entre dans `nnunet_raw` ni dans une liste de données d'entraînement.
- **Licence :** `total` et `headneck_bones_vessels` sont sous Apache-2.0 d'après le README. Aucune licence académique n'est demandée pour `totalseg_get_phase`, contrairement à `heartchambers_highres` (étape 4, licence requise). Deux points restent **à vérifier** avant usage :
  1. la licence du fichier de poids XGBoost `contrast_phase_classifiers_2024_07_19.pkl`, que le README ne mentionne pas explicitement ;
  2. le contenu des statistiques d'usage. Elles sont à désactiver dans tous les cas (D21).

### 11.4 Sources

- README de TotalSegmentator (commandes, liste des tâches Apache-2.0 et sous licence, statistiques d'usage) : https://github.com/wasserth/TotalSegmentator/blob/master/README.md
- Code de la commande : https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/bin/totalseg_get_phase.py
- Description de la méthode et performances : https://github.com/wasserth/TotalSegmentator/blob/master/resources/contrast_phase_prediction.md
- Article TotalSegmentator (jeu d'entraînement) : Wasserthal et al., *Radiology: Artificial Intelligence* 2023, https://pubs.rsna.org/doi/10.1148/ryai.230024
- Travail indépendant fondé sur les mêmes caractéristiques (autre jeu, 4 phases dont « tardive ») : https://arxiv.org/html/2501.14066v2
- Application web de l'auteur (page vide lors de la consultation) : https://compute.totalsegmentator.com/contrast-phase/
