"""
Étape 5f — Bilan de la validation croisée nnU-Net : les 15 cas de validation des 5 plis dans un seul tableau.
LECTURE SEULE : aucun fichier nnU-Net n'est modifié ; le lot de test n'est jamais lu.

Contexte : chaque pli entraîné par 05c se termine par une « validation finale » : le modèle du pli prédit ses
3 cas de validation (jamais vus pendant son entraînement) et nnU-Net écrit les scores dans
fold_X/validation/summary.json. Chaque cas d'entraînement est en validation dans exactement UN pli : les
5 plis réunis donnent donc une prédiction « hors pli » pour chacun des 15 cas. Ce script les rassemble.

Ce que fait le script :
  0. garde-fous : splits_final.json vérifié comme dans 05b et 05c (écrit par 05a, identique aux plis
     recalculés depuis le découpage figé, lu seulement) ; --sortie hors du dépôt et des dossiers nnU-Net ;
     seuls sont lus : les summary.json et les prédictions (.nii.gz) de fold_X/validation, et, pour les
     mêmes cas, imagesTr et labelsTr (jamais imagesTs ni labelsTs, jamais un cas de test) ;
  1. pour chaque pli : lit summary.json s'il existe ; vérifie que ses cas sont EXACTEMENT les cas de
     validation du pli, qu'aucun n'est un cas de test et que la référence ne vient pas de labelsTs ;
     recalcule le Dice moyen du pli et le compare à celui écrit par nnU-Net ;
  2. tableau des 15 cas : Dice par structure (étiquettes 1 à 7) et moyenne par cas, avec le pli,
     l'épaisseur de coupe et le PATIENT (1001/1020 et 1005/1012 = même patient probable, D20) ;
  3. par structure : moyenne sur les cas, moyenne par PATIENT (les deux cas d'une paire sont d'abord
     moyennés : 13 patients pour 15 cas), médiane, écart-type, minimum et maximum (avec le cas),
     moyenne par épaisseur de coupe ;
  4. cas faibles signalés, d'après des repères INDICATIFS fixés à l'avance et non validés : cellule
     « faible » si Dice < 0,85, « ÉCHEC » si Dice < 0,50. Pour chaque cellule signalée : la part d'oublis
     et d'ajouts. Puis les 3 cas à la moyenne la plus basse, et la commande du script 05g pour les examiner ;
  5. LIMITES AP / VD (suite de D31), sur les cas de validation UNIQUEMENT : pour chaque cas, on lit la
     vérité (labelsTr) et la prédiction hors pli (fold_X/validation/<cas>.nii.gz) et on chiffre deux
     décalages, en mm le long de l'axe tête-pieds :
       - BAS de l'artère pulmonaire (étiquette 7) : niveau le plus bas prédit − niveau le plus bas annoté ;
       - HAUT du ventricule droit (étiquette 5) : niveau le plus haut prédit − niveau le plus haut annoté.
     Signe : POSITIF = la prédiction s'arrête plus haut (vers la tête) que l'annotation ; NÉGATIF = plus bas.
     Exemple : une AP prédite qui descend plus bas que l'annotation donne un décalage négatif.
     Calcul repris tel quel de 05d/05g (mêmes chiffres que la section 20) : charger() de 05d remet image,
     vérité et prédiction dans le repère RAS avec la transformation de l'IMAGE ; mesurer() de 05d prend la
     coupe la plus basse (ou la plus haute) qui contient au moins un voxel de la structure et la convertit
     en mm avec l'affine (coordonnée « tête » du repère du scanner, au droit du centre de gravité de la
     structure) ; les deux niveaux sont arrondis au dixième de mm, puis soustraits.
     TOUS les morceaux d'un masque comptent, y compris un fragment isolé : pour 1018, l'AP prédite est en
     3 morceaux, dont un isolé, et son niveau le plus bas est celui du morceau le plus bas (−69 mm dans 05g).
     Ce choix mesure ce que le modèle a réellement prédit ; le nombre de morceaux prédits est affiché à côté
     pour repérer ces cas (et la partie 5 bis recalcule le bas de l'AP sans les fragments).
     Si la structure est absente de la vérité OU de la prédiction d'un cas (ou si un fichier manque), ce
     cas n'a pas de décalage : sa ligne est SIGNALÉE et il est exclu des statistiques (le nombre de cas
     utilisés est affiché). Statistiques de chaque décalage : minimum et maximum (avec le cas), médiane,
     écart-type, moyenne et médiane des valeurs absolues, sur les cas puis PAR PATIENT (les deux cas d'une
     paire sont d'abord moyennés ; si un seul des deux a un décalage, il représente le patient).
  5 bis. BAS DE L'AP SUR LE PLUS GRAND MORCEAU, À CÔTÉ du calcul de la partie 5 (qui reste inchangé, tous
     morceaux comptés). But : séparer la variabilité de la limite elle-même de l'effet des fragments isolés
     (un petit fragment loin sous l'AP suffit à tirer le niveau le plus bas vers le bas).
       - on ne garde, dans la PRÉDICTION de l'AP (étiquette 7), que son plus grand morceau (plus grande
         composante connexe, connexité 26 : la MÊME que le comptage des morceaux de mesurer() dans 05d) ;
       - la VÉRITÉ n'est PAS modifiée : on compare toujours au niveau le plus bas de l'annotation ENTIÈRE
         (celui de la partie 5). Raison : l'annotation est la référence ; en retirer des morceaux reviendrait à
         changer la référence selon la prédiction, et le décalage ne mesurerait plus la même chose d'un cas à
         l'autre. On ne « nettoie » que le côté dont on veut mesurer l'effet des fragments ;
       - pour chaque cas : niveau le plus bas du plus grand morceau, décalage (prédit − annoté, même axe et même
         signe que la partie 5 : positif = vers la tête), volume du plus grand morceau (mL) et part du volume
         prédit de l'AP qu'il représente, nombre de morceaux. Le niveau est lu par mesurer() de 05d sur ce seul
         morceau (donc au droit de SON centre de gravité ; ne change rien si les axes ne sont pas inclinés) ;
       - statistiques comme en partie 5 (sur les cas puis par patient), et un tableau côte à côte : décalage
         tous morceaux / décalage plus grand morceau / nombre de morceaux ;
       - ce N'EST PAS le post-traitement de nnU-Net : aucune prédiction, aucun fichier n'est modifié ; le plus
         grand morceau est calculé EN MÉMOIRE, pour la mesure seulement. Le haut du VD n'a pas cette variante
         (son nombre de morceaux prédits est seulement rappelé dans le tableau côte à côte).
  Si des plis ne sont pas terminés, le bilan est PARTIEL (clairement marqué) et le code de sortie vaut 2.
  Le script NE CONCLUT PAS et ne fixe aucun seuil, ni pour le Dice (les repères ci-dessus sont indicatifs)
  ni pour les décalages, ni pour le contrôle de plausibilité (D28, D29).

Entrées :
  --raw           dossier nnUNet_raw (défaut /content/drive/MyDrive/poc_ct/nnunet_raw) ; sont lus dans
                  Dataset101_MMWHS : les en-têtes de imagesTr (épaisseurs de coupe, contrôle des plis), et,
                  pour les 15 cas de validation seulement, imagesTr et labelsTr (vérité). L'image est lue EN
                  ENTIER par charger() de 05d (environ 0,3 à 0,4 Go par cas, un cas à la fois, libérée aussitôt),
                  mais seul son repère (affine) sert aux décalages
  --preprocessed  dossier nnUNet_preprocessed (défaut .../poc_ct/nnunet_preprocessed) : splits_final.json,
                  sa fiche splits_final_05a.json et le marqueur verification_05a.json (lus seulement)
  --results       dossier nnUNet_results (défaut .../poc_ct/nnunet_results) ; seul
                  .../nnUNetTrainer_250epochs_reprise10__nnUNetPlans__3d_fullres/fold_X/validation/ est lu :
                  summary.json et les prédictions <cas>.nii.gz des cas de validation du pli
  --sortie        dossier de sortie (défaut /content/drive/MyDrive/poc_ct/analyses/05f_bilan)
  --split         découpage figé (défaut : ../splits/mmwhs_split.json à côté du script)
Sorties (uniquement dans --sortie) :
  bilan_validation_croisee.csv (une ligne par cas : Dice, puis niveaux et décalages AP / VD, puis les
                  colonnes bas_AP_plus_grand_morceau_* : niveau, décalage, volume en mL, part du volume prédit) ;
  synthese_05f.json (tableau, statistiques, signaux, et la rubrique « limites_AP_VD », qui contient la
                  sous-rubrique « bas_AP_plus_grand_morceau » ; les valeurs par cas sont dans « lignes »)
Codes de sortie :
  0 = les 5 plis sont rassemblés ; 1 = garde-fou (rien n'est écrit) ; 2 = bilan PARTIEL (plis non terminés,
  fichiers écrits quand même) ; 6 = un summary.json ne correspond pas aux plis, ou image, vérité et
  prédiction d'un cas n'ont pas la même forme (rien n'est écrit : ne pas utiliser ces résultats).

Vocabulaire :
  - Dice : recouvrement entre la prédiction et la vérité, de 0 (aucun) à 1 (parfait) ;
  - oubli (faux négatif) : voxel de la vérité que le modèle n'a pas mis dans la structure.
    Part d'oublis = oublis / voxels de la vérité ;
  - ajout (faux positif) : voxel prédit dans la structure alors que la vérité ne l'y met pas.
    Part d'ajouts = ajouts / voxels prédits ;
  - prédiction hors pli : prédiction d'un cas par le modèle qui ne l'a pas vu à l'entraînement ;
  - moyenne par patient : 1001 et 1020 sont deux examens du même patient (de même 1005 et 1012). Les
    compter comme deux cas donne deux fois plus de poids à ce patient : la moyenne par patient corrige cela ;
  - écart-type : dispersion des valeurs autour de la moyenne ;
  - axe tête-pieds : axe vertical du patient allongé (de la tête vers les pieds). Les niveaux sont en mm
    dans le repère du scanner : un niveau plus grand est plus près de la tête ;
  - repère RAS : on remet chaque volume dans le même sens (1er axe vers la droite du patient, 2e vers
    l'avant, 3e vers la tête), comme 05d ;
  - affine : petite matrice de l'en-tête NIfTI qui convertit un numéro de voxel en position en mm ;
  - médiane des valeurs absolues : médiane des décalages sans tenir compte de leur signe. Elle dit de
    combien la limite prédite s'écarte « typiquement » de l'annotation, quel que soit le sens ;
  - fragment isolé : morceau d'un masque séparé du reste (voir « morceaux » dans 05d) ;
  - composante connexe (ou « morceau ») : ensemble de voxels d'un masque qui se touchent de proche en
    proche. Connexité 26 : deux voxels se touchent s'ils partagent une face, une arête ou un coin (les 26
    voisins d'un voxel dans un cube 3 × 3 × 3) ;
  - plus grand morceau : la composante connexe qui contient le plus de voxels. Les autres morceaux (souvent
    de petits fragments isolés) sont ignorés pour cette mesure seulement.
"""
import argparse, csv, datetime, importlib.util, json, math, os, re, statistics, sys

# Fonctions reprises plutôt que réécrites (les noms de fichiers commencent par un chiffre : on les charge
# par leur chemin ; aucun n'exécute quoi que ce soit à l'import).
ICI = os.path.dirname(os.path.abspath(__file__))


def charger_module(nom, fichier):
    spec = importlib.util.spec_from_file_location(nom, os.path.join(ICI, fichier))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


essai = charger_module("essai05b", "05b_essai_nnunet.py")     # contrôle de splits_final.json (comme 05c)
ana = charger_module("analyse05d", "05d_analyser_1019.py")     # garde-fous de sortie, nom du trainer,
                                                               # charger() et mesurer() (niveaux en mm)
prep = essai.prep                                              # module du script 05a

# --- Paramètres fixes -------------------------------------------------------------------------
NOM_DS = ana.NOM_DS
STRUCTURES = {1: "myo_VG", 2: "OG", 3: "VG", 4: "OD", 5: "VD", 6: "aorte", 7: "AP"}   # étiquettes du script 03
SEUIL_FAIBLE, SEUIL_ECHEC = 0.85, 0.50   # repères INDICATIFS, fixés avant de voir les 5 plis, non validés
NB_PLUS_FAIBLES = 3
TOLERANCE_MOYENNE = 1e-4                 # écart toléré entre notre moyenne et celle de nnU-Net
NOM_CSV, NOM_SYNTHESE = "bilan_validation_croisee.csv", "synthese_05f.json"
# Limites mesurées (partie 5) : nom court, étiquette, mesure de mesurer() (05d), libellé lisible.
LIMITES = (("bas_AP", 7, "niveau_bas_mm", "bas de l'AP (7)"),
           ("haut_VD", 5, "niveau_haut_mm", "haut du VD (5)"))
# Partie 5 bis : la limite recalculée sur le plus grand morceau de la prédiction (bas de l'AP seulement).
CLE_PGM, BASE_PGM = "bas_AP_plus_grand_morceau", "bas_AP"


def arret(code, message):
    print(f"ARRÊT : {message}")
    sys.exit(code)


def lire_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=f"{prep.DRIVE}/nnunet_raw")
    p.add_argument("--preprocessed", default=f"{prep.DRIVE}/nnunet_preprocessed")
    p.add_argument("--results", default=f"{prep.DRIVE}/nnunet_results")
    p.add_argument("--sortie", default=f"{prep.DRIVE}/analyses/05f_bilan")
    p.add_argument("--split", default=os.path.join(prep.DEPOT, "splits", "mmwhs_split.json"))
    return p.parse_args()


# --- 0. Garde-fous (communs avec 05g) ---------------------------------------------------------
def verifier_sortie(sortie, *dossiers_nnunet):
    """--sortie ne doit être ni dans le dépôt, ni dans un dossier nnU-Net (même règle que 05d)."""
    sortie = os.path.abspath(sortie)
    interdits = [d for d in sortie.split(os.sep) if ana.DOSSIERS_INTERDITS.match(d)]
    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if os.environ.get(var) and prep.dans(sortie, os.path.abspath(os.environ[var])):
            interdits.append(f"${var}")
    if (prep.dans(sortie, prep.DEPOT) or any(prep.dans(sortie, os.path.abspath(d)) for d in dossiers_nnunet)
            or interdits):
        arret(1, f"--sortie ({sortie}) est dans le dépôt ou dans un dossier nnU-Net ({interdits}). "
                 "Choisir un dossier de /content/drive/MyDrive/poc_ct/analyses/.")
    return sortie


def lire_plis(raw, pre, split):
    """Plis vérifiés (splits_final.json = celui de 05a, sinon arrêt en code 1), strates et groupes."""
    try:
        cas, cas_test, groupes, _ = prep.lire_decoupage(split)          # lecture seule du découpage figé
    except (OSError, ValueError, KeyError) as err:
        arret(1, f"découpage figé illisible ou incohérent ({split}) : {err}")
    _, _, _, trace, plis = essai.verifier_prerequis_05a(raw, pre, NOM_DS, split, 0)
    print("(La ligne « Pli 0 » ci-dessus vient du contrôle commun avec 05c ; le bilan porte sur les 5 plis.)")
    return plis, trace["strates_mm"], groupes, set(cas), set(cas_test)


def dossier_pli(res, pli):
    return os.path.join(res, NOM_DS, f"{ana.TRAINER}__nnUNetPlans__{ana.CONFIGURATION}", f"fold_{pli}")


def nombre(x):
    """Valeur numérique, ou None si absente ou NaN (nnU-Net écrit NaN quand la structure est vide dans
    la vérité ET dans la prédiction : le Dice n'est alors pas défini)."""
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


# --- 1. Lecture d'un summary.json -------------------------------------------------------------
def lire_summary(chemin, val_attendus, cas_test):
    """Renvoie ({cas: {étiquette: {dice, n_ref, n_pred, TP, FP, FN}}}, Dice moyen écrit par nnU-Net,
    problèmes). Un problème = le fichier ne correspond pas au pli : ses résultats ne doivent pas servir."""
    try:
        with open(chemin, encoding="utf-8") as f:
            s = json.load(f)                   # json accepte NaN, écrit par nnU-Net
    except (OSError, ValueError) as err:       # ex. fichier tronqué (copie Drive en cours)
        return {}, None, [f"{chemin} illisible : {err}"]
    cas, pb = {}, []
    for entree in s.get("metric_per_case", []):
        nom = os.path.basename(entree.get("prediction_file", ""))
        c = nom[:-len(".nii.gz")] if nom.endswith(".nii.gz") else nom
        ref = entree.get("reference_file", "")
        if c in cas_test or any(d in ("imagesTs", "labelsTs") for d in re.split(r"[\\/]", ref)):
            pb.append(f"{c} : cas de TEST ou référence lue dans un dossier de test ({ref})")
        if c in cas:
            pb.append(f"{c} apparaît deux fois")
        m = {}
        for e in STRUCTURES:
            x = (entree.get("metrics") or {}).get(str(e))
            if x is None:
                pb.append(f"{c} : étiquette {e} absente du summary.json")
                continue
            m[e] = {"dice": nombre(x.get("Dice")),
                    **{k: int(round(float(x[k]))) for k in ("n_ref", "n_pred", "TP", "FP", "FN") if k in x}}
        cas[c] = m
    if sorted(cas) != sorted(val_attendus):
        pb.append(f"cas du summary.json {sorted(cas)} ≠ cas de validation du pli {sorted(val_attendus)}")
    return cas, nombre((s.get("foreground_mean") or {}).get("Dice")), pb


def moyenne_comme_nnunet(cas):
    """Même calcul que nnU-Net : moyenne sur les cas pour chaque structure (en ignorant les Dice non
    définis), puis moyenne de ces 7 moyennes."""
    par_structure = [statistics.fmean(v) for e in STRUCTURES
                     if (v := [m[e]["dice"] for m in cas.values() if e in m and m[e]["dice"] is not None])]
    return statistics.fmean(par_structure) if par_structure else None


# --- 2 et 3. Tableau et statistiques ----------------------------------------------------------
def construire_lignes(resultats, strates, groupes):
    patient_de = {c: "+".join(x[-4:] for x in g) for g in groupes for c in g}
    lignes = []
    for pli in sorted(resultats):
        for c in sorted(resultats[pli]):
            m = resultats[pli][c]
            dice = {e: (m[e]["dice"] if e in m else None) for e in STRUCTURES}
            definis = [d for d in dice.values() if d is not None]
            lignes.append({"pli": pli, "cas": c, "patient": patient_de.get(c, c[-4:]),
                           "epaisseur_mm": strates.get(c), "dice": dice,
                           "moyenne_cas": statistics.fmean(definis) if definis else None, "detail": m})
    return lignes


def resumer(paires):
    """paires = [(nom, valeur)] ; statistiques sur les valeurs définies."""
    v = [(n, x) for n, x in paires if x is not None]
    if not v:
        return None
    xs = [x for _, x in v]
    mini, maxi = min(v, key=lambda t: t[1]), max(v, key=lambda t: t[1])
    return {"n": len(xs), "moyenne": statistics.fmean(xs), "mediane": statistics.median(xs),
            "ecart_type": statistics.stdev(xs) if len(xs) > 1 else None,
            "min": mini[1], "cas_min": mini[0], "max": maxi[1], "cas_max": maxi[0]}


def statistiques(lignes):
    """Par structure (et pour la moyenne par cas) : sur les cas, par patient, par épaisseur de coupe."""
    colonnes = {**{STRUCTURES[e]: (lambda l, e=e: l["dice"][e]) for e in STRUCTURES},
                "moyenne_cas": lambda l: l["moyenne_cas"]}
    patients = {}
    for l in lignes:
        patients.setdefault(l["patient"], []).append(l)
    stats = {}
    for nom, lire in colonnes.items():
        par_patient = []                       # (patient, moyenne de ses cas, épaisseur)
        for p, ls in patients.items():
            v = [lire(l) for l in ls if lire(l) is not None]
            par_patient.append((p, statistics.fmean(v) if v else None, ls[0]["epaisseur_mm"]))
        epaisseurs = sorted({l["epaisseur_mm"] for l in lignes}, key=str)
        stats[nom] = {"sur_les_cas": resumer([(l["cas"][-4:], lire(l)) for l in lignes]),
                      "par_patient": resumer([(p, x) for p, x, _ in par_patient]),
                      "par_epaisseur": {str(ep): resumer([(l["cas"][-4:], lire(l)) for l in lignes
                                                          if l["epaisseur_mm"] == ep]) for ep in epaisseurs},
                      # Même chose en comptant chaque patient une fois (la paire 1001/1020 est à 0,45 mm).
                      "par_epaisseur_patients": {str(ep): resumer([(p, x) for p, x, e in par_patient if e == ep])
                                                 for ep in epaisseurs}}
    return stats


# --- 4. Cas faibles ---------------------------------------------------------------------------
def part(a, b):
    return None if not b else a / b


def signaler(lignes):
    signaux = []
    for l in lignes:
        for e, d in l["dice"].items():
            det = l["detail"].get(e, {})
            if d is None:
                if det.get("n_ref", 0) == 0:
                    signaux.append({"cas": l["cas"], "pli": l["pli"], "etiquette": e, "structure": STRUCTURES[e],
                                    "niveau": "ABSENTE de la vérité", "dice": None})
                continue
            if d < SEUIL_FAIBLE:
                signaux.append({"cas": l["cas"], "pli": l["pli"], "etiquette": e, "structure": STRUCTURES[e],
                                "niveau": "ÉCHEC" if d < SEUIL_ECHEC else "faible", "dice": d,
                                "part_oublis": part(det.get("FN", 0), det.get("n_ref")),
                                "part_ajouts": part(det.get("FP", 0), det.get("n_pred")),
                                "voxels_verite": det.get("n_ref"), "voxels_predits": det.get("n_pred")})
    return sorted(signaux, key=lambda s: (s["dice"] is not None, s["dice"] or 0))


def commandes_05g(lignes, signaux):
    """Pour chaque cas signalé : commande 05g avec ses structures signalées (4 au plus, les plus faibles
    d'abord) et un cas de COMPARAISON proposé : même épaisseur de coupe, autre patient, meilleur Dice moyen
    sur ces structures. Proposition seulement : on peut choisir un autre cas de comparaison."""
    par_cas = {}
    for s in signaux:
        if s["dice"] is not None:
            par_cas.setdefault(s["cas"], []).append(s)
    lignes_de = {l["cas"]: l for l in lignes}
    commandes = {}
    for c, ss in sorted(par_cas.items(), key=lambda t: min(s["dice"] for s in t[1])):
        etiq = [s["etiquette"] for s in sorted(ss, key=lambda s: s["dice"])][:4]
        l = lignes_de[c]

        def score(autre):
            v = [autre["dice"][e] for e in etiq if autre["dice"][e] is not None]
            return statistics.fmean(v) if v else -1

        candidats = [x for x in lignes if x["patient"] != l["patient"] and x["epaisseur_mm"] == l["epaisseur_mm"]
                     and x["cas"] not in par_cas]
        comp = max(candidats, key=score)["cas"][-4:] if candidats else None
        commandes[c] = (f"--cas {c[-4:]}{' ' + comp if comp else ''} --structures {' '.join(map(str, sorted(etiq)))}")
    return commandes


# --- 5. Limites AP / VD (cas de validation uniquement) ----------------------------------------
def chemins_cas(raw, res, pli, cas, cas_test):
    """Les 3 fichiers lus pour un cas de validation. Garde-fous : jamais un cas de test, jamais un
    dossier de test (même règle que 05d et 05g)."""
    if cas in cas_test:
        arret(1, f"{cas} fait partie du LOT DE TEST : refusé (le test ne sert qu'à l'étape 6).")
    ds = os.path.join(raw, NOM_DS)
    c = {"image": os.path.join(ds, "imagesTr", f"{cas}_0000.nii.gz"),        # lue en entier par charger() (05d) ; seul son repère sert
         "verite": os.path.join(ds, "labelsTr", f"{cas}.nii.gz"),
         "pred": os.path.join(dossier_pli(res, pli), "validation", f"{cas}.nii.gz")}
    if any(d in ("imagesTs", "labelsTs") for x in c.values() for d in re.split(r"[\\/]", x)):
        arret(1, "un chemin passe par imagesTs ou labelsTs : refusé.")
    return c


def sans_decalage(raison):
    """Rubrique d'une limite qu'on ne peut pas mesurer sur ce cas (le cas sera exclu des statistiques)."""
    return {"verite_mm": None, "prediction_mm": None, "decalage_mm": None,
            "morceaux_verite": None, "morceaux_prediction": None, "signal": raison}


def plus_grand_morceau(m):
    """Masque (même forme que m) du plus grand morceau de m, nombre de morceaux, et True si plusieurs
    morceaux ont exactement la taille maximale (on garde alors le premier trouvé par ndimage.label).
    Calcul EN MÉMOIRE seulement : rien n'est écrit, la prédiction lue n'est pas modifiée.
    Connexité 26 (structure 3 × 3 × 3 pleine), la MÊME que le comptage des « morceaux » de mesurer()
    dans 05d : le nombre de morceaux trouvé ici est donc celui affiché en partie 5.
    Pour information, le post-traitement de nnU-Net (nnunetv2/postprocessing/remove_connected_components.py)
    appelle remove_all_but_largest_component d'acvl_utils, qui utilise skimage.measure.label avec
    connectivity=None, c'est-à-dire la connexité complète, soit 26 en 3D : même connexité. Vérifié dans le
    code source de nnunetv2 2.8.1 et acvl_utils 0.2.7, PAS dans la version installée dans Colab. Autres
    différences : nnU-Net le teste d'abord sur l'union de toutes les étiquettes, puis étiquette par
    étiquette, et ne le garde que s'il améliore le Dice de la validation croisée ; ici on ne garde rien.
    Ex aequo : nnU-Net garde TOUS les morceaux de taille maximale égale ; ici, seul le premier (cas signalé « = »)."""
    import numpy as np
    from scipy import ndimage
    garde = np.zeros(m.shape, dtype=bool)
    if not m.any():
        return garde, 0, False
    # Recadrage sur la boîte englobante, comme mesurer() de 05d : moins de mémoire.
    bornes = [np.where(m.any(axis=tuple(x for x in range(3) if x != ax)))[0] for ax in range(3)]
    boite = tuple(slice(int(b.min()), int(b.max()) + 1) for b in bornes)
    etiq, nb = ndimage.label(m[boite], structure=np.ones((3, 3, 3)))   # connexité 26, comme 05d
    tailles = np.bincount(etiq.ravel())[1:]                             # taille de chaque morceau (voxels)
    i = int(np.argmax(tailles))                                         # 1er morceau de taille maximale
    garde[boite] = etiq == i + 1
    return garde, int(nb), bool((tailles == tailles[i]).sum() > 1)


def sans_plus_grand_morceau(raison, verite_mm=None):
    """Rubrique 5 bis d'un cas sans décalage (même exclusion que la partie 5)."""
    return {"verite_mm": verite_mm, "prediction_mm": None, "decalage_mm": None, "volume_ml": None,
            "volume_predit_total_ml": None, "part_volume_predit": None, "morceaux_prediction": None,
            "ex_aequo": None, "signal": raison}


def limite_plus_grand_morceau(pred, affine, esp, base):
    """Partie 5 bis pour un cas. base = rubrique « bas_AP » de la partie 5 (vérité ENTIÈRE, inchangée) :
    son niveau annoté est réutilisé tel quel. Seule la prédiction est réduite à son plus grand morceau."""
    if base["decalage_mm"] is None:                     # structure absente : même exclusion qu'en partie 5
        return sans_plus_grand_morceau(base["signal"], base["verite_mm"])
    etiquette, mesure = next((e, ms) for cle, e, ms, _ in LIMITES if cle == BASE_PGM)
    masque = pred == etiquette                          # copie booléenne : pred n'est pas modifié
    garde, nb, ex_aequo = plus_grand_morceau(masque)
    mg = ana.mesurer(garde, affine, esp)                # même mesure que la partie 5, sur ce seul morceau
    total = ana.mesurer(masque, affine, esp)            # pour le volume prédit total (mêmes arrondis)
    return {"verite_mm": base["verite_mm"], "prediction_mm": mg[mesure],
            "decalage_mm": round(mg[mesure] - base["verite_mm"], 1),       # même arrondi que la partie 5
            "volume_ml": mg["volume_ml"], "volume_predit_total_ml": total["volume_ml"],
            "part_volume_predit": round(mg["voxels"] / total["voxels"], 4),
            "morceaux_prediction": nb, "ex_aequo": ex_aequo,
            "signal": None}


def mesurer_limites(lignes, raw, res, cas_test):
    """Ajoute à chaque ligne (= un cas de validation) la rubrique « limites » : niveau annoté, niveau
    prédit, décalage (prédit − annoté, en mm, + = vers la tête), nombre de morceaux, signal éventuel ;
    et la rubrique CLE_PGM (partie 5 bis : bas de l'AP sur le plus grand morceau de la prédiction)."""
    for l in lignes:
        c = chemins_cas(raw, res, l["pli"], l["cas"], cas_test)
        manquants = [x for x in c.values() if not os.path.isfile(x)]
        if manquants:
            l["limites"] = {cle: sans_decalage(f"fichier absent : {manquants}") for cle, *_ in LIMITES}
            l[CLE_PGM] = sans_plus_grand_morceau(f"fichier absent : {manquants}")
            l["remarques_geometrie"] = []
            continue
        print(f"  {l['cas']} (pli {l['pli']}) : lecture de la vérité et de la prédiction...", flush=True)
        # Même lecture que 05d et 05g : les trois volumes remis dans le repère RAS avec la transformation
        # de l'image. Si les formes diffèrent, charger() arrête le script (code 6, rien n'est écrit).
        ct, ver, pred, affine, esp, geo = ana.charger(c)
        del ct                                              # l'image elle-même ne sert pas ici
        remarques = list(geo["anomalies"])                  # ex. affine de la vérité ≠ affine de l'image
        if abs(float(affine[2, 0])) + abs(float(affine[2, 1])) > 1e-3:
            remarques.append("axes du volume inclinés par rapport à l'axe tête-pieds : niveau lu au droit du "
                             "centre de gravité de la structure, donc légèrement approché")
        limites = {}
        for cle, etiquette, mesure, _ in LIMITES:
            # mesurer() de 05d : TOUS les voxels de l'étiquette comptent, fragments isolés compris.
            mv, mp = ana.mesurer(ver == etiquette, affine, esp), ana.mesurer(pred == etiquette, affine, esp)
            absente = [nom for nom, m in (("vérité", mv), ("prédiction", mp)) if not m["voxels"]]
            if absente:
                limites[cle] = sans_decalage(f"{STRUCTURES[etiquette]} absente de la {' et de la '.join(absente)} : "
                                             "pas de décalage, cas exclu des statistiques")
                limites[cle]["verite_mm"], limites[cle]["prediction_mm"] = mv.get(mesure), mp.get(mesure)
                continue
            limites[cle] = {"verite_mm": mv[mesure], "prediction_mm": mp[mesure],
                            # même arrondi que 05g : niveaux au dixième de mm, puis différence au dixième
                            "decalage_mm": round(mp[mesure] - mv[mesure], 1),
                            "morceaux_verite": mv["morceaux"], "morceaux_prediction": mp["morceaux"],
                            "signal": None}
        l["limites"], l["remarques_geometrie"] = limites, remarques
        # Partie 5 bis, À CÔTÉ de la partie 5 (qui n'est pas touchée) : vérité entière, prédiction réduite
        # en mémoire à son plus grand morceau.
        l[CLE_PGM] = limite_plus_grand_morceau(pred, affine, esp, limites[BASE_PGM])
        del ver, pred                                      # un seul cas en mémoire à la fois


def resumer_decalages(paires, total):
    """Statistiques d'un décalage : celles de resumer() (n, moyenne, médiane, écart-type, min et max avec
    le cas) + la médiane des valeurs absolues ; total = nombre de cas (ou de patients) possibles."""
    r = resumer(paires) or {"n": 0}
    if r["n"]:
        r["mediane_valeurs_absolues"] = statistics.median(abs(x) for _, x in paires if x is not None)
    r["n_total"] = total
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}


def statistiques_decalage(lignes, lire):
    """Statistiques d'UN décalage (lire(ligne) -> décalage ou None) : sur les cas, puis par patient
    (paires moyennées d'abord)."""
    patients = {}
    for l in lignes:
        patients.setdefault(l["patient"], []).append(l)
    sur_les_cas = [(l["cas"][-4:], lire(l)) for l in lignes]
    par_patient = []
    for p, ls in patients.items():
        v = [lire(l) for l in ls if lire(l) is not None]
        par_patient.append((p, statistics.fmean(v) if v else None))       # un seul cas défini : il compte seul
    return {"sur_les_cas": resumer_decalages(sur_les_cas, len(lignes)),
            "par_patient": resumer_decalages(par_patient, len(patients))}


def statistiques_limites(lignes):
    """Pour chaque limite de la partie 5 (tous morceaux) : statistiques sur les cas, puis par patient."""
    return {cle: statistiques_decalage(lignes, lambda l, cle=cle: l["limites"][cle]["decalage_mm"])
            for cle, *_ in LIMITES}


def statistiques_plus_grand_morceau(lignes):
    """Partie 5 bis : mêmes statistiques, sur le décalage du plus grand morceau."""
    return statistiques_decalage(lignes, lambda l: l[CLE_PGM]["decalage_mm"])


# --- Affichage --------------------------------------------------------------------------------
def f3(x):
    return "  —  " if x is None else f"{x:.3f}"


def afficher_tableau(lignes):
    print(f"\n{'pli':>3}  {'cas':<6}{'patient':<11}{'ép. mm':>7}  "
          + "".join(f"{n:>9}" for n in STRUCTURES.values()) + f"{'moy. cas':>10}")
    for l in lignes:
        cellules = ""
        for e in STRUCTURES:
            d = l["dice"][e]
            marque = " " if d is None or d >= SEUIL_FAIBLE else ("!" if d < SEUIL_ECHEC else "*")
            cellules += f"{f3(d):>8}{marque}"
        print(f"{l['pli']:>3}  {l['cas'][-4:]:<6}{l['patient']:<11}{str(l['epaisseur_mm']):>7}  {cellules}"
              f"{f3(l['moyenne_cas']):>10}")
    print(f"  * = faible (Dice < {SEUIL_FAIBLE:.2f}) ; ! = ÉCHEC (Dice < {SEUIL_ECHEC:.2f}) ; — = non défini "
          "(structure vide dans la vérité et la prédiction). Repères indicatifs, non validés.")


def afficher_stats(stats):
    print(f"\n{'':<34}" + "".join(f"{('moy. cas' if n == 'moyenne_cas' else n):>9}" for n in stats))
    for libelle, cle, champ in (("moyenne sur les cas", "sur_les_cas", "moyenne"),
                                ("moyenne PAR PATIENT", "par_patient", "moyenne"),
                                ("médiane sur les cas", "sur_les_cas", "mediane"),
                                ("écart-type sur les cas", "sur_les_cas", "ecart_type"),
                                ("minimum sur les cas", "sur_les_cas", "min"),
                                ("  ... atteint par le cas", "sur_les_cas", "cas_min"),
                                ("maximum sur les cas", "sur_les_cas", "max")):
        valeurs = [(s[cle] or {}).get(champ) for s in stats.values()]
        print(f"  {libelle:<32}" + "".join(f"{(v if isinstance(v, str) else f3(v)):>9}" for v in valeurs))
    epaisseurs = next(iter(stats.values()))["par_epaisseur"]
    for ep in epaisseurs:
        n = (epaisseurs[ep] or {}).get("n", 0)
        nb_patients = (next(iter(stats.values()))["par_epaisseur_patients"].get(ep) or {}).get("n", 0)
        valeurs = [(s["par_epaisseur"].get(ep) or {}).get("moyenne") for s in stats.values()]
        print(f"  {f'moy., {ep} mm ({n} cas)':<32}" + "".join(f"{f3(v):>9}" for v in valeurs))
        valeurs = [(s["par_epaisseur_patients"].get(ep) or {}).get("moyenne") for s in stats.values()]
        print(f"  {f'  par patient ({nb_patients} pat.)':<32}" + "".join(f"{f3(v):>9}" for v in valeurs))


def pct(x):
    return "—" if x is None else f"{100 * x:.0f} %"


def mm1(x, signe=False):
    """Nombre de mm au dixième (avec son signe + ou − si demandé), ou « — » s'il manque."""
    return "—" if x is None else (f"{x:+.1f}" if signe else f"{x:.1f}")


def afficher_limites(lignes, stats):
    print("\n" + "=" * 110)
    print("5. LIMITES AP / VD — décalage = niveau PRÉDIT − niveau ANNOTÉ, en mm le long de l'axe tête-pieds")
    print("   + = la prédiction s'arrête plus HAUT (vers la tête) que l'annotation ; − = plus BAS.")
    print("   Tous les morceaux comptent (fragment isolé compris). Validation hors pli seulement, pas le test.")
    entete = "".join(f"| {libelle:<36}" for _, _, _, libelle in LIMITES)
    print(f"\n{'pli':>3}  {'cas':<6}{'patient':<11}{'ép. mm':>7}  {entete}")
    print(f"{'':>31}" + "".join(f"| {'annoté':>8}{'prédit':>9}{'décalage':>10}{'morc.':>7}  " for _ in LIMITES))
    for l in lignes:
        cellules = ""
        for cle, *_ in LIMITES:
            x = l["limites"][cle]
            marque = "  " if x["signal"] is None else " !"
            cellules += (f"| {mm1(x['verite_mm']):>8}{mm1(x['prediction_mm']):>9}{mm1(x['decalage_mm'], True):>10}"
                         f"{(x['morceaux_prediction'] if x['morceaux_prediction'] is not None else '—'):>7}{marque}")
        print(f"{l['pli']:>3}  {l['cas'][-4:]:<6}{l['patient']:<11}{str(l['epaisseur_mm']):>7}  {cellules}")
    print("  annoté / prédit = niveau en mm dans le repère du scanner (plus grand = plus près de la tête) ;")
    print("  morc. = nombre de morceaux de la PRÉDICTION ; ! = pas de décalage pour ce cas (voir ci-dessous).")

    signales = [(l, cle, l["limites"][cle]["signal"]) for l in lignes for cle, *_ in LIMITES
                if l["limites"][cle]["signal"]]
    remarques = [(l, r) for l in lignes for r in l.get("remarques_geometrie", [])]
    if signales or remarques:
        print("\nCas signalés :")
    for l, cle, s in signales:
        print(f"  {l['cas']} (pli {l['pli']}) {cle} : {s}  <== ANOMALIE")
    for l, r in remarques:
        print(f"  {l['cas']} (pli {l['pli']}) géométrie : {r}  <== À VÉRIFIER")

    for titre, rubrique, unite in (("SUR LES CAS", "sur_les_cas", "cas"),
                                   ("PAR PATIENT (paires 1001/1020 et 1005/1012 moyennées d'abord)",
                                    "par_patient", "patients")):
        print(f"\nDispersion des décalages (mm), {titre} :")
        print(f"  {'':<34}" + "".join(f"{libelle:>22}" for _, _, _, libelle in LIMITES))
        lignes_stats = (
            (f"{unite} utilisés / possibles", lambda s: f"{s['n']} / {s['n_total']}"),
            ("minimum", lambda s: f"{mm1(s.get('min'), True)} ({s.get('cas_min', '—')})"),
            ("maximum", lambda s: f"{mm1(s.get('max'), True)} ({s.get('cas_max', '—')})"),
            ("médiane", lambda s: mm1(s.get("mediane"), True)),
            ("écart-type", lambda s: mm1(s.get("ecart_type"))),
            ("moyenne", lambda s: mm1(s.get("moyenne"), True)),
            ("médiane des valeurs absolues", lambda s: mm1(s.get("mediane_valeurs_absolues"))))
        for libelle, extraire in lignes_stats:
            print(f"  {libelle:<34}" + "".join(f"{extraire(stats[cle][rubrique]):>22}" for cle, *_ in LIMITES))
    print("\n  Ces chiffres DÉCRIVENT l'écart entre le modèle et UN annotateur ; le script ne fixe aucun seuil et "
          "ne conclut pas (voir D31 : la variabilité peut venir de l'annotation).")


def afficher_plus_grand_morceau(lignes, stats, stats_pgm):
    """Partie 5 bis : tableau côte à côte (tous morceaux / plus grand morceau) et statistiques."""
    print("\n" + "=" * 110)
    print("5 bis. BAS DE L'AP : tous morceaux (partie 5) / PLUS GRAND MORCEAU de la prédiction seulement")
    print("   Même décalage (prédit − annoté, mm, + = vers la tête). La VÉRITÉ reste ENTIÈRE (niveau annoté de la")
    print("   partie 5). Plus grand morceau calculé en mémoire (connexité 26, comme 05d) : aucune prédiction ni")
    print("   aucun fichier n'est modifié ; ce n'est PAS le post-traitement de nnU-Net.")
    print(f"\n{'pli':>3}  {'cas':<6}{'patient':<11}{'annoté':>8} | {'tous morceaux':>22} | "
          f"{'plus grand morceau':>34} | {'morceaux prédits':>16}")
    print(f"{'':>30} | {'prédit':>9}{'décalage':>11}   | {'prédit':>9}{'décalage':>11}{'vol. mL':>8}{'part':>6} | "
          f"{'AP':>6}{'VD':>6}")
    for l in lignes:
        t, g = l["limites"][BASE_PGM], l[CLE_PGM]
        mvd = l["limites"].get("haut_VD", {}).get("morceaux_prediction")
        marque = " !" if g["signal"] else (" =" if g["ex_aequo"] else "  ")
        print(f"{l['pli']:>3}  {l['cas'][-4:]:<6}{l['patient']:<11}{mm1(t['verite_mm']):>8} | "
              f"{mm1(t['prediction_mm']):>9}{mm1(t['decalage_mm'], True):>11}   | "
              f"{mm1(g['prediction_mm']):>9}{mm1(g['decalage_mm'], True):>11}{mm1(g['volume_ml']):>8}"
              f"{pct(g['part_volume_predit']):>6} | "
              f"{(t['morceaux_prediction'] if t['morceaux_prediction'] is not None else '—'):>6}"
              f"{(mvd if mvd is not None else '—'):>6}{marque}")
    print("  vol. mL = volume du plus grand morceau ; part = part du volume prédit de l'AP qu'il représente ;")
    print("  morceaux prédits AP / VD = nombre de morceaux de la prédiction (VD : rappel, pas de variante 5 bis) ;")
    print("  ! = pas de décalage (voir « Cas signalés » de la partie 5) ; = = deux morceaux de taille maximale égale "
          "(le premier trouvé est gardé).")

    for titre, rubrique, unite in (("SUR LES CAS", "sur_les_cas", "cas"),
                                   ("PAR PATIENT (paires 1001/1020 et 1005/1012 moyennées d'abord)",
                                    "par_patient", "patients")):
        print(f"\nDispersion du décalage du bas de l'AP (mm), {titre} :")
        print(f"  {'':<34}{'tous morceaux':>22}{'plus grand morceau':>22}")
        for libelle, extraire in (
                (f"{unite} utilisés / possibles", lambda s: f"{s['n']} / {s['n_total']}"),
                ("minimum", lambda s: f"{mm1(s.get('min'), True)} ({s.get('cas_min', '—')})"),
                ("maximum", lambda s: f"{mm1(s.get('max'), True)} ({s.get('cas_max', '—')})"),
                ("médiane", lambda s: mm1(s.get("mediane"), True)),
                ("écart-type", lambda s: mm1(s.get("ecart_type"))),
                ("moyenne", lambda s: mm1(s.get("moyenne"), True)),
                ("médiane des valeurs absolues", lambda s: mm1(s.get("mediane_valeurs_absolues")))):
            print(f"  {libelle:<34}{extraire(stats[BASE_PGM][rubrique]):>22}{extraire(stats_pgm[rubrique]):>22}")
    print("\n  Lecture : si les deux colonnes sont proches pour un cas, les fragments ne changent pas sa limite ;")
    print("  si elles diffèrent, c'est un fragment isolé qui fixait le niveau le plus bas. Aucun seuil, aucune "
          "conclusion ici.")


# --- Programme principal ----------------------------------------------------------------------
def main():
    a = lire_arguments()
    raw, pre, res = (os.path.abspath(x) for x in (a.raw, a.preprocessed, a.results))
    sortie = verifier_sortie(a.sortie, raw, pre, res)
    plis, strates, groupes, cas_entr, cas_test = lire_plis(raw, pre, a.split)

    # --- 1. Un summary.json par pli ---------------------------------------------------------------
    resultats, etat_plis, problemes = {}, {}, []
    for k, p in enumerate(plis):
        val = sorted(p["val"])
        if set(val) & cas_test or not set(val) <= cas_entr:
            arret(1, f"pli {k} : cas de validation hors du lot d'entraînement : {val}")
        chemin = os.path.join(dossier_pli(res, k), "validation", "summary.json")
        if not os.path.isfile(chemin):
            etat_plis[k] = ("pas commencé" if not os.path.isdir(dossier_pli(res, k))
                            else "non terminé (entraînement ou validation finale en cours)")
            print(f"Pli {k} ({', '.join(c[-4:] for c in val)}) : {etat_plis[k]}, summary.json absent.")
            continue
        cas, moy_nnunet, pb = lire_summary(chemin, val, cas_test)
        problemes += [f"pli {k} : {x}" for x in pb]
        moy = moyenne_comme_nnunet(cas)
        ecart = None if moy is None or moy_nnunet is None else abs(moy - moy_nnunet)
        etat_plis[k] = "terminé"
        print(f"Pli {k} ({', '.join(c[-4:] for c in val)}) : terminé ; Dice moyen recalculé {f3(moy)}, "
              f"écrit par nnU-Net {f3(moy_nnunet)}"
              + ("" if ecart is not None and ecart <= TOLERANCE_MOYENNE else "  <== ATTENTION : écart ou valeur absente"))
        resultats[k] = cas
    if problemes:
        arret(6, "summary.json incohérent(s) avec les plis, résultats à NE PAS utiliser :\n  - "
                 + "\n  - ".join(problemes))
    if not resultats:
        arret(1, "aucun pli terminé : rien à rassembler.")
    manquants = [k for k in range(len(plis)) if k not in resultats]

    # --- 2 à 4. Tableau, statistiques, signaux --------------------------------------------------------
    lignes = construire_lignes(resultats, strates, groupes)
    stats = statistiques(lignes)
    signaux = signaler(lignes)
    plus_faibles = sorted((l for l in lignes if l["moyenne_cas"] is not None),
                          key=lambda l: l["moyenne_cas"])[:NB_PLUS_FAIBLES]
    commandes = commandes_05g(lignes, signaux)

    titre = (f"BILAN COMPLET : {len(lignes)} cas de validation, 5 plis" if not manquants else
             f"BILAN PARTIEL : {len(lignes)} cas sur {len(cas_entr)}, plis manquants {manquants} "
             "(les moyennes changeront)")
    print("\n" + "=" * 110 + f"\n{titre}\n" + "=" * 110)
    print("Dice de validation HORS PLI (chaque cas prédit par le modèle qui ne l'a pas vu), pas le test.")
    afficher_tableau(lignes)
    print("\nRAPPEL : 1001/1020 et 1005/1012 sont deux examens du MÊME patient probable (D20). Ils sont dans le "
          "même pli, donc jamais vus à l'entraînement de leur pli, mais ils ne sont pas indépendants : deux "
          "bons scores sur une paire valent une seule confirmation. La « moyenne par patient » compte chaque "
          f"patient une fois ({len({l['patient'] for l in lignes})} patients pour {len(lignes)} cas).")
    afficher_stats(stats)

    print(f"\nCellules signalées (repères indicatifs : faible < {SEUIL_FAIBLE:.2f}, ÉCHEC < {SEUIL_ECHEC:.2f}) :")
    if not signaux:
        print("  aucune.")
    for s in signaux:
        if s["dice"] is None:
            print(f"  {s['cas']} (pli {s['pli']}) {s['etiquette']} {s['structure']:<7} {s['niveau']}  <== ANOMALIE")
            continue
        print(f"  {s['cas']} (pli {s['pli']}) {s['etiquette']} {s['structure']:<7} Dice {s['dice']:.3f}  {s['niveau']:<6}"
              f"  oublis {pct(s['part_oublis']):>5} de la vérité ; ajouts {pct(s['part_ajouts']):>5} de la prédiction "
              f"({s['voxels_verite']} voxels dans la vérité, {s['voxels_predits']} prédits)")
    print("  Lecture : surtout des oublis = le modèle en met moins que l'annotation ; surtout des ajouts = il en "
          "met plus ; les deux = la structure est décalée (frontière ou étendue différente). À vérifier sur les "
          "figures de 05g.")
    print(f"\nLes {NB_PLUS_FAIBLES} cas à la moyenne la plus basse : "
          + " ; ".join(f"{l['cas']} (pli {l['pli']}, {l['moyenne_cas']:.3f})" for l in plus_faibles))
    if commandes:
        print("\nPour examiner les cas signalés avec les figures de 05d (script 05g, notebook section 20) :")
        for c, cmd in commandes.items():
            print(f"  python scripts/05g_examiner_cas.py {cmd}")
        print("  (2e numéro = cas de comparaison proposé : même épaisseur de coupe, autre patient, meilleur Dice "
              "sur ces structures ; on peut en choisir un autre.)")

    # --- 5. Limites AP / VD : lit la vérité et la prédiction des cas de validation des plis terminés ----
    print(f"\nLimites AP / VD : lecture des {len(lignes)} cas de validation (labelsTr et fold_X/validation, "
          "jamais le test)...")
    mesurer_limites(lignes, raw, res, cas_test)
    stats_limites = statistiques_limites(lignes)
    afficher_limites(lignes, stats_limites)
    stats_pgm = statistiques_plus_grand_morceau(lignes)       # partie 5 bis (calcul déjà fait en mémoire)
    afficher_plus_grand_morceau(lignes, stats_limites, stats_pgm)

    # --- Écriture -------------------------------------------------------------------------------------
    os.makedirs(sortie, exist_ok=True)                      # le seul dossier où le script écrit
    chemin_csv = os.path.join(sortie, NOM_CSV)
    with open(chemin_csv + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Colonnes des limites : niveau annoté, niveau prédit, décalage (prédit − annoté, + = vers la tête),
        # morceaux de la prédiction ; cellule vide = pas de valeur (voir « signaux_limites »).
        champs_limites = (("verite_mm", "annote_mm"), ("prediction_mm", "predit_mm"),
                          ("decalage_mm", "decalage_mm"), ("morceaux_prediction", "morceaux_predits"))
        # Colonnes de la partie 5 bis, AJOUTÉES À LA FIN (les colonnes existantes ne bougent pas) : le niveau
        # annoté et le nombre de morceaux sont déjà dans les colonnes bas_AP_* ; part = fraction (0 à 1).
        champs_pgm = (("prediction_mm", "predit_mm"), ("decalage_mm", "decalage_mm"), ("volume_ml", "volume_ml"),
                      ("part_volume_predit", "part_volume_predit"), ("ex_aequo", "ex_aequo"))
        w.writerow(["pli", "cas", "patient", "epaisseur_mm", *[f"dice_{e}_{n}" for e, n in STRUCTURES.items()],
                    "moyenne_cas", "signaux",
                    *[f"{cle}_{nom}" for cle, *_ in LIMITES for _, nom in champs_limites], "signaux_limites",
                    *[f"{CLE_PGM}_{nom}" for _, nom in champs_pgm]])
        for l in lignes:
            sig = "; ".join(f"{s['structure']} {s['niveau']}" for s in signaux if s["cas"] == l["cas"])
            sig_lim = "; ".join([f"{cle} : {l['limites'][cle]['signal']}" for cle, *_ in LIMITES
                                 if l["limites"][cle]["signal"]] + l.get("remarques_geometrie", []))
            w.writerow([l["pli"], l["cas"], l["patient"], l["epaisseur_mm"],
                        *[("" if l["dice"][e] is None else round(l["dice"][e], 4)) for e in STRUCTURES],
                        "" if l["moyenne_cas"] is None else round(l["moyenne_cas"], 4), sig,
                        *[("" if l["limites"][cle][champ] is None else l["limites"][cle][champ])
                          for cle, *_ in LIMITES for champ, _ in champs_limites], sig_lim,
                        *[("" if l[CLE_PGM][champ] is None else l[CLE_PGM][champ]) for champ, _ in champs_pgm]])
    os.replace(chemin_csv + ".tmp", chemin_csv)             # jamais de fichier à moitié écrit
    synthese = {"objet": "bilan de la validation croisée nnU-Net (validation hors pli, PAS le test), LECTURE SEULE",
                "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "complet": not manquants, "plis_manquants": manquants, "etat_plis": etat_plis,
                "trainer": ana.TRAINER, "configuration": ana.CONFIGURATION,
                "split_sha256": prep.sha256_fichier(a.split), "code_commit": prep.commit_git(),
                "plis": [{"val": sorted(p["val"])} for p in plis], "groupes_patients": groupes,
                "reperes_indicatifs": {"faible": SEUIL_FAIBLE, "echec": SEUIL_ECHEC},
                "structures": STRUCTURES,
                "lignes": [{k: v for k, v in l.items()} for l in lignes],
                "statistiques": stats, "signaux": signaux,
                "plus_faibles": [l["cas"] for l in plus_faibles], "commandes_05g": commandes,
                # Partie 5 ; les valeurs par cas sont dans « lignes » (rubriques « limites » et
                # « remarques_geometrie »).
                "limites_AP_VD": {
                    "definition": {cle: f"{libelle} : {mesure} prédit − {mesure} annoté" for cle, _, mesure, libelle in LIMITES},
                    "unite": "mm le long de l'axe tête-pieds (repère du scanner)",
                    "convention_signe": "positif = la prédiction s'arrête plus haut (vers la tête) que l'annotation",
                    "methode": "charger() et mesurer() de 05d, comme 05g ; tous les morceaux comptent, fragments "
                               "isolés compris ; niveaux arrondis au dixième de mm avant la différence",
                    "par_patient": "les deux cas d'une paire sont moyennés d'abord ; un seul cas défini compte seul",
                    "seuil": None,
                    "statistiques": stats_limites,
                    "cas_sans_decalage": {cle: [l["cas"] for l in lignes if l["limites"][cle]["decalage_mm"] is None]
                                          for cle, *_ in LIMITES},
                    # Partie 5 bis ; valeurs par cas dans « lignes », rubrique « bas_AP_plus_grand_morceau ».
                    CLE_PGM: {
                        "definition": "bas de l'AP (7) : niveau_bas_mm du PLUS GRAND MORCEAU de la prédiction − "
                                      "niveau_bas_mm de la vérité ENTIÈRE",
                        "unite": "mm le long de l'axe tête-pieds (repère du scanner)",
                        "convention_signe": "positif = la prédiction s'arrête plus haut (vers la tête) que l'annotation",
                        "connexite": "26 (face, arête ou coin ; structure 3 × 3 × 3), comme le comptage des morceaux "
                                     "de mesurer() dans 05d",
                        "verite": "non modifiée : annotation entière, même niveau annoté que bas_AP",
                        "methode": "plus grande composante connexe de l'étiquette 7 prédite, calculée en mémoire ; "
                                   "mesurer() de 05d sur ce seul morceau ; aucune prédiction ni aucun fichier modifié ; "
                                   "ce n'est pas le post-traitement de nnU-Net",
                        "part_volume_predit": "voxels du plus grand morceau / voxels prédits de l'AP (fraction de 0 à 1)",
                        "par_patient": "les deux cas d'une paire sont moyennés d'abord ; un seul cas défini compte seul",
                        "seuil": None,
                        "statistiques": stats_pgm,
                        "cas_sans_decalage": [l["cas"] for l in lignes if l[CLE_PGM]["decalage_mm"] is None],
                        "cas_ex_aequo": [l["cas"] for l in lignes if l[CLE_PGM]["ex_aequo"]]}}}
    chemin = os.path.join(sortie, NOM_SYNTHESE)
    with open(chemin + ".tmp", "w", encoding="utf-8") as f:
        json.dump(synthese, f, indent=2, ensure_ascii=False)
    os.replace(chemin + ".tmp", chemin)
    print(f"\nÉcrit dans {sortie} : {NOM_CSV}, {NOM_SYNTHESE}")
    print("Rien n'a été écrit dans nnUNet_raw, nnUNet_preprocessed ni nnUNet_results ; aucun cas de test lu.")
    if manquants:
        print(f"BILAN PARTIEL (code 2) : relancer quand les plis {manquants} seront terminés.")
        sys.exit(2)


if __name__ == "__main__":
    main()
