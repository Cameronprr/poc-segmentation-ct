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
     seuls les summary.json de fold_X/validation sont lus (aucune image, aucun masque) ;
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
     et d'ajouts. Puis les 3 cas à la moyenne la plus basse, et la commande du script 05g pour les examiner.
  Si des plis ne sont pas terminés, le bilan est PARTIEL (clairement marqué) et le code de sortie vaut 2.
  Le script NE CONCLUT PAS et ne fixe aucun seuil du contrôle de plausibilité (D28, D29).

Entrées :
  --raw           dossier nnUNet_raw (défaut /content/drive/MyDrive/poc_ct/nnunet_raw) ; seuls les
                  EN-TÊTES de Dataset101_MMWHS/imagesTr sont lus (épaisseurs de coupe, contrôle des plis)
  --preprocessed  dossier nnUNet_preprocessed (défaut .../poc_ct/nnunet_preprocessed) : splits_final.json,
                  sa fiche splits_final_05a.json et le marqueur verification_05a.json (lus seulement)
  --results       dossier nnUNet_results (défaut .../poc_ct/nnunet_results) ; seuls les
                  .../nnUNetTrainer_250epochs_reprise10__nnUNetPlans__3d_fullres/fold_X/validation/summary.json
                  sont lus
  --sortie        dossier de sortie (défaut /content/drive/MyDrive/poc_ct/analyses/05f_bilan)
  --split         découpage figé (défaut : ../splits/mmwhs_split.json à côté du script)
Sorties (uniquement dans --sortie) :
  bilan_validation_croisee.csv (une ligne par cas) ; synthese_05f.json (tableau, statistiques, signaux)
Codes de sortie :
  0 = les 5 plis sont rassemblés ; 1 = garde-fou (rien n'est écrit) ; 2 = bilan PARTIEL (plis non terminés,
  fichiers écrits quand même) ; 6 = un summary.json ne correspond pas aux plis (rien n'est écrit : ne pas
  utiliser ces résultats).

Vocabulaire :
  - Dice : recouvrement entre la prédiction et la vérité, de 0 (aucun) à 1 (parfait) ;
  - oubli (faux négatif) : voxel de la vérité que le modèle n'a pas mis dans la structure.
    Part d'oublis = oublis / voxels de la vérité ;
  - ajout (faux positif) : voxel prédit dans la structure alors que la vérité ne l'y met pas.
    Part d'ajouts = ajouts / voxels prédits ;
  - prédiction hors pli : prédiction d'un cas par le modèle qui ne l'a pas vu à l'entraînement ;
  - moyenne par patient : 1001 et 1020 sont deux examens du même patient (de même 1005 et 1012). Les
    compter comme deux cas donne deux fois plus de poids à ce patient : la moyenne par patient corrige cela ;
  - écart-type : dispersion des valeurs autour de la moyenne.
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
ana = charger_module("analyse05d", "05d_analyser_1019.py")     # garde-fous de sortie, nom du trainer
prep = essai.prep                                              # module du script 05a

# --- Paramètres fixes -------------------------------------------------------------------------
NOM_DS = ana.NOM_DS
STRUCTURES = {1: "myo_VG", 2: "OG", 3: "VG", 4: "OD", 5: "VD", 6: "aorte", 7: "AP"}   # étiquettes du script 03
SEUIL_FAIBLE, SEUIL_ECHEC = 0.85, 0.50   # repères INDICATIFS, fixés avant de voir les 5 plis, non validés
NB_PLUS_FAIBLES = 3
TOLERANCE_MOYENNE = 1e-4                 # écart toléré entre notre moyenne et celle de nnU-Net
NOM_CSV, NOM_SYNTHESE = "bilan_validation_croisee.csv", "synthese_05f.json"


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

    # --- Écriture -------------------------------------------------------------------------------------
    os.makedirs(sortie, exist_ok=True)                      # le seul dossier où le script écrit
    chemin_csv = os.path.join(sortie, NOM_CSV)
    with open(chemin_csv + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pli", "cas", "patient", "epaisseur_mm", *[f"dice_{e}_{n}" for e, n in STRUCTURES.items()],
                    "moyenne_cas", "signaux"])
        for l in lignes:
            sig = "; ".join(f"{s['structure']} {s['niveau']}" for s in signaux if s["cas"] == l["cas"])
            w.writerow([l["pli"], l["cas"], l["patient"], l["epaisseur_mm"],
                        *[("" if l["dice"][e] is None else round(l["dice"][e], 4)) for e in STRUCTURES],
                        "" if l["moyenne_cas"] is None else round(l["moyenne_cas"], 4), sig])
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
                "plus_faibles": [l["cas"] for l in plus_faibles], "commandes_05g": commandes}
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
