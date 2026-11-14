"""
Étape 5j — CALIBRER le contrôle de plausibilité (D28, D29, D32, D33, D34) et l'APPLIQUER aux 15 prédictions hors
pli post-traitées. LECTURE SEULE : aucun fichier nnU-Net n'est modifié ; le lot de test n'est jamais lu ;
rien n'est écrit dans le dépôt (ni dans config/) : les seuils sont PROPOSÉS, pas figés.

Contexte : la spécification (docs/specification_controle_plausibilite.md, v1.0) définit 9 règles, R0 à R8,
qui signalent un masque invraisemblable (statut ALERTE). Leurs seuils se calculent sur les cas
d'entraînement, 1019 exclu (D28, § 6.4), selon les formules du § 5 et du § 6.3 (marges fixées par Q2).
Ce script calcule ces seuils, puis vérifie le critère du § 7 (Q3). Il NE DÉCIDE RIEN : si le critère n'est
pas atteint, il le dit ; il ne change ni une marge, ni une règle pour l'atteindre.

Ce que fait le script :
  0. garde-fous : découpage figé lu (jamais modifié) ; splits_final.json vérifié comme dans 05f ; tout cas
     hors de « entrainement_validation_croisee » est refusé (cas de test 1004, 1010, 1011, 1013, 1014 :
     code 1) ; jamais imagesTs ni labelsTs ; les prédictions sont lues UNIQUEMENT dans
     crossval_results_folds_0_1_2_3_4/postprocessed/, qui doit contenir exactement les 15 cas
     d'entraînement ; le post-traitement de postprocessing.json doit être celui de D33 (plus grand morceau
     pour OG, OD, VD et aorte, rien d'autre) ; --sortie hors du dépôt et des dossiers nnU-Net ;
  1. MESURES (module controle_plausibilite.py, qui reprend 05e, 05h et 05i à l'identique) sur la VÉRITÉ et
     sur la PRÉDICTION post-traitée de chacun des 15 cas, un scanner à la fois ;
  2. CALIBRAGE sur les 14 cas sans 1019 (12 patients) :
       R0  fixé : même forme, affine à 0,01 mm près, étiquettes 0 à 7 (spéc. § 5, ligne R0) ;
       R1  fixé : volume de chaque étiquette 1 à 7 > 1 mL (ligne R1) ;
       R2  vérité : volume de chaque structure dans [min × 0,7 ; max × 1,3] (ligne R2, § 6.3, Q2) ;
       R3  vérité : 5 rapports de volumes dans [min × 0,7 ; max × 1,3] (ligne R3) ;
       R4  vérité : |aorte − cavité VG| ≤ écart maximal observé + 50 HU (ligne R4) ;
       R5  vérité : |AP − VD| ≤ écart maximal observé + 50 HU (ligne R5) ;
       R6  vérité : aorte − AP ≥ écart minimal observé − 50 HU (ligne R6 : « ≥ −S6 », S6 = 50 − minimum) ;
       R7  vérité : densité de la cavité VG ≥ minimum observé − 50 HU (ligne R7) ;
       R8  vérité (D34) : pour chaque étiquette, AP comprise, nombre de morceaux de plus de 0,5 mL ≤ N et
           part du plus gros morceau ≥ p, avec N = maximum et p = minimum observés sur la VÉRITÉ (§ 6.3, sans
           marge) ; la règle est ensuite APPLIQUÉE aux prédictions post-traitées. Aucun cas n'est retiré du
           calibrage d'après son résultat. Les prédictions ne servent à calibrer AUCUNE règle ;
     Limites de R8 affichées (sans rien ajuster) : si N(AP) ≥ 2, le fragment isolé de l'AP de 1018 (D31) ne
     peut pas être détecté ; si N(myocarde) ≥ 2, R8 est peu sensible pour le myocarde (D34) ;
     Arrondis : bornes basses arrondies vers le bas, bornes hautes vers le haut (l'arrondi n'exclut
     jamais une valeur du calibrage) ;
  3. VÉRIFICATION « en laissant un patient de côté » (§ 6.4 et § 7) : pour chacun des 12 patients, seuils
     recalculés sans lui (paires 1001/1020 et 1005/1012 retirées ENSEMBLE), puis :
       A (exigée, § 7) : R0 à R8 appliquées à la VÉRITÉ de ce patient ; plus la vérité de 1019 avec les
         seuils des 14 cas, R0 à R8 (seule R6 y est admise, § 7) ;
       B (complément, information) : R0 à R8 appliquées à la PRÉDICTION de ce patient. C'est la seule
         estimation des fausses alertes (de R8 surtout) sur des prédictions que les seuils n'ont pas vues ;
  4. APPLICATION des seuils proposés aux prédictions post-traitées (défaut : les 15 cas) : règles
     déclenchées, mesure et seuil, catégorie, statut OK / ALERTE ;
  5. CRITÈRE du § 7 (Q3), affiché sans conclure au-delà : 1019 en ALERTE ; 1001 et 1020 OK ; au plus une
     fausse alerte sur les 14 autres ; aucune fausse alerte en A ;
  6. familles d'épaisseur de coupe (§ 6.3) : on signale, à titre indicatif, une grandeur dont les plages
     des deux familles ne se recouvrent pas, et, par famille, l'alerte de 1019 et les fausses alertes ;
  7. concordance avec les sections 18 et 21 : densités et volumes de la vérité comparés à synthese_05e.json
     et synthese_05h.json, s'ils sont présents (sinon, c'est dit).

Entrées :
  --raw           nnUNet_raw (défaut /content/drive/MyDrive/poc_ct/nnunet_raw) : imagesTr, labelsTr,
                  dataset.json de Dataset101_MMWHS (jamais imagesTs ni labelsTs)
  --preprocessed  nnUNet_preprocessed (défaut .../poc_ct/nnunet_preprocessed) : splits_final.json et fiches
                  de 05a (lus, pour connaître le pli de chaque cas)
  --results       nnUNet_results (défaut .../poc_ct/nnunet_results) : SEULS postprocessed/<cas>.nii.gz et
                  postprocessing.json de crossval_results_folds_0_1_2_3_4 sont lus
  --sortie        dossier de sortie (défaut /content/drive/MyDrive/poc_ct/analyses/05j_controle)
  --split         découpage figé (défaut ../splits/mmwhs_split.json)
  --cas           cas auxquels APPLIQUER le contrôle (défaut : les 15) ; le calibrage utilise toujours les 14
  --ref-05e, --ref-05h  synthèses des sections 18 et 21 pour le contrôle de concordance (défauts dans Drive)
Sorties (uniquement dans --sortie, jamais dans le dépôt) :
  seuils_proposes.json  seuils au format prévu pour config/seuils_plausibilite.json (Q6) : valeurs agrégées
                        seulement, aucun identifiant ni valeur par cas, « statut » : « proposé, non figé » ;
  mesures_05j.csv       mesures par cas, vérité et prédiction ;
  application_05j.csv   une ligne par prédiction : statut, règles déclenchées par catégorie, détails ;
  lopo_05j.csv          vérification en laissant un patient de côté (A et B) ;
  synthese_05j.json     tout : seuils, cas des extrêmes, mesures, alertes, critère, familles, concordance.
Codes de sortie : 0 = terminé (lire le CRITÈRE et les AVERTISSEMENTS) ; 1 = garde-fou (rien n'est écrit) ;
  6 = rien n'est écrit, résultats inutilisables : calibrage impossible (vérité d'un cas de calibrage non
  mesurable, ou structure absente de la vérité), prédiction non mesurable (forme différente de l'image), ou
  postprocessing.json contenant une fonction inattendue (arrêt en code 6 par lire_postraitement() de 05i).

Vocabulaire :
  - calibrage : calcul des seuils à partir de cas connus ;
  - marge : tolérance ajoutée aux extrêmes observés (30 % pour les volumes et rapports, 50 HU pour les
    densités), fixée AVANT de voir les résultats (Q2) ;
  - fausse alerte : ALERTE sur un cas dont la segmentation est réussie (ici, les 14 cas autres que 1019) ;
  - leave-one-patient-out (« en laissant un patient de côté ») : on recalcule les seuils sans un patient,
    puis on contrôle ce patient avec ces seuils qui ne l'ont pas vu ; cela imite un patient nouveau ;
  - hors domaine : l'examen ne ressemble pas à ceux du calibrage (R6 : AP plus dense que l'aorte ; R7 :
    contraste insuffisant) ; c'est une limite du modèle, pas forcément une erreur du masque (Q8).
"""
import argparse, csv, datetime, importlib.util, json, math, os, re, sys

ICI = os.path.dirname(os.path.abspath(__file__))


def charger_module(nom, fichier):
    spec = importlib.util.spec_from_file_location(nom, os.path.join(ICI, fichier))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ctl = charger_module("controle_plausibilite", "controle_plausibilite.py")   # mesures et règles
bil = charger_module("bilan05f", "05f_bilan_validation_croisee.py")         # garde-fous communs (sortie, plis)
ana, prep, pp = bil.ana, bil.prep, ctl.pp                                    # 05d, 05a, 05i

# --- Paramètres fixes -------------------------------------------------------------------------
MARGE_RELATIVE = 0.30                 # Q2 : volumes et rapports, [min × (1 − m) ; max × (1 + m)]
MARGE_HU = 50                         # Q2 : densités
CAS_EXCLU = "mmwhs_1019"              # exclu du calibrage de TOUTES les règles (D28, spéc. § 6.4)
CAS_DOIVENT_ETRE_OK = ("mmwhs_1001", "mmwhs_1020")   # spéc. § 7, prédictions du pli 0
POSTRAITEMENT_D33 = [2, 4, 5, 6]      # OG, OD, VD, aorte : plus grand morceau (D33) ; rien d'autre
NOM_CV = pp.NOM_CV                    # crossval_results_folds_0_1_2_3_4
NOMS = {"seuils": "seuils_proposes.json", "mesures": "mesures_05j.csv", "application": "application_05j.csv",
        "lopo": "lopo_05j.csv", "synthese": "synthese_05j.json"}
arret = bil.arret


def lire_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=f"{prep.DRIVE}/nnunet_raw")
    p.add_argument("--preprocessed", default=f"{prep.DRIVE}/nnunet_preprocessed")
    p.add_argument("--results", default=f"{prep.DRIVE}/nnunet_results")
    p.add_argument("--sortie", default=f"{prep.DRIVE}/analyses/05j_controle")
    p.add_argument("--split", default=os.path.join(prep.DEPOT, "splits", "mmwhs_split.json"))
    p.add_argument("--cas", nargs="+", default=None)
    p.add_argument("--ref-05e", default=f"{prep.DRIVE}/analyses/05e_densites/synthese_05e.json")
    p.add_argument("--ref-05h", default=f"{prep.DRIVE}/analyses/05h_volumes/synthese_05h.json")
    return p.parse_args()


# --- 0. Garde-fous (rien n'est écrit avant qu'ils soient passés) -------------------------------
def verifier(a):
    raw, pre, res = (os.path.abspath(x) for x in (a.raw, a.preprocessed, a.results))
    sortie = bil.verifier_sortie(a.sortie, raw, pre, res)
    plis, strates, groupes, cas, cas_test = bil.lire_plis(raw, pre, a.split)
    pli_de = {c: k for k, p in enumerate(plis) for c in p["val"]}
    if set(pli_de) != cas:
        arret(1, f"les cas de validation des plis {sorted(pli_de)} ≠ les 15 cas d'entraînement.")
    appliques = []
    for n in dict.fromkeys(a.cas or [c[-4:] for c in sorted(cas)]):     # sans doublon, ordre gardé
        c = f"mmwhs_{n}"
        if not re.fullmatch(r"\d{4}", n):
            arret(1, f"numéro de cas invalide : {n} (attendu : 4 chiffres, ex. 1019).")
        if c in cas_test:
            arret(1, f"{c} fait partie du LOT DE TEST : refusé (le test ne sert qu'à l'étape 6, D18).")
        if c not in cas:
            arret(1, f"{c} n'est pas dans le lot d'entraînement du découpage figé : refusé.")
        appliques.append(c)
    ds = os.path.join(raw, ana.NOM_DS)
    with open(os.path.join(ds, "dataset.json"), encoding="utf-8") as f:
        if json.load(f).get("labels") != ana.ETIQUETTES:
            arret(1, f"les étiquettes de dataset.json ne sont pas celles attendues ({ana.ETIQUETTES}).")
    cv = os.path.join(pp.dossier_modele(res), NOM_CV)
    post = os.path.join(cv, "postprocessed")
    if not os.path.isdir(post) or not os.path.isfile(os.path.join(cv, "postprocessing.json")):
        arret(1, f"{post} ou postprocessing.json absent : lancer d'abord la section 22 (05i).")
    presents = {x[:-len(".nii.gz")] for x in os.listdir(post) if x.endswith(".nii.gz")}
    if presents & cas_test:
        arret(1, f"postprocessed/ contient des cas de TEST ({sorted(presents & cas_test)}) : refusé, rien n'est lu.")
    if presents != cas:
        arret(1, f"postprocessed/ contient {sorted(presents)} ; attendu : exactement les 15 cas d'entraînement "
                 f"(manquants {sorted(cas - presents)}, en trop {sorted(presents - cas)}).")
    _, etapes = pp.lire_postraitement(cv)                                # lecture seule de postprocessing.json
    retenus = sorted(e for x in etapes for e in x["etiquettes"]) if all(x["etape"] == "b" for x in etapes) else None
    if retenus != POSTRAITEMENT_D33:
        arret(1, f"post-traitement de postprocessing.json ({[x['cible'] for x in etapes]}) ≠ celui de D33 "
                 f"(plus grand morceau des étiquettes {POSTRAITEMENT_D33} seulement) : ces prédictions ne sont pas "
                 "celles retenues.")
    chemins = {}
    for c in sorted(cas):
        ch = {"image": os.path.join(ds, "imagesTr", f"{c}_0000.nii.gz"),
              "verite": os.path.join(ds, "labelsTr", f"{c}.nii.gz"),
              "prediction": os.path.join(post, f"{c}.nii.gz")}
        if any(pp.test_dans_chemin(x) for x in ch.values()):
            arret(1, "un chemin passe par imagesTs ou labelsTs : refusé.")
        manquants = [x for x in ch.values() if not os.path.isfile(x)]
        if manquants:
            arret(1, f"fichiers introuvables (Drive relié ?) : {manquants}")
        chemins[c] = ch
    patient_de = {c: "/".join(x[-4:] for x in g) for g in groupes for c in g}
    print(f"Garde-fous passés : 15 cas d'entraînement, aucun cas de test ; prédictions lues dans {post} "
          f"(post-traitement D33 : étiquettes {POSTRAITEMENT_D33}) ; contrôle appliqué à {len(appliques)} cas.")
    return sortie, chemins, appliques, pli_de, strates, patient_de, cas


# --- 2. Calibrage ------------------------------------------------------------------------------
def bas(x, dec):
    return math.floor(x * 10 ** dec) / 10 ** dec


def haut(x, dec):
    return math.ceil(x * 10 ** dec) / 10 ** dec


def extremes(paires):
    v = [(x, c) for x, c in paires if x is not None]
    if not v:
        return None
    return {"n": len(v), "min": min(v)[0], "cas_min": min(v)[1], "max": max(v)[0], "cas_max": max(v)[1]}


def calibrer(ver, calib):
    """Seuils (format de seuils_proposes.json, sans en-tête) et extrêmes observés (avec les cas : jamais
    dans le fichier de seuils). ver : mesures de la VÉRITÉ par cas (toutes les règles, R8 comprise, D34) ;
    calib : cas utilisés."""
    calib = sorted(calib)
    x = {}
    for e in ctl.STRUCTURES:
        x[f"R2_{e}"] = extremes([(ver[c]["volumes_ml"][e], c) for c in calib])
    for r, _, _ in ctl.RAPPORTS:
        x[f"R3_{r}"] = extremes([(ver[c]["rapports"][r], c) for c in calib])
    x["R4"] = extremes([(None if ver[c]["ecarts_hu"]["aorte_moins_VG"] is None else abs(ver[c]["ecarts_hu"]["aorte_moins_VG"]), c) for c in calib])
    x["R5"] = extremes([(None if ver[c]["ecarts_hu"]["AP_moins_VD"] is None else abs(ver[c]["ecarts_hu"]["AP_moins_VD"]), c) for c in calib])
    x["R6"] = extremes([(ver[c]["ecarts_hu"]["aorte_moins_AP"], c) for c in calib])
    x["R7"] = extremes([(ver[c]["hu"]["VG"], c) for c in calib])
    for e in ctl.STRUCTURES:
        x[f"R8_n_{e}"] = extremes([(ver[c]["morceaux"][e]["n_gros"], c) for c in calib])          # vérité (D34)
        x[f"R8_p_{e}"] = extremes([(ver[c]["morceaux"][e]["part_plus_gros"], c) for c in calib])
    m = MARGE_RELATIVE
    s = {"R0": {"tolerance_affine_mm": ctl.TOLERANCE_R0_MM, "etiquettes_autorisees": list(ctl.ETIQUETTES_AUTORISEES)},
         "R1": {"volume_min_ml": ctl.VOLUME_MIN_R1_ML},
         "R2": {"plages_ml": {str(e): {"bas": bas(x[f"R2_{e}"]["min"] * (1 - m), 3), "haut": haut(x[f"R2_{e}"]["max"] * (1 + m), 3)}
                              for e in ctl.STRUCTURES if x[f"R2_{e}"]}},
         "R3": {"plages": {r: {"numerateur": n, "denominateur": d, "bas": bas(x[f"R3_{r}"]["min"] * (1 - m), 4),
                               "haut": haut(x[f"R3_{r}"]["max"] * (1 + m), 4)}
                           for r, n, d in ctl.RAPPORTS if x[f"R3_{r}"]}},
         "R4": x["R4"] and {"ecart_hu_max": x["R4"]["max"] + MARGE_HU},
         "R5": x["R5"] and {"ecart_hu_max": x["R5"]["max"] + MARGE_HU},
         "R6": x["R6"] and {"ecart_aorte_moins_AP_min_hu": x["R6"]["min"] - MARGE_HU},
         "R7": x["R7"] and {"hu_cavite_VG_min": x["R7"]["min"] - MARGE_HU},
         "R8": {"morceau_min_ml": ctl.MORCEAU_ML, "connexite": 26,
                "par_etiquette": {str(e): {"morceaux_gros_max": x[f"R8_n_{e}"]["max"],
                                           "part_plus_gros_min": bas(x[f"R8_p_{e}"]["min"], 4)}
                                  for e in ctl.STRUCTURES if x[f"R8_n_{e}"] and x[f"R8_p_{e}"]}}}
    return s, x


def entete_seuils(s, n_cas, n_patients, sha_split, commit):
    """Fichier complet, au format prévu pour config/seuils_plausibilite.json (spéc. § 3, § 6.2, Q6)."""
    return {"statut": "proposé, non figé",
            "objet": "seuils du contrôle de plausibilité des segmentations (D28, D29, D33, D34)",
            "specification": "docs/specification_controle_plausibilite.md, version 1.0 ; règles R0 à R8 (Q1)",
            "date_calcul_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "date_gel": None, "sha256_du_fichier_gele": None,
            "marges": {"volumes_et_rapports": MARGE_RELATIVE, "densites_hu": MARGE_HU,
                       "morceaux": "aucune (spéc. § 6.3 : N maximum et p minimum observés)"},
            "calibrage": {"R2_a_R8": "vérité terrain des cas d'entraînement, cas à phase d'injection inversée exclu "
                                     "(D28, spéc. § 6.4 ; R8 : D34)",
                          "application": "prédictions hors pli post-traitées comme retenu par nnU-Net (D33)",
                          "nombre_cas": n_cas, "nombre_patients": n_patients},
            "definitions": {
                "volume_ml": "voxels × produit des 3 espacements de l'en-tête de l'image / 1000",
                "rapport": "rapport des nombres de voxels, dans la même image",
                "densite_hu": "médiane des HU au cœur de la structure prédite (voxels à plus de 3 mm du bord), "
                              "structure entière si le cœur est vide ; écarts entre médianes arrondies",
                "morceaux": "composantes connexes (connexité 26) de plus de 0,5 mL ; part du plus gros = son volume / "
                            "volume total de l'étiquette",
                "R2_R3": "déclenchée si la valeur sort de [bas ; haut]",
                "R4_R5": "déclenchée si |écart| > ecart_hu_max",
                "R6": "déclenchée si aorte − AP < ecart_aorte_moins_AP_min_hu (spéc. : ≥ −S6, S6 = −valeur)",
                "R7": "déclenchée si densité cavité VG < hu_cavite_VG_min",
                "R8": "déclenchée si morceaux > morceaux_gros_max ou part du plus gros < part_plus_gros_min",
                "arrondis": "bornes basses vers le bas, bornes hautes vers le haut"},
            **s,
            "categories": {"hors_domaine": ["R6", "R7"], "erreur_probable": ["R0", "R1", "R2", "R3", "R4", "R5", "R8"]},
            "provenance": {"script": "scripts/05j_calibrer_controle.py", "split_sha256": sha_split, "code_commit": commit,
                           "postraitement_nnunet": f"plus grand morceau des étiquettes {POSTRAITEMENT_D33} (D33)"}}


def verifier_sans_identifiant(seuils, tous_cas):
    """Q6 : aucun identifiant de cas dans le fichier de seuils (textes et clés ; empreintes exclues)."""
    numeros = {c[-4:] for c in tous_cas}
    trouves = []

    def parcourir(o, chemin):
        if isinstance(o, dict):
            for k, v in o.items():
                if k not in ("split_sha256", "code_commit"):
                    parcourir(k, chemin + "/clé")
                    parcourir(v, f"{chemin}/{k}")
        elif isinstance(o, list):
            for v in o:
                parcourir(v, chemin)
        elif isinstance(o, str):
            if re.search(r"mmwhs_|ct_train", o) or any(re.search(rf"\b{n}\b", o) for n in numeros):
                trouves.append(f"{chemin} : {o}")
    parcourir(seuils, "")
    return trouves


# --- 3 à 7. Vérifications ------------------------------------------------------------------------
def limites_de_r8(seuils):
    """Conséquences connues du calibrage de R8 sur la vérité (D34) : affichées, jamais corrigées."""
    t, lim = seuils["R8"]["par_etiquette"], []
    n_ap, n_myo = (t.get(k, {}).get("morceaux_gros_max") for k in ("7", "1"))
    if n_ap is not None and n_ap >= 2:
        lim.append(f"N(AP) = {n_ap} : le fragment de 1018 ne peut pas être détecté par R8 (limite connue, D34)")
    if n_myo is not None and n_myo >= 2:
        lim.append(f"N(myocarde) = {n_myo} : R8 peu sensible pour le myocarde, en plusieurs morceaux dans la vérité "
                   "(limite connue, D34)")
    return lim


def patients_de(calib, patient_de):
    p = {}
    for c in sorted(calib):
        p.setdefault(patient_de.get(c, c[-4:]), []).append(c)
    return p


def lopo(ver, pred, calib, patient_de):
    """Pour chaque patient : seuils sans lui, puis A (vérité, R0-R8) et B (prédiction, R0-R8)."""
    lignes = []
    for p, cs in patients_de(calib, patient_de).items():
        s, _ = calibrer(ver, set(calib) - set(cs))
        for c in cs:
            lignes.append({"patient": p, "cas": c, "A_verite": ctl.appliquer(ver[c], s),
                           "B_prediction": ctl.appliquer(pred[c], s)})
    return lignes


def familles(ver, calib, strates):
    """Plages par famille d'épaisseur (14 cas) ; « sépare » si elles ne se recouvrent pas (indicatif)."""
    lire = {f"R2 volume {e} {n}": (lambda c, e=e: ver[c]["volumes_ml"][e]) for e, n in ctl.STRUCTURES.items()}
    lire.update({f"R3 {r}": (lambda c, r=r: ver[c]["rapports"][r]) for r, _, _ in ctl.RAPPORTS})
    lire.update({"R4 |aorte − VG|": lambda c: _abs(ver[c]["ecarts_hu"]["aorte_moins_VG"]),
                 "R5 |AP − VD|": lambda c: _abs(ver[c]["ecarts_hu"]["AP_moins_VD"]),
                 "R6 aorte − AP": lambda c: ver[c]["ecarts_hu"]["aorte_moins_AP"],
                 "R7 densité VG": lambda c: ver[c]["hu"]["VG"]})
    lire.update({f"R8 morceaux > 0,5 mL {e} {n}": (lambda c, e=e: ver[c]["morceaux"][e]["n_gros"])   # vérité (D34)
                 for e, n in ctl.STRUCTURES.items()})
    sortie = {}
    for g, f in lire.items():
        par = {}
        for fam in prep.STRATES_MM:
            v = [f(c) for c in calib if strates.get(c) == fam and f(c) is not None]
            par[str(fam)] = [min(v), max(v)] if v else None
        a, b = par[str(prep.STRATES_MM[0])], par[str(prep.STRATES_MM[1])]
        sortie[g] = {"plages": par, "separe": bool(a and b and (a[1] < b[0] or b[1] < a[0]))}
    return sortie


def _abs(x):
    return None if x is None else abs(x)


def concordance(ver, chemin_05e, chemin_05h):
    """Compare les mesures de la vérité à celles écrites par 05e et 05h (mêmes fonctions : égalité attendue)."""
    rapport = {}
    for nom, chemin, lire in (
            ("05e", chemin_05e, lambda c, l: [(f"{k}_ref", l.get(f"{k}_ref"), ver[c]["hu"][k]) for k in ("aorte", "AP", "VG", "VD")]),
            ("05h", chemin_05h, lambda c, l: [(f"ml_{n}", l.get(f"ml_{n}"), ver[c]["volumes_ml"][e]) for e, n in ctl.STRUCTURES.items()]
             + [(r, l.get(r), ver[c]["rapports"][r]) for r, _, _ in ctl.RAPPORTS])):
        if not os.path.isfile(chemin):
            rapport[nom] = {"fait": False, "raison": f"{chemin} absent"}
            continue
        with open(chemin, encoding="utf-8") as f:
            lignes = {l["cas"]: l for l in json.load(f).get("lignes", [])}
        ecarts, compares = [], 0
        for c in sorted(ver):
            if c not in lignes:
                ecarts.append(f"{c} absent de {nom}")
                continue
            for cle, ref, nous in lire(c, lignes[c]):
                compares += 1
                if ref != nous and not (ref is not None and nous is not None and abs(ref - nous) <= 1e-9):
                    ecarts.append(f"{c} {cle} : {nom} {ref} ≠ 05j {nous}")
        rapport[nom] = {"fait": True, "valeurs_comparees": compares, "ecarts": ecarts}
    return rapport


def critere(app, lopo_l, verite_1019):
    """Exigences du § 7 (Q3), chacune atteinte, non atteinte ou non évaluable. Ne décide rien."""
    e = {}
    if CAS_EXCLU in app:
        r = app[CAS_EXCLU]
        e["1019_en_alerte"] = {"etat": "atteint" if r["statut"] == "ALERTE" else "NON ATTEINT", "detail": ctl.resume(r)}
    else:
        e["1019_en_alerte"] = {"etat": "non évaluable (1019 non appliqué)", "detail": ""}
    ok = [c for c in CAS_DOIVENT_ETRE_OK if c in app]
    e["1001_1020_OK"] = ({"etat": "non évaluable", "detail": ""} if len(ok) < 2 else
                         {"etat": "atteint" if all(app[c]["statut"] == "OK" for c in ok) else "NON ATTEINT",
                          "detail": "; ".join(f"{c[-4:]} : {app[c]['statut']} {ctl.resume(app[c])}" for c in ok)})
    autres = [c for c in app if c != CAS_EXCLU]
    fausses = [c for c in autres if app[c]["statut"] == "ALERTE"]
    if len(autres) < 14:
        etat = "non évaluable (application partielle)"
    elif not fausses:
        etat = "atteint"
    elif len(fausses) == 1:
        etat = "atteint SOUS RÉSERVE : la fausse alerte doit être expliquée et consignée dans le journal (Q3)"
    else:
        etat = "NON ATTEINT"
    e["fausses_alertes_14_cas"] = {"etat": etat, "nombre": len(fausses),
                                   "detail": "; ".join(f"{c[-4:]} : {ctl.resume(app[c])}" for c in fausses) or "aucune"}
    fa = [l for l in lopo_l if l["A_verite"]["statut"] == "ALERTE"]
    hors_r6 = [a for a in verite_1019["regles_declenchees"] if a["regle"] != "R6"]
    e["lopo_verite"] = {"etat": "atteint" if not fa and not hors_r6 else "NON ATTEINT",
                        "nombre": len(fa) + (1 if hors_r6 else 0),
                        "detail": "; ".join([f"{l['cas'][-4:]} : {ctl.resume(l['A_verite'])}" for l in fa]
                                            + ([f"1019 (vérité, hors R6) : {', '.join(a['regle'] for a in hors_r6)}"] if hors_r6 else []))
                        or "aucune (1019 : " + (ctl.resume(verite_1019)) + ")"}
    etats = [x["etat"] for x in e.values()]
    if any(x.startswith("NON") for x in etats):
        glob = "NON ATTEINT : la ou les règles en cause sont à revoir (décision humaine, Q3) ; rien n'a été modifié"
    elif any(x.startswith("non évaluable") for x in etats):
        glob = "NON ÉVALUABLE (application partielle : relancer sans --cas)"
    elif any("SOUS RÉSERVE" in x for x in etats):
        glob = "ATTEINT SOUS RÉSERVE (une fausse alerte à expliquer et consigner, Q3)"
    else:
        glob = "ATTEINT"
    return e, glob


# --- Affichage -----------------------------------------------------------------------------------
def txt(x):
    return "—" if x is None else (f"{x:g}" if isinstance(x, float) else str(x))


def details(r):
    return " | ".join(f"{a['regle']} {a['objet']} : {a['mesure']} ; seuil {a['seuil']}" for a in r["regles_declenchees"])


def par_categorie(r, cat):
    return ", ".join(f"{a['regle']} ({a['objet']})" for a in r["regles_declenchees"] if a["categorie"] == cat) or "—"


def afficher_seuils(s, x):
    print("\n=== SEUILS PROPOSÉS (non figés) — calibrés sur les 14 cas sans 1019 ===")
    print("  (min/max observés et cas des extrêmes : affichés ici et écrits dans synthese_05j.json, JAMAIS dans "
          "seuils_proposes.json)")
    print(f"  R0 fixé : affine à {ctl.TOLERANCE_R0_MM} mm près, étiquettes 0 à 7 ; R1 fixé : volume > {ctl.VOLUME_MIN_R1_ML} mL")
    for e, n in ctl.STRUCTURES.items():
        p, o = s["R2"]["plages_ml"].get(str(e)), x[f"R2_{e}"]
        print(f"  R2 {e} {n:<12} volume (mL) dans [{txt(p and p['bas'])} ; {txt(p and p['haut'])}]   "
              f"(observé {txt(o and o['min'])} ({(o or {}).get('cas_min', '—')[-4:]}) à {txt(o and o['max'])} ({(o or {}).get('cas_max', '—')[-4:]}))")
    for r, _, _ in ctl.RAPPORTS:
        p, o = s["R3"]["plages"].get(r), x[f"R3_{r}"]
        print(f"  R3 {r:<24} dans [{txt(p and p['bas'])} ; {txt(p and p['haut'])}]   "
              f"(observé {txt(o and o['min'])} ({(o or {}).get('cas_min', '—')[-4:]}) à {txt(o and o['max'])} ({(o or {}).get('cas_max', '—')[-4:]}))")
    for regle, libelle, cle, quoi in (("R4", "|aorte − VG| ≤", "ecart_hu_max", "max"), ("R5", "|AP − VD| ≤", "ecart_hu_max", "max"),
                                      ("R6", "aorte − AP ≥", "ecart_aorte_moins_AP_min_hu", "min"),
                                      ("R7", "densité VG ≥", "hu_cavite_VG_min", "min")):
        o = x[regle]
        print(f"  {regle} {libelle} {txt(s[regle] and s[regle][cle])} HU   "
              f"(observé : {quoi} {txt(o and o[quoi])} HU, cas {(o or {}).get('cas_' + quoi, '—')[-4:]} ; marge {MARGE_HU} HU)")
    for e, n in ctl.STRUCTURES.items():
        t, on, op = s["R8"]["par_etiquette"].get(str(e)), x[f"R8_n_{e}"], x[f"R8_p_{e}"]
        print(f"  R8 {e} {n:<12} morceaux > 0,5 mL ≤ {txt(t and t['morceaux_gros_max'])} (max observé : "
              f"{(on or {}).get('cas_max', '—')[-4:]}) ; part du plus gros ≥ {txt(t and t['part_plus_gros_min'])} "
              f"(min observé : {(op or {}).get('cas_min', '—')[-4:]})")


def afficher_application(app, pli_de, strates, patient_de):
    print("\n=== APPLICATION aux prédictions hors pli post-traitées (seuils proposés, 14 cas sans 1019) ===")
    print(f"  {'cas':<6}{'pli':>4}{'épais.':>7} {'patient':<10}{'statut':<8}{'erreur_probable':<45}hors_domaine")
    for c, r in app.items():
        print(f"  {c[-4:]:<6}{pli_de[c]:>4}{txt(strates.get(c)):>7} {patient_de.get(c, ''):<10}{r['statut']:<8}"
              f"{par_categorie(r, 'erreur_probable'):<44} {par_categorie(r, 'hors_domaine')}")
    print("  Détail des règles déclenchées (mesure ; seuil) :")
    for c, r in app.items():
        for a in r["regles_declenchees"]:
            print(f"    {c[-4:]} {a['regle']} [{a['categorie']}] {a['objet']} : {a['mesure']} ; seuil {a['seuil']}")
        for ne in r["non_evaluees"]:
            print(f"    {c[-4:]} non évaluée : {ne}")


def afficher_lopo(lignes, verite_1019):
    print("\n=== EN LAISSANT UN PATIENT DE CÔTÉ (seuils recalculés sans le patient ; paires retirées ensemble) ===")
    print("  A = vérité, R0 à R8 (EXIGÉ, spéc. § 7) ; B = prédiction post-traitée, R0 à R8 (COMPLÉMENT, information :")
    print("  seule estimation des fausses alertes, de R8 surtout, sur des prédictions que les seuils n'ont pas vues)")
    print(f"  {'patient':<11}{'cas':<6}{'A vérité':<9}{'règles A':<40}{'B prédiction':<13}règles B")
    for l in lignes:
        print(f"  {l['patient']:<11}{l['cas'][-4:]:<6}{l['A_verite']['statut']:<9}{ctl.resume(l['A_verite']):<40}"
              f"{l['B_prediction']['statut']:<13}{ctl.resume(l['B_prediction'])}")
    print(f"  1019, vérité, seuils des 14 cas (seule R6 est admise, § 7) : {verite_1019['statut']} {ctl.resume(verite_1019)}")


# --- Écriture ------------------------------------------------------------------------------------
def ecrire(chemin, ecrire_contenu):
    """Fichier temporaire puis renommage : jamais de fichier à moitié écrit."""
    with open(chemin + ".tmp", "w", encoding="utf-8", newline="") as f:
        ecrire_contenu(f)
    os.replace(chemin + ".tmp", chemin)


def ecrire_csv(chemin, lignes):
    def contenu(f):
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
    ecrire(chemin, contenu)


def ligne_mesures(c, source, m, pli, famille, patient):
    l = {"cas": c, "source": source, "pli": pli, "famille_mm": famille, "patient": patient,
         "mesurable": m["mesurable"], "R0_problemes": " ; ".join(m["r0"])}
    for e, n in ctl.STRUCTURES.items():
        l[f"ml_{e}_{n}"] = m["volumes_ml"].get(e)
    for r, _, _ in ctl.RAPPORTS:
        l[r] = m["rapports"].get(r)
    for k in ("aorte", "AP", "VG", "VD"):
        l[f"hu_{k}"] = m["hu"].get(k)
    l["hu_repli_region_entiere"] = m["hu_repli_region_entiere"]
    for k in ("aorte_moins_VG", "AP_moins_VD", "aorte_moins_AP"):
        l[k] = m["ecarts_hu"].get(k)
    for e, n in ctl.STRUCTURES.items():
        mo = m["morceaux"].get(e) or {}
        l[f"morceaux_{e}_{n}"], l[f"morceaux_gros_{e}_{n}"], l[f"part_plus_gros_{e}_{n}"] = mo.get("n"), mo.get("n_gros"), mo.get("part_plus_gros")
    l["anomalies"] = " ; ".join(m["anomalies"])
    return l


# --- Programme principal -----------------------------------------------------------------------
def main():
    a = lire_arguments()
    sortie, chemins, appliques, pli_de, strates, patient_de, cas = verifier(a)
    calib = sorted(cas - {CAS_EXCLU})

    # 1. Mesures : vérité et prédiction de chaque cas (un scanner à la fois).
    ver, pred = {}, {}
    for c, ch in chemins.items():
        print(f"  {c} (pli {pli_de[c]}) : mesures de la vérité (labelsTr) et de la prédiction (postprocessed)...", flush=True)
        ver[c] = ctl.mesurer(c, ch["image"], ch["verite"])
        pred[c] = ctl.mesurer(c, ch["image"], ch["prediction"])
    anomalies = {f"{c} {src}": m["anomalies"] + m["r0"] for src, d in (("vérité", ver), ("prédiction", pred))
                 for c, m in d.items() if m["anomalies"] or m["r0"]}
    for qui, liste in anomalies.items():
        for x in liste:
            print(f"ANOMALIE {qui} : {x}")
    # Calibrage impossible si un cas de calibrage n'a pas toutes ses mesures de vérité (spéc. : toutes les
    # structures sont présentes dans la vérité des 15 cas). Rien n'est écrit dans ce cas.
    incomplets = [c for c in calib if not ver[c]["mesurable"]
                  or any(v is None for v in [*ver[c]["volumes_ml"].values(), *ver[c]["rapports"].values(),
                                             *ver[c]["ecarts_hu"].values(), ver[c]["hu"]["VG"]])]
    if incomplets:
        arret(6, f"calibrage impossible : vérité non mesurable ou incomplète pour {incomplets} (voir les ANOMALIE) ; "
                 "rien n'est écrit.")
    # Une prédiction de forme différente de l'image ne peut pas être contrôlée voxel à voxel : on s'arrête.
    non_mesurables = [c for c in chemins if not pred[c]["mesurable"]]
    if non_mesurables:
        arret(6, f"prédiction non mesurable (forme différente de l'image) pour {non_mesurables} (voir les ANOMALIE) ; "
                 "rien n'est écrit.")

    # 2. Calibrage, 3. vérification en laissant un patient de côté, 4. application.
    seuils, ext = calibrer(ver, calib)
    limites_r8 = limites_de_r8(seuils)
    lopo_l = lopo(ver, pred, calib, patient_de)
    verite_1019 = ctl.appliquer(ver[CAS_EXCLU], seuils)
    app = {c: ctl.appliquer(pred[c], seuils) for c in appliques}
    fam = familles(ver, calib, strates)
    conc = concordance(ver, a.ref_05e, a.ref_05h)
    exig, glob = critere(app, lopo_l, verite_1019)
    n_patients = len(patients_de(calib, patient_de))
    fichier_seuils = entete_seuils(seuils, len(calib), n_patients, prep.sha256_fichier(a.split), prep.commit_git())
    identifiants = verifier_sans_identifiant(fichier_seuils, list(chemins))
    if identifiants:
        arret(6, "identifiant de cas trouvé dans le fichier de seuils (interdit, Q6) : " + " ; ".join(identifiants))

    # Affichage.
    afficher_seuils(seuils, ext)
    print("  Limites de R8 (affichées seulement, rien n'est ajusté) : " + ("aucune" if not limites_r8 else ""))
    for x in limites_r8:
        print(f"    - {x}")
    afficher_application(app, pli_de, strates, patient_de)
    afficher_lopo(lopo_l, verite_1019)
    print("\n=== FAMILLES D'ÉPAISSEUR DE COUPE (indicatif, § 6.3 ; aucun seuil par famille) ===")
    separes = [g for g, f in fam.items() if f["separe"]]
    print("  Grandeurs dont les plages 0,45 mm et 0,625 mm ne se recouvrent pas : " + (", ".join(separes) or "aucune"))
    for f in prep.STRATES_MM:
        cs = [c for c in app if strates.get(c) == f and c != CAS_EXCLU]
        x = f" ; 1019 : {app[CAS_EXCLU]['statut']}" if CAS_EXCLU in app and strates.get(CAS_EXCLU) == f else ""
        print(f"  Famille {f} mm : fausses alertes {sum(app[c]['statut'] == 'ALERTE' for c in cs)} sur {len(cs)} cas "
              f"(1019 non compté){x}")
    print("\n=== CONCORDANCE avec les sections 18 (05e) et 21 (05h), vérité ===")
    for nom, r in conc.items():
        print(f"  {nom} : " + (r["raison"] + " : contrôle non fait" if not r["fait"] else
                               f"{r['valeurs_comparees']} valeurs comparées, {len(r['ecarts'])} écart(s)"))
        for x in r.get("ecarts", [])[:20]:
            print(f"AVERTISSEMENT concordance : {x}")
    avertissements = [
        "seuils PROPOSÉS, NON FIGÉS : rien n'est écrit dans config/ ; validation humaine, puis gel (date, SHA-256) "
        "avant l'étape 6 (spéc. § 6.2)",
        "la vérité des 14 cas a servi au calibrage de R2 à R8 (leurs prédictions, non) : l'application à leurs "
        "prédictions n'est pas totalement indépendante ; la vérification par patient laissé de côté (B) le complète",
        *[f"limite de R8 (D34) : {x}" for x in limites_r8],
        "1019 signalé = vérification de cohérence, pas preuve d'efficacité : les règles ont été conçues après l'avoir vu "
        "(spéc. § 6.4) ; la sensibilité ne peut pas être mesurée"]
    if separes:
        avertissements.append(f"plages disjointes entre familles d'épaisseur (indicatif) : {', '.join(separes)}")
    print("\n=== RÉSUMÉ ===")
    for nom, x in exig.items():
        print(f"  {nom:<26}{x['etat']}" + (f" — {x['detail']}" if x["detail"] else ""))
    print(f"  CRITÈRE (§ 7, Q3) : {glob}")
    print("  (Le script affiche ; il ne modifie ni marge ni règle. Toute révision est une décision humaine, "
          "consignée dans le journal avant le gel.)")
    for x in avertissements:
        print(f"AVERTISSEMENT : {x}")

    # Écriture (seulement dans --sortie).
    os.makedirs(sortie, exist_ok=True)
    ecrire(os.path.join(sortie, NOMS["seuils"]), lambda f: json.dump(fichier_seuils, f, indent=2, ensure_ascii=False))
    ecrire_csv(os.path.join(sortie, NOMS["mesures"]),
               [ligne_mesures(c, src, d[c], pli_de[c], strates.get(c), patient_de.get(c, ""))
                for c in chemins for src, d in (("verite", ver), ("prediction", pred))])
    ecrire_csv(os.path.join(sortie, NOMS["application"]),
               [{"cas": c, "pli": pli_de[c], "famille_mm": strates.get(c), "patient": patient_de.get(c, ""),
                 "role": "exclu du calibrage (D28)" if c == CAS_EXCLU else "calibrage",
                 "statut": r["statut"], "nb_regles": len(r["regles_declenchees"]),
                 "erreur_probable": par_categorie(r, "erreur_probable"), "hors_domaine": par_categorie(r, "hors_domaine"),
                 "details": details(r), "non_evaluees": " ; ".join(r["non_evaluees"])} for c, r in app.items()])
    ecrire_csv(os.path.join(sortie, NOMS["lopo"]),
               [{"patient": l["patient"], "cas": l["cas"], "A_verite_statut": l["A_verite"]["statut"],
                 "A_regles": ctl.resume(l["A_verite"]), "B_prediction_statut": l["B_prediction"]["statut"],
                 "B_regles": ctl.resume(l["B_prediction"]), "B_details": details(l["B_prediction"])} for l in lopo_l])
    synthese = {"objet": "calibrage du contrôle de plausibilité et application aux prédictions hors pli post-traitées "
                         "(validation, PAS le test) ; seuils proposés, non figés",
                "date_utc": fichier_seuils["date_calcul_utc"], "split_sha256": prep.sha256_fichier(a.split),
                "code_commit": prep.commit_git(), "cas_calibrage": calib, "cas_exclu": CAS_EXCLU,
                "cas_appliques": appliques, "seuils_proposes": fichier_seuils, "extremes_observes": ext,
                "application": app, "lopo": lopo_l, "verite_1019_seuils_14_cas": verite_1019,
                "critere": {"exigences": exig, "global": glob}, "limites_R8": limites_r8, "familles": fam,
                "concordance": conc,
                "avertissements": avertissements, "anomalies": anomalies,
                "mesures": {"verite": ver, "prediction": pred}, "pli": pli_de, "sources": chemins}
    ecrire(os.path.join(sortie, NOMS["synthese"]), lambda f: json.dump(synthese, f, indent=2, ensure_ascii=False))
    print(f"\nÉcrit dans {sortie} : {', '.join(NOMS.values())}")
    print("Rien n'a été écrit dans config/, dans le dépôt, ni dans nnUNet_raw / nnUNet_preprocessed / nnUNet_results ; "
          "aucun cas de test lu.")


if __name__ == "__main__":
    main()
