"""
Étape 6 — Module de CALCUL commun aux phases A (06b, essai à blanc) et C (06d, ouverture du test).
Protocole figé docs/protocole_etape6.md (D42), sections 4 à 8. Le MÊME code mesure en A et en C : c'est ce
qui fait de la phase A une qualification de l'outil de mesure (P9).

Ce que fait le module : il n'exécute RIEN à l'import et ne lit AUCUN fichier de données. Ses fonctions
travaillent sur des tableaux numpy déjà chargés (masques) et sur l'espacement des voxels (mm) ; les scripts
06b et 06d lisent les fichiers (en passant par la garde de etape6_commun) et lui donnent les tableaux.

Entrées : tableaux numpy (prédiction, vérité), espacement et forme de l'image, résultats du contrôle de
  plausibilité (format de controle_plausibilite.appliquer), seuils figés. Sorties : dictionnaires Python ;
  ecrire_tableaux() écrit les CSV, le JSON complet et le rapport Markdown dans un dossier donné.

=========================================================================================================
INTERFACE PUBLIQUE
=========================================================================================================
Mesure (§ 5.2, § 6)
  plafond_hd95(forme, espacement) -> float           diagonale physique de l'image, √(Σ (nᵢ × sᵢ)²), en mm (D1).
  metriques_structure(p, v, espacement, forme) -> dict   une structure : Dice, HD95, NSD 2 mm, volumes, écarts,
                                                     cas dégénéré D1 / D2 / D3 traité AVANT MONAI.
  metriques_cas(pred, ref, espacement, forme, etiquettes, geometrie_ok=True, motif="") -> dict
                                                     {k: metriques_structure} ; géométrie différente = tout en D1 (§ 6).
  geometrie(img_a, img_b) -> (ok, motif)             même forme, affine à TOLERANCE_R0_MM près.
  mesurer_prediction(pred, ref, espacement, forme, etiquettes, resultat, restreint, geometrie_ok, motif)
                                       -> (métriques, égarés)   un modèle sur un cas, + morceaux égarés pour R8.
  dice_moyen(res_cas, etiquettes) -> (moyenne, n)    agrégat « Dice moyen des structures » (§ 5.3).
Agrégats (§ 5.3)
  agreger(par_cas, etiquettes, cas_retenus=None) -> {k: {metrique: {n, mediane, moyenne, minimum|maximum, ...}}}
Contrôle de plausibilité (§ 4 c, § 7)
  lire_controle(contenu, modele, cas_attendus, seuils, sha_seuils) -> {cas: {"mesures", "resultat", "resume"}}
                                                     lecteur TOLÉRANT du fichier de contrôle de la phase B
                                                     (contrat ci-dessous) ; ValueError si non conforme.
  restreindre_T(resultat) -> resultat                restriction de (c) À LA LECTURE (R0, R7 ; R1, R2, R8 sur 1 à 5).
  besoins_classement(resultat) -> (etiquettes_R8, verite_densites)   ce qu'il faut mesurer pour classer.
  morceaux_egares(p, v, vox_ml) -> dict              morceaux > 0,5 mL (connexité 26, recoupé avec morceaux() de
                                                     05i) qui recouvrent moins de la moitié de leurs voxels avec v.
  classer_alertes(...) -> liste                      vrai / faux / non classable pour chaque alerte (§ 7.3) ;
                                                     « non classable » = DEFINITION_NON_CLASSABLE (D43) ; tout
                                                     autre motif lève ArretClassement (code 23).
  echecs_silencieux(par_cas, controles, etiquettes) -> liste      Dice < 0,50 sans aucune alerte sur le cas.
  estimation_fausses_alertes(alertes, cas) -> dict   « x sur N » (erreur probable) et hors domaine à part.
Verdict et comparaison (§ 8)
  verdict(agregats_a, echecs_a) -> dict              C1 (médianes non arrondies) et C2 ; « acceptable » ou non.
  regle_p6(par_cas_x, par_cas_y, etiquettes) -> dict « meilleure sur k » ou « pas de différence démontrée »
                                                     (5 cas utilisables exigés ; écart > 0,02 strict, D43).
  evaluer(entrees) -> dict                           les trois comparaisons d'un coup (format de `entrees` plus bas).
  ecrire_tableaux(dossier, resultats) -> [fichiers]  CSV, resultats_etape6.json, rapport_etape6.md (arrondis § 8.1
                                                     à l'affichage seulement ; les CSV et le JSON gardent les valeurs
                                                     non arrondies).
  publier(dossier_final, res, titre, avertissement, autres_json, tmp) -> [fichiers]   tout en une fois : dossier
                                                     temporaire puis renommage ; refuse d'écraser.
  normaliser_mesures(m) -> dict                      mesures du module de contrôle après un passage par JSON.

CONTRAT DU FICHIER DE CONTRÔLE (phase B → phase C ; lu par lire_controle)
  controle_<M>.json, M ∈ {"A", "N0", "S0", "T"} :
    {"modele": M, "seuils_sha256": <SHA-256 de config/seuils_plausibilite.json>,
     "cas": {"1004": {"mesures": <controle_plausibilite.mesurer(...)>, "resultat": <appliquer(mesures, seuils)>,
                      "resume": <resume(resultat)>}, ... les 5 cas}}
  avec appliquer() appelé sur LES 9 RÈGLES R0 à R8 pour les 4 modèles (la restriction de T se fait ici, à la
  lecture). Conversions JSON tolérées : clés entières devenues chaînes ("1" pour 1 dans volumes_ml, voxels,
  morceaux), tuples devenus listes, cas donné en nombre (1004) ou en « mmwhs_1004 ». Contrôles faits : modèle
  et empreinte des seuils attendus ; exactement les cas attendus ; statut OK / ALERTE cohérent avec les règles
  déclenchées ; regles_verifiees = R0 à R8 ; resume = resume(resultat) ; et appliquer() REJOUÉ sur les mesures
  lues redonne exactement le résultat lu (le résultat figé est bien celui de ces mesures et de ces seuils).

Vocabulaire :
  - masque binaire : pour la structure k, vrai là où le voxel vaut k, faux ailleurs (« k contre le reste ») ;
  - cas dégénéré : la prédiction ou la vérité de la structure est vide (§ 6) ; D1 = prédiction vide (échec,
    compté avec la pire valeur) ; D2 = vérité vide ; D3 = les deux vides (D2 et D3 : exclus, n diminue) ;
  - valeur plafond : la plus grande distance possible dans l'image, donnée à la HD95 d'une structure oubliée ;
  - n : nombre de cas réellement pris en compte dans un agrégat ;
  - biais : moyenne (ou médiane) des écarts SIGNÉS : tendance à trop (+) ou trop peu (−) segmenter.
"""
import csv, importlib.util, io, json, math, os, statistics

ICI = os.path.dirname(os.path.abspath(__file__))


def _charger(nom, fichier):
    spec = importlib.util.spec_from_file_location(nom, os.path.join(ICI, fichier))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)              # le module de contrôle et 05i n'exécutent rien à l'import
    return module


cp = _charger("controle_plausibilite_e6m", "controle_plausibilite.py")   # module FIGÉ (D36) : jamais modifié
pp = cp.pp                                                                # 05i, tel que le contrôle l'utilise

# --- Constantes du protocole (aucune n'est réglable) -------------------------------------------------
STRUCTURES = dict(cp.STRUCTURES)                                 # {1: "myocarde_VG", ..., 7: "AP"} (dataset.json)
NOMS = {1: "myocarde VG", 2: "OG", 3: "cavité VG", 4: "OD", 5: "VD", 6: "aorte", 7: "AP"}
TOUTES_ETIQUETTES = (1, 2, 3, 4, 5, 6, 7)
ETIQUETTES_C = (1, 2, 3, 4, 5)                                   # (c) TotalSegmentator : 1 à 5 (D23)
TAU_NSD_MM = 2.0                                                 # NSD à 2 mm (P2, D30)
PERCENTILE_HD = 95                                               # HD95 (§ 5.2)
SEUILS_C1 = {1: 0.85, 2: 0.85, 3: 0.85, 4: 0.85, 5: 0.85, 6: 0.85, 7: 0.80}   # § 8.4 (P1)
REPERE_DICE_FAIBLE = 0.85                                        # § 7.3 (D30)
REPERE_ECART_VOLUME_PCT = 30.0                                   # § 7.3, R2 (D29 Q2)
REPERE_ECHEC = 0.50                                              # échec silencieux (§ 7.3, D30)
P6_ECART_MEDIANE, P6_CAS_MIN = 0.02, 4                           # P6 : > 0,02 et mieux sur au moins 4 cas sur 5
P6_CAS_REQUIS = 5                                                # P6 appliquée seulement avec 5 cas utilisables (D43)
P6_PAS_DE_DIFFERENCE = "pas de différence démontrée"             # formule unique de P6 (D43)
CODE_ARRET_CLASSEMENT = 23                                       # défaut du code ou des données en classant (D43)
# Définition EXACTE de la catégorie « non classable » (décision de l'utilisateur, déviation connue D43).
DEFINITION_NON_CLASSABLE = (
    "Une alerte est non classable quand la grandeur que la règle de classement du § 7.3 doit comparer n'est pas "
    "définie pour ce cas : (i) la vérité de la structure en cause est vide (cas dégénérés D2 ou D3, § 6), donc Dice "
    "et écart de volume sont non définis ; (ii) la mesure de densité exigée par R6 ou R7 ne peut pas être faite sur "
    "la vérité (structure absente ou trop petite pour le module figé, ou vérité que le module figé déclare non "
    "mesurable, par ex. pour sa géométrie). Une alerte non classable n'est ni vraie ni "
    "fausse : elle est rapportée à part, avec sa raison, et n'entre ni dans l'estimation des fausses alertes ni "
    "dans le compte des vraies alertes. Règle à plusieurs conditions reliées par « ou » (R3, R4, R5, R8) : l'alerte "
    "est vraie dès qu'UNE condition définie est vraie ; non classable seulement si aucune condition définie n'est "
    "vraie et qu'au moins une n'est pas définie ; fausse si toutes sont définies et fausses.")
NC_VERITE_VIDE = "(i) vérité vide"                               # préfixe de la raison, motif (i)
NC_DENSITE = "(ii) densité non mesurable sur la vérité"          # préfixe de la raison, motif (ii)
REGLES_C = ("R0", "R1", "R2", "R7", "R8")                        # (c) : R3 à R6 non appliquées (§ 4 c)
REGLES_PAR_STRUCTURE = ("R1", "R2", "R8")
# Structures dont dépend chaque règle « erreur probable » portant sur deux structures (§ 7.3).
STRUCTURES_REGLE = {"R3": (7, 6), "R4": (6, 3), "R5": (7, 5)}
PRUDENCE_C = ("Prudence (D23) : sur deux cas d'entraînement, les volumes de TotalSegmentator s'écartaient de la "
              "vérité (1002 : OD +46 %, VD −18 %), avec des frontières moins nettes. Un mauvais score sur l'OD ou "
              "le VD peut venir de la définition de la structure, pas de la qualité de l'outil.")
DEFINITION_HD95 = ("HD95 symétrique de MONAI 1.6.1 : 95e percentile des distances bord de P → bord de V, idem "
                   "V → P, et le plus grand des deux ; en mm avec l'espacement de l'image. Non directement "
                   "comparable aux HD95 calculées par d'autres logiciels (percentile de l'union des distances).")


# =================================================================================================
# Mesure d'une structure (§ 5.2) et cas dégénérés (§ 6)
# =================================================================================================
def plafond_hd95(forme, espacement):
    """Diagonale physique de l'image (mm) : la plus grande distance possible dans ce scanner (P3, D1)."""
    return math.sqrt(sum((float(n) * float(s)) ** 2 for n, s in zip(forme[:3], espacement[:3])))


def _boite(p, v):
    """Coupe autour de l'union des deux masques (1 voxel de marge, limitée à l'image). MONAI recadre déjà
    lui-même sur cette union (get_mask_edges, crop=True) : le résultat est identique, mais la mémoire
    utilisée passe de plusieurs Go (scanner entier en nombres décimaux) à quelques dizaines de Mo."""
    import numpy as np
    u = p | v
    bornes = []
    for ax in range(3):
        idx = np.where(u.any(axis=tuple(x for x in range(3) if x != ax)))[0]
        bornes.append(slice(max(int(idx.min()) - 1, 0), min(int(idx.max()) + 2, u.shape[ax])))
    return p[tuple(bornes)], v[tuple(bornes)]


def _monai(p, v, espacement):
    """Dice, HD95 et NSD à 2 mm par MONAI 1.6.1, sur une seule structure présente des deux côtés.
    Un seul canal (la structure) : le fond n'est pas transmis, donc exclu (include_background=True porte sur ce
    seul canal ; c'est la même chose que deux canaux fond + structure avec include_background=False)."""
    import warnings
    import torch
    from monai.metrics import compute_dice, compute_hausdorff_distance, compute_surface_dice
    bp, bv = _boite(p, v)
    tp = torch.from_numpy(bp[None, None].astype("float32"))
    tv = torch.from_numpy(bv[None, None].astype("float32"))
    esp = [float(x) for x in espacement[:3]]
    with warnings.catch_warnings():              # avertissement interne de MONAI 1.6.1 (option « dépréciée »), sans effet
        warnings.simplefilter("ignore", FutureWarning)
        dice = float(compute_dice(tp, tv, include_background=True)[0, 0])
        hd = float(compute_hausdorff_distance(tp, tv, include_background=True, percentile=PERCENTILE_HD,
                                              directed=False, spacing=esp)[0, 0])
        nsd = float(compute_surface_dice(tp, tv, class_thresholds=[TAU_NSD_MM], include_background=True,
                                         spacing=esp)[0, 0])
    if not all(math.isfinite(x) for x in (dice, hd, nsd)):
        raise DefautCalcul(f"MONAI a renvoyé une valeur non finie sur une structure non vide : {dice}, {hd}, {nsd}")
    return dice, hd, nsd


def metriques_structure(p, v, espacement, forme):
    """Toutes les métriques d'une structure. p, v : masques booléens (prédiction, vérité) de même forme.
    Le volume utilise l'espacement de l'IMAGE (comme 05h et le contrôle). Cas dégénérés traités AVANT MONAI."""
    import numpy as np
    vox_ml = float(np.prod([float(x) for x in espacement[:3]])) / 1000
    n_p, n_v = int(p.sum()), int(v.sum())
    r = {"n_pred": n_p, "n_ref": n_v, "vol_pred_ml": n_p * vox_ml, "vol_ref_ml": n_v * vox_ml,
         "dice": None, "hd95_mm": None, "hd95_plafond": False, "nsd_2mm": None, "cas_degenere": None, "anomalie": None}
    r["ecart_ml"] = r["vol_pred_ml"] - r["vol_ref_ml"]
    r["ecart_pct"] = 100 * r["ecart_ml"] / r["vol_ref_ml"] if n_v else None
    if n_p == 0 and n_v > 0:                     # D1 : structure oubliée = échec, compté avec la pire valeur
        r.update(dice=0.0, hd95_mm=plafond_hd95(forme, espacement), hd95_plafond=True, nsd_2mm=0.0, cas_degenere="D1",
                 ecart_ml=-r["vol_ref_ml"], ecart_pct=-100.0)         # valeurs exactes du § 6 (pas d'arrondi flottant)
    elif n_v == 0:                               # D2 (prédiction présente) ou D3 (les deux vides) : exclus
        r["cas_degenere"] = "D2" if n_p else "D3"
        r["anomalie"] = ("vérité vide, prédiction présente (D2) : anomalie de la vérité, cas exclu des agrégats"
                         if n_p else "vérité et prédiction vides (D3) : anomalie, cas exclu des agrégats")
    else:
        r["dice"], r["hd95_mm"], r["nsd_2mm"] = _monai(p, v, espacement)
    return r


def geometrie(img_a, img_b):
    """(ok, motif) : même forme (3 premiers axes) et affines égales à TOLERANCE_R0_MM près (R0)."""
    import numpy as np
    if tuple(img_a.shape[:3]) != tuple(img_b.shape[:3]):
        return False, f"forme {tuple(img_a.shape[:3])} ≠ {tuple(img_b.shape[:3])}"
    ecart = float(np.abs(np.asarray(img_a.affine) - np.asarray(img_b.affine)).max())
    if ecart > cp.TOLERANCE_R0_MM:
        return False, f"affine différente (écart max {ecart:.4f} mm > {cp.TOLERANCE_R0_MM})"
    return True, ""


def metriques_cas(pred, ref, espacement, forme, etiquettes, geometrie_ok=True, motif=""):
    """{k: métriques} pour un cas. Si la prédiction n'a pas la géométrie de la vérité (forme ou affine), aucun
    rééchantillonnage : toutes les structures sont traitées comme D1 et l'anomalie est notée (§ 6)."""
    import numpy as np
    res = {}
    for k in etiquettes:
        v = ref == k
        if geometrie_ok:
            res[k] = metriques_structure(pred == k, v, espacement, forme)
        else:
            res[k] = metriques_structure(np.zeros_like(v), v, espacement, forme)
            res[k]["anomalie"] = f"géométrie de la prédiction différente de la vérité ({motif}) : traitée comme D1 (§ 6)"
    return res


def mesurer_prediction(pred, ref, espacement, forme, etiquettes, resultat, restreint=False, geometrie_ok=True, motif=""):
    """Métriques d'une prédiction sur un cas, et morceaux égarés des structures visées par une alerte R8 du
    contrôle (restreint à 1-5 pour T). Renvoie (métriques {k: ...}, égarés {k: ...})."""
    import numpy as np
    met = metriques_cas(pred, ref, espacement, forme, etiquettes, geometrie_ok, motif)
    k8, _ = besoins_classement(restreindre_T(resultat) if restreint else resultat)
    vox_ml = float(np.prod([float(x) for x in espacement[:3]])) / 1000
    eg = {}
    for k in k8:
        if geometrie_ok:
            eg[k] = morceaux_egares(pred == k, ref == k, vox_ml)
        else:                                    # grilles différentes : recouvrement impossible ; Dice = 0 (D1)
            eg[k] = {"n_gros": None, "egares": [], "un_egare": False, "note": "géométrie différente (D1)"}
    return met, eg


def dice_moyen(res_cas, etiquettes):
    """Moyenne des Dice définis des structures demandées, et leur nombre (§ 5.3)."""
    d = [res_cas[k]["dice"] for k in etiquettes if res_cas[k]["dice"] is not None]
    return (statistics.fmean(d) if d else None), len(d)


# =================================================================================================
# Agrégats (§ 5.3)
# =================================================================================================
def _stats(valeurs, pire):
    v = [x for x in valeurs if x is not None]
    s = {"n": len(v), "mediane": statistics.median(v) if v else None, "moyenne": statistics.fmean(v) if v else None}
    s[pire] = (min(v) if pire == "minimum" else max(v)) if v else None
    return s


def _stats_volume(valeurs):
    v = [x for x in valeurs if x is not None]
    pire = max(v, key=abs) if v else None                       # plus grande valeur absolue, avec son signe
    return {"n": len(v), "mediane_signee": statistics.median(v) if v else None,
            "moyenne_signee": statistics.fmean(v) if v else None,
            "mediane_absolue": statistics.median([abs(x) for x in v]) if v else None, "pire": pire}


def agreger(par_cas, etiquettes, cas_retenus=None):
    """par_cas = {cas: {k: métriques}}. D2 et D3 exclus (n diminué) ; D1 compté avec ses valeurs."""
    cas = [c for c in sorted(par_cas) if cas_retenus is None or c in cas_retenus]
    ag = {}
    for k in etiquettes:
        lignes = [par_cas[c][k] for c in cas if par_cas[c][k]["cas_degenere"] not in ("D2", "D3")]
        ag[k] = {"dice": _stats([l["dice"] for l in lignes], "minimum"),
                 "nsd_2mm": _stats([l["nsd_2mm"] for l in lignes], "minimum"),
                 "hd95_mm": _stats([l["hd95_mm"] for l in lignes], "maximum"),
                 "ecart_ml": _stats_volume([l["ecart_ml"] for l in lignes]),
                 "ecart_pct": _stats_volume([l["ecart_pct"] for l in lignes]),
                 "hd95_plafonds": sum(l["hd95_plafond"] for l in lignes),
                 "exclus": [c for c in cas if par_cas[c][k]["cas_degenere"] in ("D2", "D3")]}
    return ag


# =================================================================================================
# Contrôle de plausibilité : lecture (contrat de la phase B), restriction de T, classement (§ 7.3)
# =================================================================================================
def _cle_cas(x):
    s = str(x)
    return s.rsplit("_", 1)[-1] if "_" in s else s


def _cles_entieres(d):
    return {int(k): v for k, v in d.items()}


def normaliser_mesures(m):
    """Mesures de controle_plausibilite.mesurer() après un passage par JSON : clés d'étiquettes remises en
    entiers (volumes_ml, voxels, morceaux) ; le reste est déjà au bon format (listes, nombres, None)."""
    m = json.loads(json.dumps(m))
    for cle in ("volumes_ml", "voxels", "morceaux"):
        m[cle] = _cles_entieres(m.get(cle) or {})
    return m


def _json(x):
    return json.loads(json.dumps(x, ensure_ascii=False))


def lire_controle(contenu, modele, cas_attendus, seuils, sha_seuils):
    """Lit et vérifie un fichier de contrôle (contrat en tête du module). Renvoie {cas: {mesures, resultat,
    resume}} (mesures normalisées). Lève ValueError avec la liste des problèmes."""
    pb = []
    if contenu.get("modele") != modele:
        pb.append(f"modele = {contenu.get('modele')!r}, attendu {modele!r}")
    if contenu.get("seuils_sha256") != sha_seuils:
        pb.append(f"seuils_sha256 = {contenu.get('seuils_sha256')!r}, attendu {sha_seuils}")
    cas = {_cle_cas(c): e for c, e in (contenu.get("cas") or {}).items()}
    if sorted(cas) != sorted(cas_attendus):
        pb.append(f"cas {sorted(cas)}, attendu {sorted(cas_attendus)}")
    sortie = {}
    for c in sorted(set(cas) & set(cas_attendus)):
        e = cas[c]
        try:
            m, r = normaliser_mesures(e["mesures"]), _json(e["resultat"])
        except (KeyError, TypeError, ValueError, AttributeError) as err:
            pb.append(f"{c} : entrée illisible ({err!r})")
            continue
        if r.get("statut") not in ("OK", "ALERTE") or (r["statut"] == "ALERTE") != bool(r.get("regles_declenchees")):
            pb.append(f"{c} : statut {r.get('statut')!r} incohérent avec {len(r.get('regles_declenchees') or [])} règle(s)")
        if [x for x in r.get("regles_verifiees", []) if not x.startswith("R0 (forcée")] != list(cp.TOUTES):
            pb.append(f"{c} : règles vérifiées {r.get('regles_verifiees')}, attendu R0 à R8 complètes")
        if e.get("resume") != cp.resume(r):
            pb.append(f"{c} : resume {e.get('resume')!r} ≠ {cp.resume(r)!r}")
        try:
            rejoue = _json(cp.appliquer(m, seuils))
        except (KeyError, TypeError, ValueError) as err:
            rejoue = f"erreur {err!r}"
        if rejoue != r:
            pb.append(f"{c} : appliquer() rejoué sur les mesures lues ne redonne pas le résultat figé")
        sortie[c] = {"mesures": m, "resultat": r, "resume": e.get("resume")}
    if pb:
        raise ValueError(f"contrôle du modèle {modele} non conforme :\n  - " + "\n  - ".join(pb))
    return sortie


def _etiquette_objet(alerte):
    """Étiquette visée par une alerte R1, R2 ou R8 (objet « 4 OD ») ; None sinon."""
    tete = str(alerte.get("objet", "")).split(" ", 1)[0]
    return int(tete) if alerte.get("regle") in REGLES_PAR_STRUCTURE and tete.isdigit() else None


def restreindre_T(resultat):
    """(c) : on ÉCARTE À LA LECTURE les résultats qui portent sur l'aorte et l'AP (§ 4 c) : on garde R0 et R7,
    et R1, R2, R8 sur les structures 1 à 5 ; R3 à R6 sont ignorées. Le module et les seuils ne changent pas."""
    garder = lambda a: a["regle"] in ("R0", "R7") or (a["regle"] in REGLES_PAR_STRUCTURE
                                                       and _etiquette_objet(a) in ETIQUETTES_C)
    dec = [a for a in resultat["regles_declenchees"] if garder(a)]
    ignorees = [a for a in resultat["regles_declenchees"] if not garder(a)]
    non_eval = [x for x in resultat.get("non_evaluees", []) if x.split(" ", 1)[0] in REGLES_C
                and not any(f" {k} " in f" {x} " for k in ("6", "7"))]
    return {"statut": "ALERTE" if dec else "OK", "regles_declenchees": dec,
            "regles_verifiees": [r for r in resultat.get("regles_verifiees", []) if r.split(" ", 1)[0] in REGLES_C],
            "non_evaluees": non_eval, "ignorees_restriction_c": ignorees}


def besoins_classement(resultat):
    """(étiquettes dont il faut chercher les morceaux égarés (R8), faut-il mesurer les densités de la vérité
    (R6, R7)) pour classer les alertes d'un cas."""
    r8 = sorted({_etiquette_objet(a) for a in resultat["regles_declenchees"] if a["regle"] == "R8"} - {None})
    return r8, any(a["regle"] in ("R6", "R7") for a in resultat["regles_declenchees"])


def morceaux_egares(p, v, vox_ml):
    """Morceaux de la prédiction p (connexité 26) de plus de 0,5 mL qui recouvrent MOINS DE LA MOITIÉ de leurs
    voxels avec la vérité v (§ 7.3, R8). Même découpage que morceaux() de 05i (scipy.ndimage.label, structure
    3×3×3, « plus de » MORCEAU_ML) ; le nombre de morceaux est recoupé avec morceaux() elle-même."""
    import numpy as np
    from scipy import ndimage
    if not p.any():
        return {"n_gros": 0, "egares": [], "un_egare": False}
    bornes = [np.where(p.any(axis=tuple(x for x in range(3) if x != ax)))[0] for ax in range(3)]
    s = tuple(slice(int(b.min()), int(b.max()) + 1) for b in bornes)
    etiq, nb = ndimage.label(p[s], structure=np.ones((3, 3, 3)))
    tailles = np.bincount(etiq.ravel())[1:]
    commun = np.bincount(etiq.ravel(), weights=v[s].ravel().astype(np.float64), minlength=nb + 1)[1:]
    gros = tailles * vox_ml > pp.MORCEAU_ML
    ref = pp.morceaux(p, vox_ml)
    if ref["n"] != nb or ref["n_gros"] != int(gros.sum()):
        raise DefautCalcul(f"découpage en morceaux différent de morceaux() de 05i ({nb}/{int(gros.sum())} ≠ "
                           f"{ref['n']}/{ref['n_gros']})")
    egares = [{"ml": float(t * vox_ml), "part_recouverte": float(c / t)}
              for t, c, g in zip(tailles, commun, gros) if g and c < 0.5 * t]
    return {"n_gros": int(gros.sum()), "egares": egares, "un_egare": bool(egares)}


def _dice_faible(dices):
    """« Dice(x) < 0,85 ou Dice(y) < 0,85 » sur valeurs non arrondies. None = non défini (vérité vide)."""
    definis = [d for d in dices if d is not None]
    if any(d < REPERE_DICE_FAIBLE for d in definis):
        return True
    return False if len(definis) == len(dices) else None


class DefautCalcul(RuntimeError):
    """Incohérence de calcul détectée par construction (valeur non finie de MONAI, morceaux différents de 05i) :
    défaut du code ou des données, jamais une panne extérieure. En phase C : statut « arret_defaut » (D43)."""


class ArretClassement(RuntimeError):
    """Défaut du code ou des données rencontré en classant une alerte : tout motif AUTRE que les deux motifs de
    DEFINITION_NON_CLASSABLE (règle inconnue, structure non mesurée pour une raison technique, Dice non défini
    hors D2/D3, densités de la vérité non mesurées alors qu'elles étaient exigées...). NB : une vérité que le
    module figé refuse de mesurer (« mesurable » faux, géométrie) relève du motif (ii), pas d'un arrêt.
    Jamais un classement silencieux : les scripts 06b et 06d arrêtent avec CODE_ARRET_CLASSEMENT (23) et ce
    message."""
    code = CODE_ARRET_CLASSEMENT

    def __init__(self, cas, alerte, motif):
        super().__init__(f"classement impossible ({cas}, {alerte.get('regle')} « {alerte.get('objet')} ») : {motif}. "
                         "Ce n'est PAS un « non classable » (définition D43) : défaut du code ou des données.")


def _dice_requis(met, e, cas, alerte):
    """Dice de la structure e pour classer une alerte : (valeur, None) si défini ; (None, raison) si non défini
    parce que la vérité est vide (D2, D3 : motif (i)) ; ArretClassement pour tout autre motif."""
    if e not in met:
        raise ArretClassement(cas, alerte, f"structure {e} non mesurée pour ce modèle")
    x = met[e]
    if x["dice"] is not None:
        return x["dice"], None
    if x.get("cas_degenere") in ("D2", "D3"):
        return None, f"{NC_VERITE_VIDE} : vérité de {NOMS[e]} vide ({x['cas_degenere']}, § 6), Dice et écart de volume non définis"
    raise ArretClassement(cas, alerte, f"Dice de {NOMS[e]} non défini sans cas dégénéré D2 ou D3")


def classer_une(alerte, met, verite_densites, egares, seuils, cas="?"):
    """(classement, justification) d'une alerte (§ 7.3). met = {k: métriques} du cas ; verite_densites = mesures
    du module figé sur la VÉRITÉ (R6, R7) ; egares = {k: morceaux_egares(...)}.

    Classement : « vraie », « fausse » ou « non classable ». « Non classable » suit EXACTEMENT
    DEFINITION_NON_CLASSABLE (D43) :
      (i)  la vérité d'une structure dont la règle compare le Dice ou l'écart de volume est vide (D2 ou D3) :
           R2, R3, R4, R5, R8. Conditions reliées par « ou » (R3 à R5 : deux Dice ; R8 : morceau égaré ou
           Dice) : l'alerte est VRAIE dès qu'une condition DÉFINIE est vraie (ex. R3 avec vérité de l'AP vide
           et Dice de l'aorte 0,50 : vraie) ; NON CLASSABLE seulement si aucune condition définie n'est vraie et
           qu'au moins une n'est pas définie ; FAUSSE si toutes sont définies et fausses. R2 : non classable dès
           que la vérité est vide (Dice et écart de volume tous deux non définis).
           R1 compare le VOLUME de la vérité, défini même vide (0 mL) : jamais non classable ;
      (ii) R6 ou R7 : le module figé, appliqué à la vérité, renvoie None pour la densité exigée (structure
           absente ou trop petite), ou déclare la vérité non mesurable (« mesurable » faux : géométrie).
    La justification d'une alerte non classable commence par « (i) » ou « (ii) » : c'est sa raison.
    Tout autre motif lève ArretClassement (arrêt, code 23), jamais un classement silencieux."""
    r, k = alerte["regle"], _etiquette_objet(alerte)
    if r == "R0":
        return "vraie", "R0 : toujours vraie (masque techniquement inutilisable)"
    if r == "R1":
        if k not in met:
            raise ArretClassement(cas, alerte, f"structure {k} non mesurée pour ce modèle")
        v = met[k]["vol_ref_ml"]
        if v is None:
            raise ArretClassement(cas, alerte, f"volume de la vérité de {NOMS[k]} absent")
        return ("vraie" if v > cp.VOLUME_MIN_R1_ML else "fausse"), f"vérité de {NOMS[k]} = {v:.4f} mL (vraie si > {cp.VOLUME_MIN_R1_ML} mL)"
    if r == "R2":
        dk, nc = _dice_requis(met, k, cas, alerte)
        if nc:
            return "non classable", nc
        pct = met[k]["ecart_pct"]
        if pct is None:
            raise ArretClassement(cas, alerte, f"écart de volume de {NOMS[k]} non défini alors que la vérité n'est pas vide")
        if dk < REPERE_DICE_FAIBLE or abs(pct) > REPERE_ECART_VOLUME_PCT:
            return "vraie", f"Dice {dk} ; écart de volume {pct} % (vraie si Dice < 0,85 ou |écart| > 30 %)"
        return "fausse", f"Dice {dk} ≥ 0,85 et |écart| {abs(pct)} % ≤ 30 %"
    if r in STRUCTURES_REGLE:
        e1, e2 = STRUCTURES_REGLE[r]
        (d1, nc1), (d2, nc2) = _dice_requis(met, e1, cas, alerte), _dice_requis(met, e2, cas, alerte)
        just = f"Dice {NOMS[e1]} {d1} ; Dice {NOMS[e2]} {d2} (vraie si l'un < 0,85)"
        faible = _dice_faible([d1, d2])          # True dès qu'un Dice DÉFINI < 0,85 ; None si indécidable
        if faible:
            nc = " ; ".join(x for x in (nc1, nc2) if x)
            return "vraie", just + (f" ; condition définie vraie malgré : {nc}" if nc else "")
        if faible is None:                       # aucun Dice défini < 0,85 et au moins une vérité vide
            return "non classable", " ; ".join(x for x in (nc1, nc2) if x)
        return "fausse", just
    if r == "R8":
        if k not in egares:
            raise ArretClassement(cas, alerte, f"morceaux égarés de {NOMS[k]} non recherchés")
        dk, nc = _dice_requis(met, k, cas, alerte)
        e = egares[k]
        if e["un_egare"] or (dk is not None and dk < REPERE_DICE_FAIBLE):
            return "vraie", (f"{len(e['egares'])} morceau(x) égaré(s) de plus de 0,5 mL ; Dice {dk}"
                             + (f" ; condition définie vraie malgré : {nc}" if nc else ""))
        if nc:                                   # aucun morceau égaré et Dice non défini (vérité vide)
            return "non classable", nc
        return "fausse", f"aucun morceau égaré ; Dice {dk} ≥ 0,85"
    if r in ("R6", "R7"):
        if verite_densites is None:
            raise ArretClassement(cas, alerte, "densités de la vérité non mesurées alors que R6 ou R7 les exige")
        if verite_densites.get("mesurable") is False:     # module figé : vérité non mesurable (géométrie) → (ii)
            return "non classable", (f"{NC_DENSITE} : le module figé n'a pas pu mesurer la vérité "
                                     f"(géométrie : {verite_densites.get('r0')})")
        if r == "R6":
            bloc, cle, s = "ecarts_hu", "aorte_moins_AP", seuils["R6"]["ecart_aorte_moins_AP_min_hu"]
            libelle = "vérité : aorte − AP = {} HU (vraie si < {} HU)"
        else:
            bloc, cle, s = "hu", "VG", seuils["R7"]["hu_cavite_VG_min"]
            libelle = "vérité : densité de la cavité VG = {} HU (vraie si < {} HU)"
        if cle not in (verite_densites.get(bloc) or {}):
            raise ArretClassement(cas, alerte, f"mesure « {bloc}.{cle} » absente des mesures de la vérité")
        x = verite_densites[bloc][cle]
        if x is None:
            return "non classable", (f"{NC_DENSITE} : le module figé renvoie None pour « {cle} » (structure absente ou "
                                     f"trop petite) ; " + libelle.format("—", s))
        return ("vraie" if x < s else "fausse"), libelle.format(x, s)
    raise ArretClassement(cas, alerte, f"règle inconnue {r!r}")


def classer_alertes(comparaison, modele, statut_alertes, controles, metriques, verites_densites, egares, seuils):
    """Une ligne par alerte : cas, règle, structure ou rapport, mesure, seuil, catégorie (§ 7.2), classement
    vrai / faux / non classable (§ 7.3, D43) et, pour une non classable, sa raison. ArretClassement si une
    alerte ne peut être classée pour un motif autre que ceux de DEFINITION_NON_CLASSABLE."""
    lignes = []
    for c in sorted(controles):
        for a in controles[c]["regles_declenchees"]:
            cl, just = classer_une(a, metriques[c], verites_densites.get(c), egares.get(c, {}), seuils, c)
            lignes.append({"comparaison": comparaison, "modele": modele, "statut_alertes": statut_alertes, "cas": c,
                           "regle": a["regle"], "categorie": a["categorie"], "objet": a["objet"],
                           "mesure": a.get("mesure"), "seuil": a.get("seuil"), "classement": cl, "justification": just,
                           "raison_non_classable": just if cl == "non classable" else None})
    return lignes


def echecs_silencieux(par_cas, controles, etiquettes):
    """Couples (cas, structure) avec Dice < 0,50 (non arrondi) SANS AUCUNE alerte sur ce cas (§ 7.3, C2)."""
    return [{"cas": c, "etiquette": k, "structure": NOMS[k], "dice": par_cas[c][k]["dice"]}
            for c in sorted(par_cas) for k in etiquettes
            if par_cas[c][k]["dice"] is not None and par_cas[c][k]["dice"] < REPERE_ECHEC
            and controles[c]["statut"] == "OK" and not controles[c]["regles_declenchees"]]


def estimation_fausses_alertes(alertes, cas):
    """« x sur N » : cas avec au moins une alerte « erreur probable » classée fausse (D36) ; hors domaine à part.
    Compte des vraies alertes par catégorie. Les alertes non classables (D43) n'entrent NI dans x NI dans le
    compte des vraies : elles sont listées à part, avec leur raison."""
    ep = sorted({a["cas"] for a in alertes if a["categorie"] == "erreur_probable" and a["classement"] == "fausse"})
    hd = sorted({a["cas"] for a in alertes if a["categorie"] == "hors_domaine" and a["classement"] == "fausse"})
    vraies = {cat: sum(a["categorie"] == cat and a["classement"] == "vraie" for a in alertes)
              for cat in ("erreur_probable", "hors_domaine")}
    nc = [{"cas": a["cas"], "regle": a["regle"], "categorie": a["categorie"], "objet": a["objet"],
           "raison": a["raison_non_classable"]} for a in alertes if a["classement"] == "non classable"]
    return {"erreur_probable": {"cas": ep, "x": len(ep), "sur": len(cas), "texte": f"{len(ep)} sur {len(cas)}"},
            "hors_domaine": {"cas": hd, "x": len(hd), "sur": len(cas), "texte": f"{len(hd)} sur {len(cas)}",
                             "alertes_fausses": sum(a["categorie"] == "hors_domaine" and a["classement"] == "fausse" for a in alertes)},
            "vraies_alertes": vraies, "alertes_non_classables": nc,
            "definition_non_classable": DEFINITION_NON_CLASSABLE}


# =================================================================================================
# Verdict (§ 8.4) et règle « meilleure méthode » (P6)
# =================================================================================================
def verdict(agregats_a, echecs_a):
    """C1 : médiane du Dice (non arrondie, n cas pris en compte) ≥ 0,85 (1 à 6) et ≥ 0,80 (AP) ; C2 : aucun
    échec silencieux. Une médiane non calculable (n = 0) ne remplit pas C1."""
    c1 = []
    for k, seuil in SEUILS_C1.items():
        s = agregats_a[k]["dice"]
        ok = s["mediane"] is not None and s["mediane"] >= seuil
        c1.append({"etiquette": k, "structure": NOMS[k], "mediane": s["mediane"], "n": s["n"], "seuil": seuil, "rempli": ok})
    c1_ok, c2_ok = all(x["rempli"] for x in c1), not echecs_a
    raisons = [f"C1 non rempli pour {x['structure']} (médiane {x['mediane']} < {x['seuil']}, n = {x['n']})"
               if x["mediane"] is not None else f"C1 non rempli pour {x['structure']} (médiane non calculable, n = 0)"
               for x in c1 if not x["rempli"]]
    raisons += [f"C2 non rempli : échec silencieux sur {e['cas']} {e['structure']} (Dice {e['dice']})" for e in echecs_a]
    return {"C1": {"rempli": c1_ok, "structures": c1}, "C2": {"rempli": c2_ok, "echecs_silencieux": echecs_a},
            "verdict": "acceptable" if c1_ok and c2_ok else "non acceptable", "raisons": raisons}


def regle_p6(par_cas_x, par_cas_y, etiquettes, nom_x="N0", nom_y="S0"):
    """P6 : X « meilleure sur k » si médiane(X) − médiane(Y) > 0,02 ET X fait strictement mieux sur au moins 4
    cas ; sinon « pas de différence démontrée ». Description, pas une preuve.

    Précisions (D43) :
      - la comparaison porte sur les valeurs FLOTTANTES CALCULÉES (Dice non arrondis, médianes calculées par
        statistics.median), avec une inégalité STRICTE (écart > 0,02) et SANS tolérance numérique : un écart
        flottant égal à 0,02 donne « pas de différence démontrée ». Conséquence assumée : des valeurs décimales
        comme 0,87 et 0,85 ont en flottant un écart de 0,020000000000000018, donc > 0,02 ;
      - la règle n'est appliquée qu'avec 5 cas utilisables : si n < 5 pour l'un des deux modèles (cas D2, D3
        exclus) ou si moins de 5 cas sont comparés (Dice défini des deux côtés), conclusion « pas de différence
        démontrée » SANS appliquer la règle (champ regle_appliquee = False et raison)."""
    sortie = {}
    for k in etiquettes:
        mx = _stats([par_cas_x[c][k]["dice"] for c in par_cas_x if par_cas_x[c][k]["cas_degenere"] not in ("D2", "D3")], "minimum")
        my = _stats([par_cas_y[c][k]["dice"] for c in par_cas_y if par_cas_y[c][k]["cas_degenere"] not in ("D2", "D3")], "minimum")
        paires = [(par_cas_x[c][k]["dice"], par_cas_y[c][k]["dice"]) for c in sorted(set(par_cas_x) & set(par_cas_y))
                  if par_cas_x[c][k]["dice"] is not None and par_cas_y[c][k]["dice"] is not None]
        x_mieux, y_mieux = sum(a > b for a, b in paires), sum(b > a for a, b in paires)
        conclusion, appliquee, raison = P6_PAS_DE_DIFFERENCE, False, ""
        if min(mx["n"], my["n"], len(paires)) < P6_CAS_REQUIS:
            raison = (f"moins de {P6_CAS_REQUIS} cas utilisables (n {nom_x} = {mx['n']}, n {nom_y} = {my['n']}, "
                      f"cas comparés = {len(paires)}) : règle non appliquée")
        else:
            appliquee = True
            if mx["mediane"] - my["mediane"] > P6_ECART_MEDIANE and x_mieux >= P6_CAS_MIN:
                conclusion = f"{nom_x} meilleure sur cette structure (règle descriptive P6, pas une preuve)"
            elif my["mediane"] - mx["mediane"] > P6_ECART_MEDIANE and y_mieux >= P6_CAS_MIN:
                conclusion = f"{nom_y} meilleure sur cette structure (règle descriptive P6, pas une preuve)"
        sortie[k] = {"mediane_" + nom_x: mx["mediane"], "mediane_" + nom_y: my["mediane"], "n_" + nom_x: mx["n"],
                     "n_" + nom_y: my["n"], "cas_compares": len(paires), f"cas_{nom_x}_mieux": x_mieux,
                     f"cas_{nom_y}_mieux": y_mieux, "regle_appliquee": appliquee, "raison": raison,
                     "conclusion": conclusion}
    return sortie


# =================================================================================================
# Les trois comparaisons d'un coup
# =================================================================================================
def evaluer(entrees):
    """entrees = {
        "cas": ["1004", ...],
        "metriques": {"A": {cas: {k: ...}}, "N0": ..., "S0": ..., "T": ...}    (modèles absents permis en phase A),
        "controles": {M: {cas: resultat de appliquer (R0 à R8, NON restreint)}},
        "egares": {M: {cas: {k: morceaux_egares}}}, "verites_densites": {cas: mesures de la vérité ou absent},
        "seuils": seuils figés, "remarques_cas": {cas: texte}}.
    Renvoie le dictionnaire complet des résultats (valeurs NON arrondies)."""
    met, ctl, cas = entrees["metriques"], entrees["controles"], list(entrees["cas"])
    seuils, vd, eg = entrees["seuils"], entrees.get("verites_densites", {}), entrees.get("egares", {})
    res = {"cas": cas, "remarques_cas": entrees.get("remarques_cas", {}), "comparaisons": {}}
    plan = (("a", "A", TOUTES_ETIQUETTES, "officielles", False), ("b", "N0", TOUTES_ETIQUETTES, "informatives", False),
            ("b", "S0", TOUTES_ETIQUETTES, "informatives", False), ("c", "T", ETIQUETTES_C, "à part, informatives", True),
            ("c", "A", ETIQUETTES_C, "officielles (rappel de (a))", False))
    for comp, m, etiq, statut, restreint in plan:
        if m not in met or (comp == "c" and "T" not in met):
            continue
        cas_m = [c for c in cas if c in met[m] and (comp != "c" or c in met["T"])]   # (c) : les cas de T
        pc = {c: {k: met[m][c][k] for k in etiq} for c in cas_m}
        controles = {c: (restreindre_T(ctl[m][c]) if restreint else ctl[m][c]) for c in cas_m}
        bloc = {"modele": m, "etiquettes": list(etiq), "statut_alertes": statut, "cas": cas_m, "par_cas": pc,
                "dice_moyen_par_cas": {c: dict(zip(("valeur", "n"), dice_moyen(pc[c], etiq))) for c in cas_m},
                "agregats": agreger(pc, etiq), "controle": controles}
        if comp == "c" and m == "A":                                 # rappel de A sur 1 à 5, sans recompter ses alertes
            res["comparaisons"].setdefault(comp, {})[m] = bloc
            continue
        bloc["alertes"] = classer_alertes(comp, m, statut, controles, pc, vd, eg.get(m, {}), seuils)
        bloc["echecs_silencieux"] = echecs_silencieux(pc, controles, etiq)
        bloc["cas_en_alerte"] = [c for c in cas_m if controles[c]["statut"] == "ALERTE"]
        if comp == "a":
            sans = [c for c in cas_m if c not in bloc["cas_en_alerte"]]
            bloc["agregats_sans_cas_en_alerte"] = {"cas": sans, "agregats": agreger(pc, etiq, set(sans))}
            bloc["estimation_fausses_alertes"] = estimation_fausses_alertes(bloc["alertes"], cas_m)
        res["comparaisons"].setdefault(comp, {})[m] = bloc
    if "A" in res["comparaisons"].get("a", {}):
        a = res["comparaisons"]["a"]["A"]
        res["verdict"] = verdict(a["agregats"], a["echecs_silencieux"])
    b = res["comparaisons"].get("b", {})
    if "N0" in b and "S0" in b:
        res["p6"] = regle_p6(b["N0"]["par_cas"], b["S0"]["par_cas"], TOUTES_ETIQUETTES)
    return res


# =================================================================================================
# Tableaux (§ 8) : CSV et JSON non arrondis ; rapport Markdown arrondi (§ 8.1, affichage seulement)
# =================================================================================================
def _r(x, n):
    return "—" if x is None else f"{x:.{n}f}"


def _pct(x):
    return "—" if x is None else f"{x:+.0f} %"


def _signe(x, n):
    return "—" if x is None else f"{x:+.{n}f}"


def _hd(m):
    return _r(m["hd95_mm"], 1) + (" (plafond)" if m["hd95_plafond"] else "")


def _sans_nan(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {str(k): _sans_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sans_nan(v) for v in o]
    return o


def _csv(lignes, colonnes):
    f = io.StringIO()
    w = csv.DictWriter(f, fieldnames=colonnes, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for l in lignes:
        w.writerow({k: ("" if l.get(k) is None else (json.dumps(l[k], ensure_ascii=False) if isinstance(l[k], (dict, list)) else l[k]))
                    for k in colonnes})
    return f.getvalue()


def _blocs(res):
    for comp in ("a", "b", "c"):
        for m, bloc in res["comparaisons"].get(comp, {}).items():
            yield comp, m, bloc


def tableaux_csv(res):
    """{nom de fichier: texte CSV} : par cas et structure, Dice moyen par cas, agrégats, alertes, échecs, P6."""
    par_cas, moy, ag, al, ech, p6 = [], [], [], [], [], []
    for comp, m, b in _blocs(res):
        mention = {"mention": b["mention"]} if b.get("mention") else {}
        for c in b["cas"]:
            for k in b["etiquettes"]:
                x = b["par_cas"][c][k]
                par_cas.append({"comparaison": comp, "modele": m, "cas": c, "remarque": res["remarques_cas"].get(c, ""),
                                "etiquette": k, "structure": NOMS[k], **x, "statut_controle": b["controle"][c]["statut"], **mention})
            moy.append({"comparaison": comp, "modele": m, "cas": c, "etiquettes": "1 à 7" if len(b["etiquettes"]) == 7 else "1 à 5",
                        "dice_moyen": b["dice_moyen_par_cas"][c]["valeur"], "n": b["dice_moyen_par_cas"][c]["n"], **mention})
        groupes = [("tous les cas", b["agregats"])]
        if "agregats_sans_cas_en_alerte" in b:
            groupes.append(("complément : sans les cas en alerte", b["agregats_sans_cas_en_alerte"]["agregats"]))
        for quoi, agr in groupes:
            for k in b["etiquettes"]:
                for met, s in agr[k].items():
                    if isinstance(s, dict):
                        ag.append({"comparaison": comp, "modele": m, "ensemble": quoi, "etiquette": k, "structure": NOMS[k],
                                   "metrique": met, **s, **mention})
        al += [{**a, **mention} for a in b.get("alertes", [])]
        ech += [{"comparaison": comp, "modele": m, **e, **mention} for e in b.get("echecs_silencieux", [])]
    for k, x in (res.get("p6") or {}).items():
        p6.append({"etiquette": k, "structure": NOMS[k], **x})
    col_pc = ["comparaison", "modele", "cas", "remarque", "etiquette", "structure", "dice", "hd95_mm", "hd95_plafond",
              "vol_pred_ml", "vol_ref_ml", "ecart_ml", "ecart_pct", "nsd_2mm", "cas_degenere", "anomalie", "n_pred",
              "n_ref", "statut_controle"]
    col_ag = ["comparaison", "modele", "ensemble", "etiquette", "structure", "metrique", "n", "mediane", "moyenne",
              "minimum", "maximum", "mediane_signee", "moyenne_signee", "mediane_absolue", "pire"]
    col_al = ["comparaison", "modele", "statut_alertes", "cas", "regle", "categorie", "objet", "mesure", "seuil",
              "classement", "justification", "raison_non_classable"]
    col_p6 = ["etiquette", "structure", "mediane_N0", "mediane_S0", "n_N0", "n_S0", "cas_compares", "cas_N0_mieux",
              "cas_S0_mieux", "regle_appliquee", "raison", "conclusion"]
    # Colonne « mention » seulement si un bloc en porte une (phase A : T marqué « vérification du code seulement »).
    mc = ["mention"] if any(b.get("mention") for _, _, b in _blocs(res)) else []
    return {"metriques_par_cas.csv": _csv(par_cas, col_pc + mc),
            "dice_moyen_par_cas.csv": _csv(moy, ["comparaison", "modele", "cas", "etiquettes", "dice_moyen", "n"] + mc),
            "agregats.csv": _csv(ag, col_ag + mc), "alertes.csv": _csv(al, col_al + mc),
            "echecs_silencieux.csv": _csv(ech, ["comparaison", "modele", "cas", "etiquette", "structure", "dice"] + mc),
            "comparaison_b_p6.csv": _csv(p6, col_p6)}


def _md_par_cas(res, b):
    l = ["| cas | structure | Dice | HD95 (mm) | V prédit (mL) | V vérité (mL) | écart (mL) | écart (%) | NSD 2 mm | remarque |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c in b["cas"]:
        for k in b["etiquettes"]:
            x = b["par_cas"][c][k]
            prudence = "prudence (D23)" if b["modele"] == "T" and k in (4, 5) else ""
            rem = "; ".join(t for t in (x["cas_degenere"] or "", prudence, res["remarques_cas"].get(c, "")) if t)
            l.append(f"| {c} | {NOMS[k]} | {_r(x['dice'], 3)} | {_hd(x)} | {_r(x['vol_pred_ml'], 1)} | {_r(x['vol_ref_ml'], 1)} "
                     f"| {_signe(x['ecart_ml'], 1)} | {_pct(x['ecart_pct'])} | {_r(x['nsd_2mm'], 3)} | {rem} |")
        d = b["dice_moyen_par_cas"][c]
        l.append(f"| {c} | **Dice moyen ({len(b['etiquettes'])} structures)** | **{_r(d['valeur'], 3)}** (n = {d['n']}) | | | | | | | |")
    return l


def _md_agregats(b, agr):
    l = ["| structure | Dice médiane / moyenne / min (n) | HD95 médiane / moyenne / max (n) | écart mL : méd. / moy. signées, méd. abs., pire (n) "
         "| écart % : méd. / moy. signées, méd. abs., pire (n) | NSD médiane / moyenne / min (n) |", "|---|---|---|---|---|---|"]
    for k in b["etiquettes"]:
        a = agr[k]
        d, h, v, p, n = a["dice"], a["hd95_mm"], a["ecart_ml"], a["ecart_pct"], a["nsd_2mm"]
        nom = NOMS[k] + (" (prudence, D23)" if b["modele"] == "T" and k in (4, 5) else "")
        l.append(f"| {nom} | {_r(d['mediane'], 3)} / {_r(d['moyenne'], 3)} / {_r(d['minimum'], 3)} ({d['n']}) "
                 f"| {_r(h['mediane'], 1)} / {_r(h['moyenne'], 1)} / {_r(h['maximum'], 1)} ({h['n']})"
                 + (f", dont {a['hd95_plafonds']} plafond(s)" if a["hd95_plafonds"] else "")
                 + f" | {_signe(v['mediane_signee'], 1)} / {_signe(v['moyenne_signee'], 1)}, {_r(v['mediane_absolue'], 1)}, "
                 f"{_signe(v['pire'], 1)} ({v['n']}) | {_pct(p['mediane_signee'])} / {_pct(p['moyenne_signee'])}, "
                 f"{_r(p['mediane_absolue'], 0)} %, {_pct(p['pire'])} ({p['n']}) | {_r(n['mediane'], 3)} / {_r(n['moyenne'], 3)} / "
                 f"{_r(n['minimum'], 3)} ({n['n']}) |" + (f" exclus : {', '.join(a['exclus'])}" if a["exclus"] else ""))
    return l


def _md_alertes(b):
    if not b.get("alertes"):
        return ["Aucune alerte."]
    l = []
    for cat, titre in (("erreur_probable", "Alertes « erreur probable »"), ("hors_domaine", "Alertes « hors domaine » (limite du modèle, pas une erreur de segmentation)")):
        lignes = [a for a in b["alertes"] if a["categorie"] == cat and a["classement"] != "non classable"]
        l += [f"**{titre}**, classées (vraies ou fausses) : {len(lignes)}", ""]
        if lignes:
            l += ["| cas | règle | structure ou rapport | mesure | seuil | classement | justification |", "|---|---|---|---|---|---|---|"]
            l += [f"| {a['cas']} | {a['regle']} | {a['objet']} | {json.dumps(a['mesure'], ensure_ascii=False)} | "
                  f"{json.dumps(a['seuil'], ensure_ascii=False)} | {a['classement']} | {a['justification']} |" for a in lignes]
        l.append("")
    nc = [a for a in b["alertes"] if a["classement"] == "non classable"]
    l += [f"**Alertes non classables** (rubrique à part, D43) : {len(nc)}", "", f"Définition : {DEFINITION_NON_CLASSABLE}", ""]
    if nc:
        l += ["| cas | règle | catégorie | structure ou rapport | raison |", "|---|---|---|---|---|"]
        l += [f"| {a['cas']} | {a['regle']} | {a['categorie']} | {a['objet']} | {a['raison_non_classable']} |" for a in nc]
        l.append("")
    return l


def rapport_markdown(res, titre, avertissement=""):
    """Rapport lisible : arrondis du § 8.1 (Dice 3 décimales, HD95 0,1 mm, volumes 0,1 mL, % à l'unité ; NSD 3
    décimales). Le verdict et la règle P6 ont été décidés sur les valeurs NON arrondies."""
    l = [f"# {titre}", ""]
    if avertissement:
        l += [f"> {avertissement}", ""]
    l += [f"Cas : {', '.join(res['cas'])}." + "".join(f" {c} : {t}." for c, t in res["remarques_cas"].items()), "",
          f"Définition de la HD95 : {DEFINITION_HD95} « (plafond) » : structure oubliée (D1), HD95 = diagonale de l'image.", "",
          "Arrondis à l'affichage seulement (§ 8.1) ; une différence de Dice de 0,005 n'est pas une différence. "
          "Pas de test statistique ni d'intervalle de confiance (§ 8.3).", ""]
    if "verdict" in res:
        v = res["verdict"]
        l += ["## Verdict (pipeline A seulement, § 8.4)", "", f"**{v['verdict'].upper()}**", "",
              "| structure | médiane du Dice (non arrondie) | n | seuil | C1 |", "|---|---|---|---|---|"]
        l += [f"| {x['structure']} | {x['mediane'] if x['mediane'] is not None else '—'} | {x['n']} | {x['seuil']} | {'rempli' if x['rempli'] else 'NON rempli'} |"
              for x in v["C1"]["structures"]]
        l += ["", f"C2 (aucun échec silencieux) : {'rempli' if v['C2']['rempli'] else 'NON rempli'}."]
        l += [f"- {x}" for x in v["raisons"]] + [""]
        l += ["Limites (§ 8.4, § 12) : seuils fixés après la validation croisée (pas totalement indépendants de nos "
              "résultats) ; avec 5 patients, verdict indicatif ; Zhuang et al. 2019 non directement comparable.", ""]
    titres = {"a": "(a) Pipeline principal : A post-traité, 7 structures", "b": "(b) Méthodes à armes égales : N0 et S0 bruts, 7 structures",
              "c": "(c) Référence externe : TotalSegmentator (T), structures 1 à 5, en regard de A"}
    for comp in ("a", "b", "c"):
        blocs = res["comparaisons"].get(comp, {})
        if not blocs:
            continue
        l += [f"## {titres[comp]}", ""]
        if comp == "c":
            l += [f"> {PRUDENCE_C}", ""]
        for m, b in blocs.items():
            l += [f"### Modèle {m}" + (" (rappel des valeurs de (a) sur les structures 1 à 5)" if comp == "c" and m == "A" else "")
                  + (f" — {b['mention']}" if b.get("mention") else ""), ""]
            if b.get("mention"):
                l += [f"> **{b['mention']}**", ""]
            l += _md_par_cas(res, b) + ["", "Agrégats (tous les cas, alertes comprises) :", ""] + _md_agregats(b, b["agregats"]) + [""]
            if "agregats_sans_cas_en_alerte" in b:
                s = b["agregats_sans_cas_en_alerte"]
                l += [f"Complément, **pas un résultat principal** (§ 8.2) : agrégats SANS les cas en alerte "
                      f"({', '.join(b['cas_en_alerte']) or 'aucun'}) :", ""] + _md_agregats(b, s["agregats"]) + [""]
            if comp == "c" and m == "A":
                continue
            l += [f"Contrôle de plausibilité (alertes {b['statut_alertes']}) : {len(b['cas_en_alerte'])} cas en alerte "
                  f"sur {len(b['cas'])} ({', '.join(b['cas_en_alerte']) or 'aucun'}).", ""] + _md_alertes(b)
            if "estimation_fausses_alertes" in b:
                e = b["estimation_fausses_alertes"]
                l += [f"**Estimation officielle des fausses alertes (D36)** : {e['erreur_probable']['texte']} cas avec au moins "
                      f"une alerte « erreur probable » classée fausse ({', '.join(e['erreur_probable']['cas']) or 'aucun'}). "
                      f"Hors domaine classées fausses, à part : {e['hors_domaine']['texte']} cas. Vraies alertes : "
                      f"{e['vraies_alertes']['erreur_probable']} « erreur probable », {e['vraies_alertes']['hors_domaine']} "
                      f"hors domaine. Alertes non classables : {len(e['alertes_non_classables'])}, rapportées à part ; elles "
                      "n'entrent ni dans l'estimation ni dans le compte des vraies alertes (D43).", ""]
            ech = b["echecs_silencieux"]
            l += [f"Échecs silencieux (Dice < 0,50 sans aucune alerte sur le cas) : "
                  + ("; ".join(f"{e['cas']} {e['structure']} ({_r(e['dice'], 3)})" for e in ech) if ech else "aucun") + ".", ""]
        if comp == "b" and res.get("p6"):
            l += ["### Règle descriptive P6 (pas une preuve)", "",
                  "Écart des médianes > 0,02 (inégalité stricte, valeurs flottantes non arrondies, sans tolérance) et "
                  "mieux sur au moins 4 cas ; règle appliquée seulement avec 5 cas utilisables pour chaque modèle "
                  "(sinon « pas de différence démontrée », D43).", "",
                  "| structure | médiane N0 | médiane S0 | cas où N0 fait mieux | cas où S0 fait mieux | conclusion |",
                  "|---|---|---|---|---|---|"]
            l += [f"| {NOMS[k]} | {_r(x['mediane_N0'], 3)} | {_r(x['mediane_S0'], 3)} | {x['cas_N0_mieux']} sur {x['cas_compares']} | "
                  f"{x['cas_S0_mieux']} sur {x['cas_compares']} | {x['conclusion']}" + (f" ({x['raison']})" if x["raison"] else "") + " |"
                  for k, x in res["p6"].items()]
            l.append("")
    return "\n".join(l) + "\n"


def ecrire_tableaux(dossier, res, titre, avertissement=""):
    """Écrit dans `dossier` (qui doit déjà exister, en général un dossier temporaire) les CSV, le JSON complet et
    le rapport Markdown. Renvoie la liste des noms écrits."""
    fichiers = tableaux_csv(res)
    fichiers["resultats_etape6.json"] = json.dumps(_sans_nan(res), ensure_ascii=False, indent=1) + "\n"
    fichiers["rapport_etape6.md"] = rapport_markdown(res, titre, avertissement)
    for nom, texte in fichiers.items():
        with open(os.path.join(dossier, nom), "w", encoding="utf-8", newline="") as f:
            f.write(texte)
    return sorted(fichiers)


def publier(dossier_final, res, titre, avertissement="", autres_json=None, tmp=None):
    """Écrit TOUTES les sorties en une fois : dossier temporaire (<final>.partiel par défaut), puis renommage
    d'un seul coup. Un dossier .partiel laissé par une panne n'est jamais effacé (il reste comme trace)."""
    tmp = tmp or dossier_final + ".partiel"
    if os.path.exists(dossier_final) or os.path.exists(tmp):
        raise FileExistsError(f"{dossier_final} ou {tmp} existe déjà : rien n'est écrasé")
    os.makedirs(tmp)
    noms = ecrire_tableaux(tmp, res, titre, avertissement)
    for nom, contenu in (autres_json or {}).items():
        with open(os.path.join(tmp, nom), "w", encoding="utf-8", newline="") as f:
            f.write(json.dumps(_sans_nan(contenu), ensure_ascii=False, indent=1) + "\n")
        noms.append(nom)
    os.replace(tmp, dossier_final)
    return sorted(noms)
