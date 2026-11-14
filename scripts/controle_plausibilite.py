"""
Contrôle de plausibilité (décision D28, spécification docs/specification_controle_plausibilite.md v1.0) :
MESURES d'un masque et APPLICATION des règles R0 à R8, avec des seuils donnés.
Module à importer (par 05j, et plus tard par l'étape 6) : il ne lit aucun argument, n'écrit aucun fichier,
ne calibre aucun seuil. Il ne lit que les deux fichiers qu'on lui donne (image CT et masque).

Les mesures reprennent TELLES QUELLES les fonctions des scripts déjà exécutés, pour que les chiffres
concordent avec les sections 18, 21 et 22 du notebook :
  - densités (R4 à R7) : mesurer_cas() et ligne() de 05e, donc intensites() de 05d : médiane des HU au
    « cœur » de la structure (voxels à plus de 3 mm du bord), repli sur la structure entière si le cœur
    est vide ; écarts aorte − VG, AP − VD, aorte − AP calculés comme dans 05e (colonnes « _ref ») ;
  - volumes et rapports (R1 à R3) : mesurer_cas() de 05h (voxels × produit des 3 espacements de l'en-tête
    de l'IMAGE / 1000 ; rapports = rapports des nombres de voxels) ;
  - morceaux (R8) : morceaux() de 05i (connectivité 26, morceaux de plus de 0,5 mL, part du plus gros
    morceau dans le volume total de l'étiquette).
Seule R0 (cohérence technique) est mesurée ici, car aucun script précédent ne la mesurait avec la tolérance
de la spécification (0,01 mm).
Toutes les règles, R8 comprise, s'appliquent indifféremment à une vérité ou à une prédiction : R0 et R1 sont
fixées, R2 à R8 sont calibrées sur la vérité (D34), puis toutes sont appliquées aux prédictions.

Vocabulaire :
  - règle « déclenchée » : la mesure sort de la plage autorisée par le seuil ; le cas passe en ALERTE ;
  - catégorie : « erreur_probable » (le masque est probablement faux) ou « hors_domaine » (R6, R7 :
    l'examen ne ressemble pas à ceux du calibrage, le masque peut être juste) ;
  - règle « non évaluée » : la mesure manque (structure absente, cœur vide...) ; c'est noté, jamais deviné.
"""
import importlib.util, os

ICI = os.path.dirname(os.path.abspath(__file__))


def _charger(nom, fichier):
    spec = importlib.util.spec_from_file_location(nom, os.path.join(ICI, fichier))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dens = _charger("densites05e", "05e_densites_vaisseaux.py")      # densités (charge lui-même 05d)
vol = _charger("volumes05h", "05h_volumes_verite.py")            # volumes et rapports
pp = _charger("postraitement05i", "05i_postraitement_nnunet.py") # comptage des morceaux

# --- Définitions fixes (spécification § 5 ; aucune n'est calibrée) ----------------------------
STRUCTURES = vol.STRUCTURES                    # {1: "myocarde_VG", 2: "OG", ..., 6: "aorte", 7: "AP"}
RAPPORTS = vol.RAPPORTS                        # 5 rapports de R3 : (nom, numérateur, dénominateur)
ETIQUETTES_AUTORISEES = tuple(range(8))        # R0 : fond 0 et étiquettes 1 à 7
TOLERANCE_R0_MM = 0.01                         # R0 : forme et affine comparées à 0,01 mm près (spéc. § 5)
VOLUME_MIN_R1_ML = 1.0                         # R1 : V min,abs = 1 mL (fixé, spéc. § 5)
MORCEAU_ML = pp.MORCEAU_ML                     # R8 : on compte les morceaux de plus de 0,5 mL (définition)
TOUTES = ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8")
CATEGORIE = {r: ("hors_domaine" if r in ("R6", "R7") else "erreur_probable") for r in TOUTES}   # spéc. § 4, Q8
LIBELLE = {
    "R0": "masque techniquement incohérent avec l'image (forme, affine ou étiquettes)",
    "R1": "structure absente (volume ≤ 1 mL)",
    "R2": "volume de la structure hors de la plage observée",
    "R3": "rapport de volumes hors de la plage observée",
    "R4": "densité de l'aorte prédite éloignée de celle de la cavité VG prédite",
    "R5": "densité de l'AP prédite éloignée de celle du VD prédit",
    "R6": "aorte moins dense que l'AP : phase d'injection hors du domaine validé",
    "R7": "cavité VG peu dense : contraste absent ou insuffisant, hors du domaine validé",
    "R8": "masque éclaté : trop de morceaux, ou plus gros morceau trop petit"}


# --- Mesures ------------------------------------------------------------------------------------
def mesurer(cas, chemin_image, chemin_masque):
    """Toutes les mesures du contrôle pour un masque (vérité ou prédiction) posé sur son image.
    Un seul scanner en mémoire. Renvoie un dictionnaire ; une valeur non mesurable vaut None."""
    import nibabel as nib
    import numpy as np
    img, lab = nib.load(chemin_image), nib.load(chemin_masque)       # en-têtes seulement à ce stade
    m = {"cas": cas, "image": chemin_image, "masque": chemin_masque, "mesurable": True,
         "epaisseur_coupe_mm": round(float(img.header.get_zooms()[2]), 4),
         "r0": [], "anomalies": [], "volumes_ml": {}, "voxels": {}, "rapports": {}, "hu": {},
         "hu_repli_region_entiere": "", "ecarts_hu": {}, "morceaux": {}}
    # R0 : même forme, même affine (0,01 mm près), étiquettes entières de 0 à 7.
    if img.shape[:3] != lab.shape[:3]:
        m["r0"].append(f"forme du masque {lab.shape[:3]} ≠ image {img.shape[:3]}")
        m["mesurable"] = False                   # comparaison voxel à voxel impossible : rien d'autre n'est mesuré
        return m
    ecart_affine = float(np.abs(img.affine - lab.affine).max())
    if ecart_affine > TOLERANCE_R0_MM:
        m["r0"].append(f"affine du masque différente de l'image (écart max {ecart_affine:.4f} mm > {TOLERANCE_R0_MM})")
    # Volumes et rapports : fonction de 05h (le masque est passé à la place de la « vérité »).
    l, an = vol.mesurer_cas(cas, {"entete_image": chemin_image, "verite": chemin_masque}, None, None)
    for e, nom in STRUCTURES.items():
        m["voxels"][e], m["volumes_ml"][e] = l[f"vox_{nom}"], l[f"ml_{nom}"]   # None si absente (comme 05h)
    for r, _, _ in RAPPORTS:
        m["rapports"][r] = l[r]
    # Densités : fonctions de 05e (donc intensites() de 05d) ; « _ref » = médiane au cœur, ou repli.
    stats, an2, ep = dens.mesurer_cas({"image": chemin_image, "verite": chemin_masque})
    d = dens.ligne(cas, stats, ep, None, None)
    for nom in dens.STRUCTURES:                                  # aorte, AP, VG, VD
        m["hu"][nom] = d[f"{nom}_ref"]
    m["hu_repli_region_entiere"] = d["ref_repli_region_entiere"]
    m["ecarts_hu"] = {"aorte_moins_VG": d["aorte_moins_VG_ref"], "AP_moins_VD": d["AP_moins_VD_ref"],
                      "aorte_moins_AP": d["aorte_moins_AP_ref"]}
    # Les textes d'anomalie de 05e/05h parlent de « vérité » : ici, c'est le masque lu, quel qu'il soit.
    m["anomalies"] = [a.replace("de la vérité", "du masque") for a in an + an2]
    # Morceaux (R8) et valeurs présentes (R0) : un seul chargement du masque.
    masque = np.asanyarray(lab.dataobj)
    valeurs = np.unique(masque)
    hors = [float(v) for v in valeurs if float(v) % 1 or int(v) not in ETIQUETTES_AUTORISEES]
    if hors:
        m["r0"].append(f"valeurs hors des étiquettes 0 à 7 : {hors[:10]}")
    vox_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000   # en-tête de l'IMAGE (spéc. § 3)
    for e in STRUCTURES:
        r = pp.morceaux(masque == e, vox_ml)
        m["morceaux"][e] = {"n": r["n"], "n_gros": r["n_gros"],
                            "part_plus_gros": None if r["part_plus_gros"] is None else round(r["part_plus_gros"], 4)}
    del masque
    return m


# --- Application des règles ----------------------------------------------------------------------
def _alerte(regle, objet, mesure, seuil):
    return {"regle": regle, "categorie": CATEGORIE[regle], "libelle": LIBELLE[regle], "objet": objet,
            "mesure": mesure, "seuil": seuil}


def appliquer(m, seuils, regles=TOUTES):
    """Applique les règles demandées aux mesures m, avec les seuils (format de seuils_proposes.json).
    Renvoie {statut, regles_declenchees, regles_verifiees, non_evaluees}. Ne modifie rien."""
    dec, non_eval = [], []
    # Un masque de forme différente de l'image est TOUJOURS signalé par R0, même si R0 n'a pas été demandée :
    # sinon le statut serait ALERTE sans aucune règle listée.
    if ("R0" in regles or not m["mesurable"]) and m["r0"]:
        dec += [_alerte("R0", "masque", p, {"tolerance_affine_mm": TOLERANCE_R0_MM, "etiquettes": "0 à 7"})
                for p in m["r0"]]
    if not m["mesurable"]:
        non_eval += [f"{r} : masque non mesurable (forme différente de l'image)" for r in regles if r != "R0"]
        verifiees = list(regles) if "R0" in regles else ["R0 (forcée : forme différente de l'image)", *regles]
        return {"statut": "ALERTE", "regles_declenchees": dec, "regles_verifiees": verifiees, "non_evaluees": non_eval}
    s = seuils
    for e, nom in STRUCTURES.items():
        v = m["volumes_ml"][e] or 0.0                            # structure absente : volume nul
        if "R1" in regles and v <= s["R1"]["volume_min_ml"]:
            dec.append(_alerte("R1", f"{e} {nom}", {"volume_ml": v}, {"volume_min_ml": s["R1"]["volume_min_ml"]}))
        p = s["R2"]["plages_ml"].get(str(e)) if "R2" in regles else None
        if "R2" in regles and p is None:
            non_eval.append(f"R2 {e} {nom} : pas de seuil")
        elif p and not p["bas"] <= v <= p["haut"]:
            dec.append(_alerte("R2", f"{e} {nom}", {"volume_ml": v}, p))
    if "R3" in regles:
        for r, _, _ in RAPPORTS:
            x, p = m["rapports"][r], s["R3"]["plages"].get(r)
            if x is None or p is None:
                non_eval.append(f"R3 {r} : {'rapport non calculable (structure absente)' if x is None else 'pas de seuil'}")
            elif not p["bas"] <= x <= p["haut"]:
                dec.append(_alerte("R3", r, {"rapport": x}, {"bas": p["bas"], "haut": p["haut"]}))
    ec, hu = m["ecarts_hu"], m["hu"]
    tests = (("R4", "aorte_moins_VG", lambda x, t: abs(x) <= t["ecart_hu_max"]),
             ("R5", "AP_moins_VD", lambda x, t: abs(x) <= t["ecart_hu_max"]),
             ("R6", "aorte_moins_AP", lambda x, t: x >= t["ecart_aorte_moins_AP_min_hu"]))
    for regle, cle, passe in tests:
        if regle not in regles:
            continue
        if ec[cle] is None or s[regle] is None:
            non_eval.append(f"{regle} : {'densité non mesurable (structure absente)' if ec[cle] is None else 'pas de seuil'}")
        elif not passe(ec[cle], s[regle]):
            dec.append(_alerte(regle, cle, {"ecart_hu": ec[cle], **_hu_de(cle, hu)}, s[regle]))
    if "R7" in regles:
        if hu["VG"] is None or s["R7"] is None:
            non_eval.append("R7 : densité de la cavité VG non mesurable" if hu["VG"] is None else "R7 : pas de seuil")
        elif hu["VG"] < s["R7"]["hu_cavite_VG_min"]:
            dec.append(_alerte("R7", "cavité VG", {"hu_VG": hu["VG"]}, s["R7"]))
    if "R8" in regles:
        for e, nom in STRUCTURES.items():
            mo, t = m["morceaux"][e], s["R8"]["par_etiquette"].get(str(e))
            if t is None or mo["n_gros"] == 0:
                non_eval.append(f"R8 {e} {nom} : {'pas de seuil' if t is None else 'aucun morceau de plus de 0,5 mL'}")
            elif mo["n_gros"] > t["morceaux_gros_max"] or mo["part_plus_gros"] < t["part_plus_gros_min"]:
                dec.append(_alerte("R8", f"{e} {nom}", {"morceaux_gros": mo["n_gros"], "morceaux_total": mo["n"],
                                                         "part_plus_gros": mo["part_plus_gros"]}, t))
    return {"statut": "ALERTE" if dec else "OK", "regles_declenchees": dec, "regles_verifiees": list(regles),
            "non_evaluees": non_eval}


def _hu_de(cle, hu):
    """Densités qui composent un écart (pour le rapport)."""
    a, b = cle.split("_moins_")
    return {f"hu_{a}": hu[a], f"hu_{b}": hu[b]}


def resume(resultat):
    """Texte court : « R2 (6 aorte), R4 (aorte_moins_VG) » ; « — » si aucune règle."""
    return ", ".join(f"{a['regle']} ({a['objet']})" for a in resultat["regles_declenchees"]) or "—"
