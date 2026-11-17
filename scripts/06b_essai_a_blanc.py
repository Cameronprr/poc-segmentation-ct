"""
Étape 6, phase A — Essai à blanc de l'outil de mesure sur les CAS D'ENTRAÎNEMENT (protocole § 10.2 A, P9).
Le lot de test n'est JAMAIS lu : ni labelsTs, ni imagesTs, ni aucune prédiction d'un cas de test.

Pourquoi : avant de mesurer la « pièce » (le test, une seule fois), on vérifie que la « règle » mesure juste.
C'est la qualification opérationnelle (OQ) de l'outil : le MÊME code de calcul que la phase C
(scripts/etape6_metriques.py) est appliqué à des cas dont on connaît déjà les résultats (étape 5), et l'on
compare.

Ce que fait le script :
  1. vérifications au démarrage du § 9 : points 1 à 4 (dépôt seulement) et point 5 (MONAI 1.6.1) ;
  2. relève les fichiers (tous des cas d'ENTRAÎNEMENT ; la garde de etape6_commun, phase « A », refuse tout
     chemin de test) : vérité labelsTr et en-tête des images imagesTr ; prédictions hors pli de nnU-Net
     brutes et post-traitées (crossval_results_folds_0_1_2_3_4 et postprocessed/, écrites par 05i) ;
     prédictions de SegResNet pli 0 à 1 250 époques (05m, predictions_validation_finale) ; en option, les 2
     prédictions de TotalSegmentator sur 1002 et 1003 (section 11, 04b) ;
  3. contrôle de plausibilité : pour les prédictions post-traitées, mesures relues dans synthese_05j.json
     (section 23) si ce fichier existe ET si elles portent sur le fichier attendu (même nom et même dossier
     pour le masque post-traité, même nom pour l'image), sinon recalculées (idem pour les mesures de la
     vérité) ; pour les autres, calculées (module figé) ; les
     règles R0 à R8 sont appliquées avec les seuils FIGÉS ; le résultat passe par JSON puis par le lecteur
     lire_controle(), comme les fichiers de la phase B ;
  4. calcule, avec etape6_metriques, les trois comparaisons comme en phase C, sur des cas d'entraînement :
     (a) « A » = nnU-Net post-traité, 15 cas ; (b) « N0 » = nnU-Net brut des 3 cas du pli 0 et « S0 » =
     SegResNet pli 0 sur les mêmes cas ; (c) « T » = TotalSegmentator sur 1002 et 1003 (si présents),
     structures 1 à 5, avec contrôle restreint. Classement des alertes, échecs silencieux et verdict sont
     appliqués COMME ESSAI DU CODE : ce verdict n'a AUCUNE valeur (cas d'entraînement, déjà vus). Les scores
     de TotalSegmentator sur 1002 et 1003 servent UNIQUEMENT à vérifier le code (D43) : ils sont marqués
     MENTION_TS dans toutes les sorties (titre et avertissement du rapport, colonne « mention » des CSV, champ
     « mention » du JSON, qualification_06b.json, affichage) et aucune décision ne peut s'appuyer dessus ;
  5. concordance (critère P9) :
       - Dice de nnU-Net brut et post-traité = summary.json de nnU-Net (et synthese_05i.json s'il existe),
         écart ≤ 0,0005 (l'arrondi à 3 décimales du journal), nombres de voxels identiques ;
       - Dice de S0 = bilan_05m.json, écart ≤ 0,0005 ;
       - volumes de la vérité = synthese_05h.json (section 21), écart ≤ 0,0006 mL (05h arrondit à 0,001 mL) ;
       - HD95 et NSD plausibles : valeurs finies, 0 ≤ HD95 ≤ plafond, 0 ≤ NSD ≤ 1, Dice = 1 ⇒ HD95 = 0 ;
         aucun cas dégénéré (les 7 structures sont présentes dans la vérité des 15 cas) ;
  6. écrit les tableaux (mêmes fichiers qu'en phase C) et qualification_06b.json, puis conclut
     « OUTIL DE MESURE QUALIFIÉ » ou « OUTIL DE MESURE NON QUALIFIÉ » avec la liste des écarts.

Entrées (chemins Drive par défaut, à passer en paramètres dans Colab) :
  --raw            nnUNet_raw (labelsTr, en-têtes de imagesTr)      défaut /content/drive/MyDrive/poc_ct/nnunet_raw
  --results        nnUNet_results (prédictions hors pli, summary.json)   défaut .../poc_ct/nnunet_results
  --auto3dseg      dossier de travail Auto3DSeg (05m)               défaut .../poc_ct/auto3dseg
  --analyses       dossier des analyses (05h_volumes, 05i_postraitement, 05j_controle)   défaut .../poc_ct/analyses
  --totalseg-definitions  prédictions de TotalSegmentator sur 1002 et 1003 (04b)  défaut .../poc_ct/predictions/totalsegmentator_definitions
  --sortie         dossier de la phase A                            défaut .../poc_ct/etape6/phaseA
  --ignorer-verrou comme 05c
Sorties (dans --sortie/essai_<SESSION>/, écrites en une fois : dossier .partiel puis renommage) :
  metriques_par_cas.csv, dice_moyen_par_cas.csv, agregats.csv, alertes.csv, echecs_silencieux.csv,
  comparaison_b_p6.csv, resultats_etape6.json, rapport_etape6.md, qualification_06b.json ;
  et --sortie/sortie_06b_<SESSION>.txt (copie de l'affichage). Chaque lancement écrit un NOUVEAU dossier.
Matériel et durée : pas de GPU (environnement processeur). Environ 15 à 30 min : HD95 et NSD sur
  15 cas × 7 structures × 2 jeux (brut et post-traité) + 3 cas S0 (+ 2 cas × 5 structures pour T), soit
  environ 240 structures à 2 à 5 s chacune (mesuré localement : 2,3 s pour une structure de 125 mL), plus la
  lecture des fichiers sur Drive et 8 contrôles de plausibilité (38 si synthese_05j.json est absent).
Reprise après une coupure : relancer la même cellule ; tout est recalculé (rien n'est réutilisé d'un essai
  coupé ; un dossier .partiel éventuel reste comme trace).
Codes de sortie :
  0 = OUTIL DE MESURE QUALIFIÉ ;   7 = NON QUALIFIÉ (résultats écrits ; envoyer la sortie) ;
  1 = garde-fou (fichier manquant, --sortie dans le dépôt, verrou) ; 6 = summary.json de nnU-Net non
  conforme (arrêt de 05i) ; 10 à 14 = vérification au démarrage (protocole, seuils, quatre valeurs,
  découpage, versions) ; 18 = lecture refusée par la garde (chemin de test) : rien n'est lu ;
  23 = une alerte ne peut pas être classée pour un motif autre que « non classable » (D43) : défaut du code
  ou des données, rien n'est écrit ; envoyer la sortie, ne rien corriger seul.

Vocabulaire :
  - qualification opérationnelle (OQ) : on vérifie qu'un instrument mesure juste sur des étalons connus ;
  - concordance : deux calculs indépendants du même chiffre donnent la même valeur (à l'arrondi près) ;
  - prédiction hors pli : prédiction d'un cas par le modèle qui ne l'a pas vu à l'entraînement.
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import etape6_commun as e6                        # n'exécute rien à l'import
import etape6_metriques as em

PHASE, SCRIPT = "A", "06b"
TOL_DICE, TOL_ML = 5e-4, 6e-4
MENTION_TS = ("VÉRIFICATION DU CODE SEULEMENT — aucune décision ne peut s'appuyer sur ces valeurs (D43) : "
              "scores de TotalSegmentator sur les cas d'entraînement 1002 et 1003")
NOMS_05M = {1: "myo_VG", 2: "OG", 3: "VG", 4: "OD", 5: "VD", 6: "aorte", 7: "AP"}   # clés de bilan_05m.json
DS = "Dataset101_MMWHS"


def lire_arguments():
    p = argparse.ArgumentParser(description="Phase A : essai à blanc de l'outil de mesure (cas d'entraînement).")
    p.add_argument("--raw", default=f"{e6.DRIVE}/nnunet_raw")
    p.add_argument("--results", default=f"{e6.DRIVE}/nnunet_results")
    p.add_argument("--auto3dseg", default=f"{e6.DRIVE}/auto3dseg")
    p.add_argument("--analyses", default=f"{e6.DRIVE}/analyses")
    p.add_argument("--totalseg-definitions", default=f"{e6.DRIVE}/predictions/totalsegmentator_definitions")
    p.add_argument("--sortie", default=e6.DOSSIERS_PHASES[PHASE])
    p.add_argument("--ignorer-verrou", action="store_true")
    return p.parse_args()


def chemins(a, cas):
    """Tous les fichiers lus, par cas d'entraînement (aucun n'est ouvert ici)."""
    cv = os.path.join(em.pp.dossier_modele(a.results), em.pp.NOM_CV)
    pli0 = os.path.join(a.auto3dseg, "entrainement_segresnet_1250", "pli_0")
    ts = a.totalseg_definitions
    f = {c: {"image": os.path.join(a.raw, DS, "imagesTr", f"mmwhs_{c}_0000.nii.gz"),
             "verite": os.path.join(a.raw, DS, "labelsTr", f"mmwhs_{c}.nii.gz"),
             "brut": os.path.join(cv, f"mmwhs_{c}.nii.gz"), "post": os.path.join(cv, "postprocessed", f"mmwhs_{c}.nii.gz"),
             "S0": os.path.join(pli0, "predictions_validation_finale", f"mmwhs_{c}.nii.gz"),
             "T": os.path.join(ts, f"mmwhs_{c}_totalsegmentator.nii.gz")} for c in cas}
    autres = {"summary_brut": os.path.join(cv, "summary.json"), "summary_post": os.path.join(cv, "postprocessed", "summary.json"),
              "bilan_05m": os.path.join(pli0, "bilan_05m.json"), "fold0": os.path.join(em.pp.dossier_modele(a.results), "fold_0", "validation"),
              "05h": os.path.join(a.analyses, "05h_volumes", "synthese_05h.json"),
              "05i": os.path.join(a.analyses, "05i_postraitement", "synthese_05i.json"),
              "05j": os.path.join(a.analyses, "05j_controle", "synthese_05j.json")}
    return f, autres


def controle(cas, image, masque, seuils, mesures=None):
    """Contrôle de plausibilité complet (R0 à R8, seuils figés) ; mesures recalculées si non fournies."""
    if mesures is None:
        e6.verifier_chemin(image, PHASE), e6.verifier_chemin(masque, PHASE)   # le module ouvre les fichiers lui-même
        mesures = em.cp.mesurer(cas, image, masque)
    r = em.cp.appliquer(em.normaliser_mesures(mesures), seuils)
    return {"mesures": mesures, "resultat": r, "resume": em.cp.resume(r)}


def _fin_chemin(x, n=2):
    """Les n derniers éléments d'un chemin (dossier + nom), séparateurs / ou \\ indifférents."""
    return tuple(str(x or "").replace("\\", "/").rstrip("/").split("/")[-n:])


def mesures_05j_conformes(mesures, image, masque):
    """Mesures de synthese_05j.json réutilisables seulement si elles portent sur CE masque (nom et dossier) et
    sur CETTE image (nom) ; sinon None (elles seront recalculées) et la raison."""
    if not mesures:
        return None, None
    if _fin_chemin(mesures.get("masque")) != _fin_chemin(masque):
        return None, f"masque mesuré par 05j {mesures.get('masque')} ≠ attendu {masque}"
    if _fin_chemin(mesures.get("image"), 1) != _fin_chemin(image, 1):
        return None, f"image mesurée par 05j {mesures.get('image')} ≠ attendue {image}"
    return mesures, None


def relire(modele, par_cas, seuils, cas):
    """Passage par JSON puis par le lecteur de la phase C (exerce lire_controle sur de vraies données)."""
    contenu = json.loads(json.dumps({"modele": modele, "seuils_sha256": e6.SHA256_SEUILS, "cas": par_cas}, ensure_ascii=False))
    lu = em.lire_controle(contenu, modele, cas, seuils, e6.SHA256_SEUILS)
    return {c: x["resultat"] for c, x in lu.items()}


def ecart_dice(nous, eux):
    if nous is None or eux is None:
        return None if nous is None and eux is None else float("inf")
    return abs(nous - eux)


def main():
    journal = e6.demarrer_sortie()
    t0 = time.time()
    print(f"Étape 6, phase A (06b) — essai à blanc sur les cas d'ENTRAÎNEMENT — session {e6.SESSION} ; début : {e6.heure()}")
    print(f"Commit du dépôt : {e6.commit_git()}")
    a = lire_arguments()
    sortie = e6.verifier_hors_depot(a.sortie, "--sortie")
    verifs = e6.verifier_demarrage(PHASE)
    versions = e6.verifier_versions(monai=True)
    seuils = e6.lire_json(os.path.join(e6.DEPOT, e6.FICHIER_SEUILS))
    with open(os.path.join(e6.DEPOT, e6.FICHIER_SPLIT), encoding="utf-8") as fic:
        cas = sorted(c.rsplit("_", 1)[-1] for c in json.load(fic)["entrainement_validation_croisee"])
    if len(cas) != 15 or set(cas) & set(e6.CAS_TEST_ATTENDUS):
        e6.arret(1, f"cas d'entraînement inattendus : {cas}")
    f, autres = chemins(a, cas)
    for c in cas:                                 # garde AVANT tout : aucun chemin de test (code 18)
        for x in f[c].values():
            e6.verifier_chemin(x, PHASE)
    bilan = e6.lire_json(autres["bilan_05m"], PHASE) if os.path.isfile(autres["bilan_05m"]) else None
    if bilan is None:
        e6.arret(1, f"{autres['bilan_05m']} absent (section 25, 05m) : S0 ne peut pas être essayé.")
    if not bilan.get("conforme"):
        e6.arret(1, f"{autres['bilan_05m']} : évaluation finale de 05m non conforme ({bilan.get('problemes')}).")
    pli0 = sorted(em._cle_cas(c) for c in bilan["cas"])
    fold0 = sorted(n[len("mmwhs_"):-len(".nii.gz")] for n in os.listdir(e6.verifier_chemin(autres["fold0"], PHASE)) if n.endswith(".nii.gz"))
    ts_cas = [c for c in cas if os.path.isfile(f[c]["T"])]
    manquants = [x for c in cas for k, x in f[c].items() if k not in ("S0", "T") and not os.path.isfile(x)]
    manquants += [f[c]["S0"] for c in pli0 if not os.path.isfile(f[c]["S0"])]
    manquants += [autres[k] for k in ("summary_brut", "summary_post", "05h") if not os.path.isfile(autres[k])]
    if manquants or pli0 != fold0 or len(pli0) != 3:
        e6.arret(1, "fichiers manquants ou cas du pli 0 incohérents :\n  - " + "\n  - ".join(
            manquants + ([f"pli 0 : 05m {pli0} ≠ nnU-Net fold_0 {fold0}"] if pli0 != fold0 else [])))
    print(f"Fichiers présents : 15 cas d'entraînement ; pli 0 = {', '.join(pli0)} ; TotalSegmentator (04b) : "
          f"{', '.join(ts_cas) or 'absent, comparaison (c) non essayée'}.")
    os.makedirs(sortie, exist_ok=True)
    print(f"Copie de l'affichage : {e6.ouvrir_sortie(journal, sortie, SCRIPT)}")
    e6.poser_verrou(sortie, "verrou_06b", a.ignorer_verrou)

    # Contrôle de plausibilité (seuils figés) : 05j relu s'il existe, sinon recalculé.
    s05j = e6.lire_json(autres["05j"], PHASE) if os.path.isfile(autres["05j"]) else None
    mes_05j = {em._cle_cas(c): m for c, m in ((s05j or {}).get("mesures", {}).get("prediction") or {}).items()}
    ver_05j = {em._cle_cas(c): m for c, m in ((s05j or {}).get("mesures", {}).get("verite") or {}).items()}
    print(f"Contrôle de plausibilité de A : {'mesures de synthese_05j.json (section 23)' if s05j else 'recalculé (05j absent)'}.")
    ctl = {"A": {}, "N0": {}, "S0": {}, "T": {}}
    for c in cas:
        print(f"  {c} : contrôle de plausibilité...", flush=True)
        mes, raison = mesures_05j_conformes(mes_05j.get(c), f[c]["image"], f[c]["post"])
        if raison:
            print(f"    mesures de 05j NON réutilisées pour {c} ({raison}) : recalculées.")
        ctl["A"][c] = controle(c, f[c]["image"], f[c]["post"], seuils, mes)
        if c in pli0:
            ctl["N0"][c] = controle(c, f[c]["image"], f[c]["brut"], seuils)
            ctl["S0"][c] = controle(c, f[c]["image"], f[c]["S0"], seuils)
        if c in ts_cas:
            ctl["T"][c] = controle(c, f[c]["image"], f[c]["T"], seuils)
    app_05j = {em._cle_cas(x): r for x, r in ((s05j or {}).get("application") or {}).items()}
    ecarts_05j = [c for c in cas if c in app_05j and em._json(ctl["A"][c]["resultat"]) != em._json(app_05j[c])]
    res_ctl = {m: relire(m, ctl[m], seuils, sorted(ctl[m])) for m in ctl if ctl[m]}

    # Métriques (un cas en mémoire à la fois).
    import numpy as np
    met = {"A": {}, "brut": {}, "S0": {}, "T": {}}
    egares, vdens, anomalies, plafonds = {m: {} for m in ("A", "N0", "S0", "T")}, {}, [], {}
    for c in cas:
        print(f"  {c} : Dice, HD95, NSD (MONAI)...", flush=True)
        img = e6.ouvrir_nifti(f[c]["image"], PHASE)                 # en-tête seulement : espacement et forme
        ref_img = e6.ouvrir_nifti(f[c]["verite"], PHASE)
        ok, motif = em.geometrie(img, ref_img)
        if not ok:
            anomalies.append(f"{c} : vérité de géométrie différente de l'image ({motif})")
        ref = np.asanyarray(ref_img.dataobj).astype(np.uint8)
        esp, forme = tuple(float(x) for x in img.header.get_zooms()[:3]), tuple(img.shape[:3])
        plafonds[c] = em.plafond_hd95(forme, esp)
        # (clé des métriques, fichier, contrôle utilisé pour classer, étiquettes) ; brut = N0 sur le pli 0
        jeux = [("A", "post", "A", em.TOUTES_ETIQUETTES), ("brut", "brut", "N0" if c in pli0 else None, em.TOUTES_ETIQUETTES)]
        jeux += [("S0", "S0", "S0", em.TOUTES_ETIQUETTES)] if c in pli0 else []
        jeux += [("T", "T", "T", em.ETIQUETTES_C)] if c in ts_cas else []
        for cle, fichier, modele_ctl, etiq in jeux:
            pimg = e6.ouvrir_nifti(f[c][fichier], PHASE)
            gok, gmotif = em.geometrie(pimg, ref_img)
            if not gok:
                anomalies.append(f"{c} {cle} : géométrie de la prédiction différente de la vérité ({gmotif})")
            pred = np.asanyarray(pimg.dataobj).astype(np.uint8) if gok else np.zeros_like(ref)
            r_ctl = res_ctl[modele_ctl][c] if modele_ctl else {"regles_declenchees": []}
            met[cle][c], eg = em.mesurer_prediction(pred, ref, esp, forme, etiq, r_ctl, cle == "T", gok, gmotif)
            if modele_ctl:
                egares[modele_ctl][c] = eg
            del pred
        besoin = any(em.besoins_classement(em.restreindre_T(res_ctl[m][c]) if m == "T" else res_ctl[m][c])[1]
                     for m in res_ctl if c in res_ctl[m])
        if besoin:
            mv, raison = mesures_05j_conformes(ver_05j.get(c), f[c]["image"], f[c]["verite"])
            if raison:
                print(f"    mesures de la vérité de 05j NON réutilisées pour {c} ({raison}) : recalculées.")
            vdens[c] = mv or controle(c, f[c]["image"], f[c]["verite"], seuils)["mesures"]
            vdens[c] = em.normaliser_mesures(vdens[c])
        del ref

    # Les trois comparaisons, comme en phase C (verdict SANS VALEUR : cas d'entraînement).
    entrees = {"cas": cas, "seuils": seuils, "verites_densites": vdens, "egares": egares,
               "metriques": {"A": met["A"], "N0": {c: met["brut"][c] for c in pli0}, "S0": met["S0"],
                             **({"T": met["T"]} if ts_cas else {})},
               "controles": res_ctl, "remarques_cas": {"1019": "phase d'injection atypique (D27)"}}
    try:
        res = em.evaluer(entrees)
    except em.ArretClassement as err:             # jamais un classement silencieux (D43)
        e6.arret(em.CODE_ARRET_CLASSEMENT, f"{err} Rien n'est écrit. Envoyer la sortie ; ne rien corriger seul.")
    if ts_cas:                                    # D43 : scores de T marqués dans toutes les sorties
        res["comparaisons"]["c"]["T"]["mention"] = MENTION_TS
        res["mention_totalsegmentator"] = MENTION_TS

    # Concordance (P9).
    ecarts, maxi = [], {"dice_brut": 0.0, "dice_post": 0.0, "dice_S0": 0.0, "volume_verite_ml": 0.0}
    noms = {f"mmwhs_{c}" for c in cas}
    tests = {f"mmwhs_{c}" for c in e6.CAS_TEST_ATTENDUS}
    for cle, quoi in (("brut", "summary_brut"), ("A", "summary_post")):
        e6.verifier_chemin(autres[quoi], PHASE)
        lus, _ = em.pp.lire_summary(autres[quoi], noms, tests)
        for c in cas:
            for k in em.TOUTES_ETIQUETTES:
                x, y = met[cle][c][k], lus[f"mmwhs_{c}"][k]
                d = ecart_dice(x["dice"], y["dice"])
                if x["cas_degenere"] == "D2" and y["dice"] == 0:
                    d = 0.0                             # nnU-Net donne 0, le protocole exclut (D2) : écart attendu
                if d is None:
                    continue
                maxi["dice_" + ("brut" if cle == "brut" else "post")] = max(maxi["dice_" + ("brut" if cle == "brut" else "post")], d)
                if d > TOL_DICE or (x["n_pred"], x["n_ref"]) != (y["n_pred"], y["n_ref"]):
                    ecarts.append(f"{c} {em.NOMS[k]} ({quoi}) : Dice {x['dice']} contre {y['dice']} ; voxels "
                                  f"{x['n_pred']}/{x['n_ref']} contre {y['n_pred']}/{y['n_ref']}")
    if os.path.isfile(autres["05i"]):
        s05i = e6.lire_json(autres["05i"], PHASE)
        for c, x in s05i.get("cas", {}).items():
            for k, y in x.get("structures", {}).items():
                d = ecart_dice(met["A"][em._cle_cas(c)][int(k)]["dice"], y.get("dice_apres"))
                if d is not None and d > TOL_DICE:
                    ecarts.append(f"{em._cle_cas(c)} {em.NOMS[int(k)]} (synthese_05i) : Dice {met['A'][em._cle_cas(c)][int(k)]['dice']} contre {y.get('dice_apres')}")
    for c, x in bilan["cas"].items():
        for k, nom in NOMS_05M.items():
            d = ecart_dice(met["S0"][em._cle_cas(c)][k]["dice"], x["dice"].get(nom))
            if d is not None:
                maxi["dice_S0"] = max(maxi["dice_S0"], d)
                if d > TOL_DICE:
                    ecarts.append(f"{em._cle_cas(c)} {em.NOMS[k]} (bilan_05m) : Dice {met['S0'][em._cle_cas(c)][k]['dice']} contre {x['dice'].get(nom)}")
    s05h = e6.lire_json(autres["05h"], PHASE)
    lignes_05h = {em._cle_cas(l["cas"]): l for l in s05h.get("lignes", [])}
    for c in cas:
        for k, nom in em.STRUCTURES.items():
            ref_ml = lignes_05h.get(c, {}).get(f"ml_{nom}")
            d = float("inf") if ref_ml is None else abs(met["A"][c][k]["vol_ref_ml"] - ref_ml)
            maxi["volume_verite_ml"] = max(maxi["volume_verite_ml"], d)
            if d > TOL_ML:
                ecarts.append(f"{c} {em.NOMS[k]} (05h) : volume de la vérité {met['A'][c][k]['vol_ref_ml']} mL contre {ref_ml}")
    for cle in met:                                   # plausibilité de HD95, NSD, Dice
        for c in met[cle]:
            for k, x in met[cle][c].items():
                if x["cas_degenere"]:
                    ecarts.append(f"{c} {em.NOMS[k]} ({cle}) : cas dégénéré {x['cas_degenere']} inattendu (vérité complète)")
                    continue
                if not (0 <= x["dice"] <= 1 and 0 <= x["nsd_2mm"] <= 1 and 0 <= x["hd95_mm"] <= plafonds[c]) \
                        or (x["dice"] == 1 and (x["hd95_mm"] != 0 or x["nsd_2mm"] != 1)):
                    ecarts.append(f"{c} {em.NOMS[k]} ({cle}) : valeurs impossibles Dice {x['dice']}, HD95 {x['hd95_mm']}, NSD {x['nsd_2mm']}")
    ecarts += [f"anomalie : {x}" for x in anomalies]
    avert = ([f"contrôle de A : résultat (seuils figés) différent de application de 05j pour {ecarts_05j} "
              "(05j avait appliqué les seuils PROPOSÉS ; information)"] if ecarts_05j else [])
    qualifie = not ecarts
    conclusion = "OUTIL DE MESURE QUALIFIÉ" if qualifie else "OUTIL DE MESURE NON QUALIFIÉ"
    qualif = {"conclusion": conclusion, "ecarts": ecarts, "avertissements": avert, "ecarts_maximaux": maxi,
              "tolerances": {"dice": TOL_DICE, "volume_ml": TOL_ML}, "pli0": pli0, "cas_totalseg": ts_cas,
              "mention_totalseg": MENTION_TS if ts_cas else None,
              "controle_A_source": "synthese_05j.json (mesures)" if s05j else "recalculé",
              "verifications_demarrage": verifs, "versions": versions, "duree_s": time.time() - t0,
              "tracabilite": e6.tracabilite(PHASE, SCRIPT), "sources": {"fichiers": f, "autres": autres}}
    dossier = os.path.join(sortie, f"essai_{e6.SESSION}")
    noms_ecrits = em.publier(dossier, res, "Étape 6, phase A : ESSAI À BLANC sur les cas d'entraînement (PAS le test)",
                             "Cas d'ENTRAÎNEMENT déjà vus à l'étape 5 : ce verdict et ce classement des alertes n'ont "
                             "AUCUNE VALEUR ; ils servent seulement à essayer le code de la phase C."
                             + (f" TotalSegmentator (1002, 1003) : {MENTION_TS}." if ts_cas else ""),
                             {"qualification_06b.json": qualif})
    print(f"\nÉcrit dans {dossier} : {', '.join(noms_ecrits)}")
    if ts_cas:
        print(f"TotalSegmentator ({', '.join(ts_cas)}) : {MENTION_TS}.")
    print("\n" + "=" * 100)
    print(f"Écarts maximaux : Dice brut {maxi['dice_brut']:.2e}, post-traité {maxi['dice_post']:.2e}, S0 {maxi['dice_S0']:.2e} "
          f"(tolérance {TOL_DICE}) ; volumes de la vérité {maxi['volume_verite_ml']:.2e} mL (tolérance {TOL_ML}).")
    for x in avert:
        print(f"AVERTISSEMENT : {x}")
    for x in ecarts:
        print(f"ÉCART : {x}")
    v = res.get("verdict", {})
    print(f"Essai du verdict (SANS VALEUR, cas d'entraînement) : {v.get('verdict')} ; fausses alertes (essai) : "
          f"{res['comparaisons']['a']['A']['estimation_fausses_alertes']['erreur_probable']['texte']}.")
    print(f"Durée : {(time.time() - t0) / 60:.1f} min ; fin : {e6.heure()}")
    print(f"\nRésultat : {conclusion}" + ("" if qualifie else f" ({len(ecarts)} écart(s) ci-dessus)"))
    sys.exit(0 if qualifie else 7)                # le verrou est retiré à la sortie (atexit, comme 05c)


if __name__ == "__main__":
    main()
