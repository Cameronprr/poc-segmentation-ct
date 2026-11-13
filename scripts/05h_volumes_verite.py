"""
Étape 5h — Volumes des 7 structures sur la VÉRITÉ TERRAIN des 15 cas d'entraînement (règles R2 et R3, D28/D29).
LECTURE SEULE : aucun fichier nnU-Net n'est modifié ; le lot de test n'est jamais lu.

Contexte : le contrôle de plausibilité (docs/specification_controle_plausibilite.md, section 5) compare
  - R2 : le VOLUME (mL) de chaque structure à la plage observée sur la vérité ;
  - R3 : cinq RAPPORTS DE VOLUMES (rapport des nombres de voxels dans une même image) : aorte / cavité VG,
         AP / aorte, OG / cavité VG, VD / cavité VG, myocarde / cavité VG.
Ce script MESURE ces grandeurs et décrit leur répartition. Il NE FIXE AUCUN SEUIL et n'écrit RIEN dans
config/ : le calcul des plages [min × (1 − m), max × (1 + m)] et le gel de config/seuils_plausibilite.json
viendront ensuite, à part (spécification, section 6).

Ce que fait le script :
  0. garde-fous (comme 05e) : chaque cas doit être en ENTRAÎNEMENT dans le découpage figé
     (splits/mmwhs_split.json, lu seulement) ; tout cas de test est refusé ; seuls labelsTr, dataset.json
     et l'EN-TÊTE des images de imagesTr sont lus (jamais leurs voxels), jamais imagesTs ni labelsTs ;
     --sortie doit être hors du dépôt et hors des dossiers nnU-Net ; étiquettes de dataset.json vérifiées ;
  1. pour chaque cas : nombre de voxels de chaque étiquette 1 à 7 du masque, volume en mL
     (voxels × produit des 3 espacements de l'en-tête de l'IMAGE / 1000, comme la spécification, section 3),
     et les 5 rapports de R3 (rapport des nombres de voxels) ;
  2. contrôles signalés, jamais corrigés : forme du masque différente de l'image (cas NON mesuré, comme
     05e) ; affine ou espacement du masque différents de l'image (mesuré quand même, signalé) ; valeurs
     non entières ; étiquette inattendue ; structure absente (volume nul : rapports non calculés). Le
     tableau par cas donne aussi la forme, l'orientation (image et masque) et les étiquettes présentes ;
  3. statistiques par grandeur : n, min (cas), max (cas), médiane, moyenne, écart-type, en deux versions
     principales : « tous » (15 cas) et « sans_1019 » (14 cas, ceux du calibrage, D28) ; en complément, la
     même chose PAR PATIENT (paires 1001/1020 et 1005/1012 moyennées d'abord) et, à titre indicatif,
     par famille d'épaisseur de coupe (0,45 / 0,625 mm), sans 1019.
Entrées :
  --raw       dossier nnUNet_raw (défaut /content/drive/MyDrive/poc_ct/nnunet_raw) ; seuls
              Dataset101_MMWHS/labelsTr, dataset.json et les en-têtes de imagesTr sont lus
  --cas       numéros des cas (défaut : les 15 cas d'entraînement du découpage figé)
  --sortie    dossier de sortie dans Drive (défaut /content/drive/MyDrive/poc_ct/analyses/05h_volumes)
  --split     découpage figé (défaut : ../splits/mmwhs_split.json à côté du script)
Sorties (uniquement dans --sortie, jamais dans le dépôt : valeurs par cas, Q6) :
  volumes_verite.csv (une ligne par cas) ; synthese_05h.json (lignes, statistiques, définitions, anomalies)
Codes de sortie : 0 = mesure faite (lire les lignes ANOMALIE et AVERTISSEMENT) ; 1 = garde-fou (rien n'est écrit).

Vocabulaire :
  - voxel : petit pavé de l'image 3D ; son volume (mm³) = produit des 3 espacements lus dans l'en-tête ;
  - mL : millilitre = 1 000 mm³ (le VG d'un adulte contient de l'ordre de 100 mL de sang) ;
  - rapport de volumes : volume d'une structure divisé par celui d'une autre, dans le même examen. Il ne
    dépend ni de la taille des voxels (elle s'annule) ni, en grande partie, de la taille du patient ;
  - affine : petite matrice de l'en-tête qui place chaque voxel dans l'espace (en mm) ; orientation :
    sens des axes qu'on en déduit (ex. RAS = droite, avant, tête). Image et masque doivent avoir les mêmes ;
  - médiane : valeur du milieu une fois les cas triés ; écart-type : dispersion autour de la moyenne
    (formule « échantillon », division par n − 1) ;
  - par patient : les deux examens d'un même patient (D20) sont d'abord moyennés, pour ne pas compter deux
    fois la même personne dans la moyenne et l'écart-type (le min et le max n'en souffrent pas, spéc. 6.3).
"""
import argparse, csv, datetime, importlib.util, json, os, re, statistics

# Fonctions reprises du script 05d (qui charge lui-même 05a) plutôt que réécrites ; l'import n'exécute rien.
ICI = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("analyse05d", os.path.join(ICI, "05d_analyser_1019.py"))
ana = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ana)
prep = ana.prep                                  # module du script 05a, déjà chargé par 05d

# --- Paramètres fixes -------------------------------------------------------------------------
STRUCTURES = {1: "myocarde_VG", 2: "OG", 3: "cavite_VG", 4: "OD", 5: "VD", 6: "aorte", 7: "AP"}  # dataset.json
RAPPORTS = (("aorte_sur_cavite_VG", 6, 3), ("AP_sur_aorte", 7, 6), ("OG_sur_cavite_VG", 2, 3),   # R3, spéc. § 5
            ("VD_sur_cavite_VG", 5, 3), ("myocarde_sur_cavite_VG", 1, 3))
CAS_ETUDIE = "mmwhs_1019"                        # exclu du calibrage (D28, spécification § 6.4)
TOLERANCE = 1e-3                                 # mm : écart d'affine ou d'espacement masque/image toléré (comme 05e)
NOM_CSV, NOM_JSON = "volumes_verite.csv", "synthese_05h.json"
GRANDEURS = [f"ml_{n}" for n in STRUCTURES.values()] + [r for r, _, _ in RAPPORTS]


def lire_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=f"{prep.DRIVE}/nnunet_raw")
    p.add_argument("--cas", nargs="+", default=None)
    p.add_argument("--sortie", default=f"{prep.DRIVE}/analyses/05h_volumes")
    p.add_argument("--split", default=os.path.join(prep.DEPOT, "splits", "mmwhs_split.json"))
    return p.parse_args()


# --- 0. Garde-fous (mêmes règles que 05e ; rien n'est écrit avant qu'ils soient passés) --------
def verifier(a):
    raw, sortie = os.path.abspath(a.raw), os.path.abspath(a.sortie)
    interdits = [d for d in sortie.split(os.sep) if ana.DOSSIERS_INTERDITS.match(d)]
    interdits += [f"${v}" for v in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
                  if os.environ.get(v) and prep.dans(sortie, os.environ[v])]
    if prep.dans(sortie, prep.DEPOT) or prep.dans(sortie, raw) or interdits:
        ana.arret(1, f"--sortie ({sortie}) est dans le dépôt ou dans un dossier nnU-Net ({interdits}). "
                     "Choisir par exemple /content/drive/MyDrive/poc_ct/analyses/05h_volumes.")
    try:
        entr, test, groupes, _ = prep.lire_decoupage(a.split)    # lecture seule du découpage figé
    except (OSError, ValueError, KeyError) as err:
        ana.arret(1, f"découpage figé illisible ou incohérent ({a.split}) : {err}")
    ds = os.path.join(raw, ana.NOM_DS)
    chemins = {}
    for n in dict.fromkeys(a.cas or [c[-4:] for c in sorted(entr)]):   # sans doublon, ordre gardé
        cas = f"mmwhs_{n}"
        if not re.fullmatch(r"\d{4}", n):
            ana.arret(1, f"numéro de cas invalide : {n} (attendu : 4 chiffres, ex. 1019).")
        if cas in test:
            ana.arret(1, f"{cas} fait partie du LOT DE TEST : refusé (le test ne sert qu'à l'étape 6, D18).")
        if cas not in entr:
            ana.arret(1, f"{cas} n'est pas dans le lot d'entraînement du découpage figé : refusé.")
        c = {"entete_image": os.path.join(ds, "imagesTr", f"{cas}_0000.nii.gz"),
             "verite": os.path.join(ds, "labelsTr", f"{cas}.nii.gz")}
        if any(d in ("imagesTs", "labelsTs") for x in c.values() for d in x.split(os.sep)):
            ana.arret(1, "un chemin passe par imagesTs ou labelsTs : refusé.")
        manquants = [x for x in c.values() if not os.path.isfile(x)]
        if manquants:
            ana.arret(1, f"fichiers introuvables (Drive relié ?) : {manquants}")
        chemins[cas] = c
    with open(os.path.join(ds, "dataset.json"), encoding="utf-8") as f:
        etiquettes_ds = json.load(f).get("labels")
    if etiquettes_ds != ana.ETIQUETTES:
        ana.arret(1, f"les étiquettes de dataset.json ({etiquettes_ds}) ne sont pas celles attendues ({ana.ETIQUETTES}).")
    strates, pb = prep.lire_strates(ds, entr)                    # en-têtes seuls des 15 cas (définition 03b/05a)
    groupe = {c: "/".join(x[-4:] for x in g) for g in groupes for c in g}
    print(f"{len(chemins)} cas vérifiés, tous en ENTRAÎNEMENT (aucun cas de test). "
          "Étiquettes de dataset.json conformes (1 myocarde VG ... 7 artère pulmonaire).")
    return sortie, chemins, strates, pb, groupe


# --- 1. Mesures d'un cas (un seul masque en mémoire) ------------------------------------------
def mesurer_cas(cas, c, strate, groupe):
    """Voxels, volumes (mL) et rapports de R3 d'après la vérité. Renvoie (ligne du tableau, anomalies).
    La ligne a TOUJOURS les mêmes colonnes ; une valeur non mesurée vaut None (case vide dans le CSV)."""
    import nibabel as nib
    import numpy as np
    img, lab = nib.load(c["entete_image"]), nib.load(c["verite"])   # nib.load ne lit que l'en-tête
    esp = [float(x) for x in img.header.get_zooms()[:3]]            # mm, en-tête de l'IMAGE (spéc. § 3)
    vol_voxel = esp[0] * esp[1] * esp[2]                           # mm³
    l = {"cas": cas, "groupe_patient": groupe or "", "epaisseur_coupe_mm": round(esp[2], 4), "famille_mm": strate,
         "forme": " x ".join(str(int(x)) for x in img.shape[:3]),
         "espacement_mm": " x ".join(f"{x:.4f}" for x in esp), "volume_voxel_mm3": round(vol_voxel, 6),
         "orientation_image": "".join(nib.aff2axcodes(img.affine)),
         "orientation_masque": "".join(nib.aff2axcodes(lab.affine)), "etiquettes_presentes": ""}
    for nom in STRUCTURES.values():
        l[f"vox_{nom}"], l[f"ml_{nom}"] = None, None
    for r, _, _ in RAPPORTS:
        l[r] = None
    anomalies = []
    if img.shape[:3] != lab.shape[:3]:
        # Masque et image sur des grilles différentes : l'espacement de l'image ne décrit pas le masque.
        # On ne devine pas : cas NON mesuré (comme 05e), à examiner à la main.
        anomalies.append(f"forme du masque {lab.shape} ≠ image {img.shape} : cas NON mesuré")
        l["anomalies"] = " ; ".join(anomalies)
        return l, anomalies
    if not np.allclose(img.affine, lab.affine, atol=TOLERANCE):
        anomalies.append(f"affine du masque différente de l'image (écart max {np.abs(img.affine - lab.affine).max():.4f}) : "
                         "volumes calculés quand même (le compte de voxels n'en dépend pas)")
    esp_lab = [float(x) for x in lab.header.get_zooms()[:3]]
    if not np.allclose(esp, esp_lab, atol=TOLERANCE):
        anomalies.append(f"espacement du masque {esp_lab} ≠ image {esp} : volume calculé avec celui de l'IMAGE")
    donnees = np.asanyarray(lab.dataobj)                           # seules données lues : le masque
    valeurs, nombres = np.unique(donnees, return_counts=True)
    del donnees                                                    # libère la mémoire avant le cas suivant
    if not np.all(np.mod(valeurs, 1) == 0):                        # ex. 0,5 après une interpolation
        anomalies.append(f"valeurs NON ENTIÈRES dans le masque : {[float(v) for v in valeurs if v % 1][:10]} "
                         "(comptées seulement si elles valent exactement 1 à 7)")
    compte = {}
    for v, n in zip(valeurs, nombres):
        if float(v) % 1 == 0:
            compte[int(v)] = int(n)
    l["etiquettes_presentes"] = " ".join(str(v) for v in sorted(compte))
    inattendues = sorted(set(compte) - set(ana.ETIQUETTES.values()))
    if inattendues:
        anomalies.append(f"étiquettes inattendues dans le masque : {inattendues} (ignorées)")
    for e, nom in STRUCTURES.items():
        l[f"vox_{nom}"] = compte.get(e, 0)
        # Structure absente : volume laissé VIDE (None), pas 0. Un 0 ramènerait le minimum de R2 à 0, et la
        # borne basse (min × 0,7) ne signalerait plus jamais une structure trop petite (aorte de 1019).
        l[f"ml_{nom}"] = round(l[f"vox_{nom}"] * vol_voxel / 1000, 3) if l[f"vox_{nom}"] else None
        if l[f"vox_{nom}"] == 0:
            anomalies.append(f"{nom} (étiquette {e}) absente de la vérité : volume non compté, rapports non calculés")
    for r, num, den in RAPPORTS:                                   # rapport des NOMBRES DE VOXELS (spéc. R3)
        n, d = l[f"vox_{STRUCTURES[num]}"], l[f"vox_{STRUCTURES[den]}"]
        l[r] = round(n / d, 4) if n and d else None
    l["anomalies"] = " ; ".join(anomalies)
    return l, anomalies


# --- 2. Statistiques descriptives (aucun seuil) -----------------------------------------------
def statistiques(paires):
    """paires = [(valeur, étiquette du cas ou du patient)]. n, min, max (avec qui), médiane, moyenne, écart-type."""
    v = [(x, qui) for x, qui in paires if x is not None]
    if not v:
        return None
    xs = [x for x, _ in v]
    return {"n": len(v), "min": min(v)[0], "qui_min": min(v)[1], "max": max(v)[0], "qui_max": max(v)[1],
            "mediane": round(statistics.median(xs), 4), "moyenne": round(statistics.mean(xs), 4),
            "ecart_type": round(statistics.stdev(xs), 4) if len(xs) > 1 else None}


def par_patient(lignes, cle):
    """Une valeur par patient : moyenne des cas d'une paire (D20), valeur du cas sinon."""
    patients = {}
    for l in lignes:
        patients.setdefault(l["groupe_patient"] or l["cas"][-4:], []).append(l[cle])
    return [(round(statistics.mean(v), 4) if None not in v else None, p) for p, v in patients.items()]


def toutes_statistiques(lignes):
    sans = [l for l in lignes if l["cas"] != CAS_ETUDIE]
    versions = {"tous": lignes, "sans_1019": sans}
    st = {g: {k: statistiques([(l[k], l["cas"][-4:]) for l in ls]) for k in GRANDEURS} for g, ls in versions.items()}
    for g, ls in versions.items():                                 # complément : par patient
        st[f"par_patient_{g}"] = {k: statistiques(par_patient(ls, k)) for k in GRANDEURS}
    for f in prep.STRATES_MM:                                      # indicatif : par famille d'épaisseur, sans 1019
        ls = [l for l in sans if l["famille_mm"] == f]
        st[f"famille_{f}_sans_1019"] = {k: statistiques([(l[k], l["cas"][-4:]) for l in ls]) for k in GRANDEURS}
    return st


# --- Affichage et écriture --------------------------------------------------------------------
def nb(x, n=1):
    """Nombre formaté avec n décimales, ou « — » si absent."""
    return "—" if x is None else f"{x:.{n}f}"


def afficher(lignes, st):
    noms = list(STRUCTURES.values())
    print("\n=== Volumes de la vérité terrain (mL) — R2 ===")
    print(f"  {'cas':<11}{'épais.':>7} {'paire':>10}" + "".join(f"{n:>12}" for n in noms))
    for l in lignes:
        print(f"  {l['cas']:<11}{l['epaisseur_coupe_mm']:>7} {l['groupe_patient']:>10}"
              + "".join(f"{nb(l['ml_' + n]):>12}" for n in noms))
    print("\n=== Rapports de volumes (nombres de voxels) — R3 ===")
    print(f"  {'cas':<11}" + "".join(f"{r:>24}" for r, _, _ in RAPPORTS))
    for l in lignes:
        print(f"  {l['cas']:<11}" + "".join(f"{nb(l[r], 3):>24}" for r, _, _ in RAPPORTS))
    print("\n=== Statistiques (descriptives : AUCUN seuil n'est fixé ici) ===")
    print("  tous = 15 cas ; sans_1019 = 14 cas (calibrage, D28) ; par patient = paires moyennées (complément)")
    for k in GRANDEURS:
        print(f"  {k}")
        for g in ("tous", "sans_1019", "par_patient_tous", "par_patient_sans_1019"):
            s = st[g][k]
            print(f"    {g:<22}" + ("aucune valeur" if s is None else
                  f"n={s['n']:<3}min {nb(s['min'], 3)} ({s['qui_min']}), max {nb(s['max'], 3)} ({s['qui_max']}), "
                  f"médiane {nb(s['mediane'], 3)}, moyenne {nb(s['moyenne'], 3)}, écart-type {nb(s['ecart_type'], 3)}"))
        familles = [(fm, st[f"famille_{fm}_sans_1019"][k]) for fm in prep.STRATES_MM]
        print("    par épaisseur (indicatif, sans 1019) : " + " | ".join(
              f"{fm} mm : " + ("—" if s is None else f"n={s['n']}, médiane {nb(s['mediane'], 3)}, "
                                                      f"min {nb(s['min'], 3)}, max {nb(s['max'], 3)}")
              for fm, s in familles))


def ecrire(chemin, ecrire_contenu):
    """Écriture dans un fichier temporaire puis renommage : jamais de fichier à moitié écrit."""
    with open(chemin + ".tmp", "w", encoding="utf-8", newline="") as fic:
        ecrire_contenu(fic)
    os.replace(chemin + ".tmp", chemin)


def main():
    a = lire_arguments()
    sortie, chemins, strates, pb_strates, groupe = verifier(a)
    anomalies = {"strates (en-têtes des 15 cas)": pb_strates} if pb_strates else {}
    lignes = []
    for cas, c in chemins.items():
        print(f"  {cas} : lecture du masque (labelsTr) et de l'en-tête de l'image...", flush=True)
        l, an = mesurer_cas(cas, c, strates.get(cas), groupe.get(cas))
        if an:
            anomalies[cas] = an
        lignes.append(l)
    st = toutes_statistiques(lignes)
    afficher(lignes, st)
    avertissements = ["statistiques descriptives : aucun seuil n'est fixé ici ; rien n'est écrit dans config/ "
                      "(seuils R2/R3 calculés et figés à part, spécification § 6)"]
    mesures = [l["cas"] for l in lignes if l[f"vox_{STRUCTURES[3]}"] is not None]   # cas réellement mesurés
    if len(mesures) != prep.NB_CAS_ENTRAINEMENT:
        non_mesures = sorted(set(chemins) - set(mesures))
        avertissements.insert(0, f"{len(mesures)} cas mesurés sur {prep.NB_CAS_ENTRAINEMENT}"
                                 + (f" (non mesurés : {', '.join(non_mesures)})" if non_mesures else "")
                                 + " : statistiques INCOMPLÈTES, ne pas les utiliser pour calibrer R2/R3")
    absentes = sorted({f"{l['cas'][-4:]} ({k[3:]})" for l in lignes for k in GRANDEURS[:7]
                       if l.get(f"vox_{k[3:]}") == 0})
    if absentes:                                                   # structure absente de la vérité (spéc. : jamais)
        avertissements.insert(0, f"structures absentes de la vérité : {', '.join(absentes)} : "
                                 "statistiques INCOMPLÈTES, ne pas les utiliser pour calibrer R2/R3")
    if CAS_ETUDIE not in chemins:
        avertissements.append(f"{CAS_ETUDIE} non mesuré : « tous » et « sans_1019 » sont identiques")
    print()
    for x in avertissements:
        print(f"AVERTISSEMENT : {x}")
    for qui, liste in anomalies.items():
        for x in liste:
            print(f"ANOMALIE {qui} : {x}")

    os.makedirs(sortie, exist_ok=True)                          # le seul dossier où le script écrit
    def contenu_csv(fic):
        w = csv.DictWriter(fic, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
    ecrire(os.path.join(sortie, NOM_CSV), contenu_csv)
    synthese = {"objet": "volumes (mL) des étiquettes 1 à 7 et rapports de volumes, vérité terrain des cas "
                         "d'entraînement : base de calibrage des règles R2 et R3 du contrôle de plausibilité (D28, D29)",
                "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "split_sha256": prep.sha256_fichier(a.split), "code_commit": prep.commit_git(),
                "cas_lus": len(lignes), "cas_mesures": len(mesures), "avertissements": avertissements,
                "definitions": {
                    "volume_ml": "nombre de voxels de l'étiquette × produit des 3 espacements (mm) de l'en-tête de "
                                 "l'image / 1000 (spécification § 3)",
                    "rapports_R3": {r: f"voxels {STRUCTURES[n]} / voxels {STRUCTURES[d]} (même image)" for r, n, d in RAPPORTS},
                    "tous": "tous les cas mesurés", "sans_1019": f"tous sauf {CAS_ETUDIE} (calibrage, D28 et spéc. § 6.4)",
                    "par_patient_*": "complément : paires de groupes_patients moyennées d'abord (moyenne des valeurs "
                                     "des deux cas, y compris pour les rapports), puis statistiques sur les patients",
                    "famille_*_sans_1019": "indicatif : cas d'une même épaisseur de coupe (strate de 05a), sans 1019",
                    "ecart_type": "écart-type d'échantillon (division par n − 1)"},
                "statistiques": st, "anomalies": anomalies, "lignes": lignes, "sources": chemins}
    ecrire(os.path.join(sortie, NOM_JSON), lambda fic: json.dump(synthese, fic, indent=2, ensure_ascii=False))
    print(f"\nÉcrit dans {sortie} : {NOM_CSV}, {NOM_JSON}")
    print("Rien n'a été écrit dans config/, nnUNet_raw, nnUNet_preprocessed ni nnUNet_results ; aucun cas de test lu.")


if __name__ == "__main__":
    main()
