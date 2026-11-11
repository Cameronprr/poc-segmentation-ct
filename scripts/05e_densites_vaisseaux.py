"""
Étape 5e — Densité des vaisseaux sur les 15 cas d'entraînement : repérer la phase d'injection (suite de D27).
LECTURE SEULE : aucun fichier nnU-Net n'est modifié ; le lot de test n'est jamais lu.

Contexte : sur mmwhs_1019, l'aorte (253 HU) est MOINS dense que l'artère pulmonaire (416 HU), à l'inverse de
1001 (aorte 562 HU, AP 192 HU) : l'examen a été pris à une phase d'injection atypique (D27). Ce script
mesure la même chose sur CHACUN des cas d'entraînement, pour savoir si 1019 est seul dans ce cas, et pour
préparer les seuils du contrôle de plausibilité (D28). Il NE FIXE AUCUN SEUIL : il décrit.

Ce que fait le script :
  0. garde-fous (repris du script 05d) : chaque cas doit être en ENTRAÎNEMENT dans le découpage figé
     (splits/mmwhs_split.json, lu seulement) ; tout cas de test est refusé ; seuls imagesTr, labelsTr et
     dataset.json sont lus, jamais imagesTs ni labelsTs ; --sortie doit être hors du dépôt et hors des
     dossiers nnU-Net ; les étiquettes de dataset.json doivent être celles du script 03 (3 = cavité VG) ;
  1. pour chaque cas, UN À LA FOIS en mémoire, d'après la VÉRITÉ TERRAIN : densité (HU) médiane, p5 et p95
     de l'aorte (6), de l'artère pulmonaire (7), de la cavité VG (3) et du VD (5), puis médiane au « cœur »
     de chaque région (voxels à plus de 3 mm de son bord : fonction intensites() de 05d, même définition) ;
  2. MESURE DE RÉFÉRENCE (spécification D28, section 5) : médiane au cœur ; si le cœur est vide (région trop
     fine), médiane de la région entière, et le repli est noté (colonne ref_repli_region_entiere) ;
  3. écarts aorte − AP, aorte − VG, AP − VD et rapport aorte / AP, sur la région entière ET sur la référence ;
  4. DRAPEAU : « INVERSÉ » si médiane aorte < médiane AP ; « proche » si l'aorte est moins de 20 % plus dense
     que l'AP (rapport < 1,20 : repère INDICATIF, non validé). Calculé sur la région entière (drapeau) et
     sur la référence (drapeau_ref) ;
  5. tableau trié (rapport aorte/AP croissant : les cas suspects en premier), CSV, JSON de synthèse avec la
     répartition (min, médiane, max) sur tous les cas, SANS 1019 (spécification D28, section 6.4) et hors
     cas INVERSÉS, et une figure.
Les anomalies (forme ou en-tête image/masque différents, étiquette inattendue, structure absente) sont
SIGNALÉES, jamais corrigées ; un cas dont l'image et le masque n'ont pas la même forme n'est pas mesuré.

Entrées :
  --raw       dossier nnUNet_raw (défaut /content/drive/MyDrive/poc_ct/nnunet_raw) ; seuls
              Dataset101_MMWHS/imagesTr, labelsTr et dataset.json sont lus
  --cas       numéros des cas (défaut : les 15 cas d'entraînement du découpage figé)
  --sortie    dossier de sortie dans Drive (défaut /content/drive/MyDrive/poc_ct/analyses/05e_densites)
  --split     découpage figé (défaut : ../splits/mmwhs_split.json à côté du script)
Sorties (uniquement dans --sortie) :
  densites_vaisseaux.csv (une ligne par cas) ; synthese_05e.json (lignes + répartition) ;
  densites_vaisseaux.png (figure)
Codes de sortie : 0 = analyse faite (lire les lignes ANOMALIE) ; 1 = garde-fou (rien n'est écrit).

Vocabulaire :
  - HU (unités Hounsfield) : échelle des intensités CT (eau = 0, air = -1000, sang injecté 250 à 500) ;
  - phase d'injection : moment de l'acquisition après l'injection du produit de contraste. Tôt, le produit
    est surtout dans le cœur droit et l'artère pulmonaire ; plus tard, dans le cœur gauche et l'aorte ;
  - p5 / p95 : 5 % des voxels sont sous p5, 5 % au-dessus de p95 (étendue « normale » sans les extrêmes) ;
  - effet de bord : au bord d'une région, un voxel mélange sang et paroi, sa densité est faussée.
"""
import argparse, csv, datetime, importlib.util, json, os, re, statistics

# Fonctions reprises du script 05d (qui charge lui-même 05a) plutôt que réécrites ; l'import n'exécute rien.
ICI = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("analyse05d", os.path.join(ICI, "05d_analyser_1019.py"))
ana = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ana)
prep = ana.prep                                  # module du script 05a, déjà chargé par 05d

# --- Paramètres fixes -------------------------------------------------------------------------
STRUCTURES = {"aorte": 6, "AP": 7, "VG": 3, "VD": 5}   # étiquettes vérifiées dans dataset.json (ana.ETIQUETTES)
CAS_ETUDIE = "mmwhs_1019"                        # exclu d'une des répartitions (calibrage de R6, spéc. D28 § 6.4)
SEUIL_PROCHE = 1.20                              # repère INDICATIF : aorte < 1,20 × AP → « proche »
NOM_CSV, NOM_JSON, NOM_FIGURE = "densites_vaisseaux.csv", "synthese_05e.json", "densites_vaisseaux.png"
COULEURS = {"aorte": "#2a78d6", "AP": "#eb6834", "VG": "#1baf7a", "VD": "#eda100"}   # bleu, orange, vert d'eau, jaune
SYMBOLES = {"aorte": "o", "AP": "s", "VG": "^", "VD": "D"}  # forme en plus de la couleur (lisible sans les couleurs)
FOND_DRAPEAU = {"INVERSÉ": "#f6c9c9", "proche": "#fbe3a6"}


def lire_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=f"{prep.DRIVE}/nnunet_raw")
    p.add_argument("--cas", nargs="+", default=None)
    p.add_argument("--sortie", default=f"{prep.DRIVE}/analyses/05e_densites")
    p.add_argument("--split", default=os.path.join(prep.DEPOT, "splits", "mmwhs_split.json"))
    return p.parse_args()


# --- 0. Garde-fous (mêmes règles que 05d ; rien n'est écrit avant qu'ils soient passés) --------
def verifier(a):
    raw, sortie = os.path.abspath(a.raw), os.path.abspath(a.sortie)
    interdits = [d for d in sortie.split(os.sep) if ana.DOSSIERS_INTERDITS.match(d)]
    interdits += [f"${v}" for v in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
                  if os.environ.get(v) and prep.dans(sortie, os.environ[v])]
    if prep.dans(sortie, prep.DEPOT) or prep.dans(sortie, raw) or interdits:
        ana.arret(1, f"--sortie ({sortie}) est dans le dépôt ou dans un dossier nnU-Net ({interdits}). "
                     "Choisir par exemple /content/drive/MyDrive/poc_ct/analyses/05e_densites.")
    try:
        entr, test, groupes, _ = prep.lire_decoupage(a.split)    # lecture seule du découpage figé
    except (OSError, ValueError, KeyError) as err:
        ana.arret(1, f"découpage figé illisible ou incohérent ({a.split}) : {err}")
    ds = os.path.join(raw, ana.NOM_DS)
    numeros = a.cas or [c[-4:] for c in sorted(entr)]
    chemins = {}
    for n in dict.fromkeys(numeros):                             # sans doublon, ordre gardé
        cas = f"mmwhs_{n}"
        if not re.fullmatch(r"\d{4}", n):
            ana.arret(1, f"numéro de cas invalide : {n} (attendu : 4 chiffres, ex. 1019).")
        if cas in test:
            ana.arret(1, f"{cas} fait partie du LOT DE TEST : refusé (le test ne sert qu'à l'étape 6, D18).")
        if cas not in entr:
            ana.arret(1, f"{cas} n'est pas dans le lot d'entraînement du découpage figé : refusé.")
        c = {"image": os.path.join(ds, "imagesTr", f"{cas}_0000.nii.gz"),
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
          "Étiquettes de dataset.json conformes : 3 = cavite_VG, 6 = aorte_ascendante, 7 = artere_pulmonaire.")
    return sortie, chemins, strates, pb, groupe


# --- 1. Mesures d'un cas (un seul scanner en mémoire) -----------------------------------------
def mesurer_cas(c):
    """Statistiques HU (fonction intensites() de 05d) de l'aorte, de l'AP, du VG et du VD de la vérité.
    Renvoie (statistiques ou None, anomalies, épaisseur de coupe lue dans l'en-tête)."""
    import nibabel as nib
    import numpy as np
    img, lab = nib.load(c["image"]), nib.load(c["verite"])
    epaisseur = round(float(img.header.get_zooms()[2]), 4)
    if img.shape[:3] != lab.shape[:3]:
        return None, [f"forme du masque {lab.shape} ≠ image {img.shape} : cas NON mesuré"], epaisseur
    anomalies = []
    if not np.allclose(img.affine, lab.affine, atol=1e-3):
        anomalies.append(f"affine du masque différente de l'image (écart max "
                         f"{np.abs(img.affine - lab.affine).max():.4f}) : mesure faite voxel à voxel")
    esp = [float(x) for x in img.header.get_zooms()[:3]]        # mm le long de chaque axe du tableau
    ct = img.get_fdata(dtype=np.float32)                         # pente/ordonnée de l'en-tête appliquées
    ver = np.asanyarray(lab.dataobj)
    presentes = {int(x) for x in np.unique(ver)}
    if presentes - set(ana.ETIQUETTES.values()):
        anomalies.append(f"étiquettes inattendues dans le masque : {sorted(presentes - set(ana.ETIQUETTES.values()))}")
    stats = {nom: ana.intensites(ct, ver == e, esp) for nom, e in STRUCTURES.items()}
    anomalies += [f"{nom} (étiquette {STRUCTURES[nom]}) absente de la vérité" for nom, s in stats.items() if s is None]
    del ct, ver                                                  # libère le scanner avant le cas suivant
    return stats, anomalies, epaisseur


def ecart(x, y):
    return None if x is None or y is None else round(x - y)


def rapport(x, y):
    return None if x is None or y is None or y <= 0 else round(x / y, 2)


def drapeau(ao, ap):
    """« INVERSÉ » si aorte < AP ; « proche » si aorte < SEUIL_PROCHE × AP ; « » sinon."""
    if ao is None or ap is None:
        return "incomplet"
    if ao < ap:
        return "INVERSÉ"
    return "proche" if ap > 0 and ao < SEUIL_PROCHE * ap else ""


def ligne(cas, stats, epaisseur, strate, groupe):
    """Une ligne du tableau (clés plates, pour le CSV)."""
    l = {"cas": cas, "epaisseur_coupe_mm": epaisseur, "famille_mm": strate, "groupe_patient": groupe or ""}
    for nom in STRUCTURES:
        s = (stats or {}).get(nom) or {}
        for cle in ("mediane", "p5", "p95", "coeur_mediane", "coeur_voxels"):
            l[f"{nom}_{cle}"] = s.get(cle)
    # Référence = médiane au cœur ; repli sur la région entière si le cœur est vide (noté).
    repli = [n for n in STRUCTURES if l[f"{n}_coeur_mediane"] is None and l[f"{n}_mediane"] is not None]
    for nom in STRUCTURES:
        l[f"{nom}_ref"] = l[f"{nom}_mediane"] if nom in repli else l[f"{nom}_coeur_mediane"]
    l["ref_repli_region_entiere"] = " ".join(repli)
    for suffixe, cle in (("", "mediane"), ("_ref", "ref")):
        ao, ap, vg, vd = (l[f"{n}_{cle}"] for n in STRUCTURES)
        l[f"aorte_moins_AP{suffixe}"], l[f"aorte_moins_VG{suffixe}"] = ecart(ao, ap), ecart(ao, vg)
        l[f"AP_moins_VD{suffixe}"], l[f"aorte_sur_AP{suffixe}"] = ecart(ap, vd), rapport(ao, ap)
        l[f"drapeau{suffixe}"] = drapeau(ao, ap)
    return l


# --- 2. Répartition sur les cas (pour calibrer plus tard les seuils de D28) --------------------
def repartition(lignes, cle):
    v = [(l[cle], l["cas"]) for l in lignes if l[cle] is not None]
    if not v:
        return None
    return {"n": len(v), "min": min(v)[0], "cas_min": min(v)[1], "mediane": statistics.median(x for x, _ in v),
            "max": max(v)[0], "cas_max": max(v)[1]}


def ordre(l):
    """Tri du tableau : rapport aorte/AP croissant (cas suspects en premier), cas non mesurés à la fin."""
    return (l["aorte_sur_AP"] is None, l["aorte_sur_AP"] or 0, l["cas"])


# --- 3. Figure --------------------------------------------------------------------------------
def figure(lignes, chemin):
    import matplotlib
    matplotlib.use("Agg")                                       # figure enregistrée en PNG
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    x = range(len(lignes))
    fig, (h, b) = plt.subplots(2, 1, figsize=(max(10, 0.8 * len(lignes) + 3), 9), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
    for i, l in zip(x, lignes):                                 # bande de fond + mot sur les cas signalés
        if l["drapeau"] in FOND_DRAPEAU:
            for ax in (h, b):
                ax.axvspan(i - 0.45, i + 0.45, color=FOND_DRAPEAU[l["drapeau"]], zorder=0)
    for k, nom in enumerate(STRUCTURES):                        # médiane + barre p5-p95 par structure
        pts = [(i + (k - 1.5) * 0.2, l) for i, l in zip(x, lignes) if l[f"{nom}_mediane"] is not None]
        h.errorbar([p for p, _ in pts], [l[f"{nom}_mediane"] for _, l in pts],
                   yerr=[[l[f"{nom}_mediane"] - l[f"{nom}_p5"] for _, l in pts],
                         [l[f"{nom}_p95"] - l[f"{nom}_mediane"] for _, l in pts]],
                   fmt=SYMBOLES[nom], ms=8, color=COULEURS[nom], elinewidth=1.5, capsize=3,
                   label=f"{nom} (étiquette {STRUCTURES[nom]}) : médiane, barre p5-p95")
    haut = max([l[f"{n}_p95"] for l in lignes for n in STRUCTURES if l[f"{n}_p95"] is not None], default=1)
    for i, l in zip(x, lignes):
        if l["drapeau"]:
            h.text(i, haut * 1.04, l["drapeau"], ha="center", fontsize=9, fontweight="bold")
    h.set_ylim(top=haut * 1.12)
    h.set_ylabel("densité (HU), vérité terrain")
    h.set_title("Densité de l'aorte, de l'artère pulmonaire (AP), de la cavité VG et du VD (région entière), cas d'entraînement\n"
                "(trié par rapport aorte/AP croissant ; fond rouge = INVERSÉ, aorte < AP ; fond jaune = proche, "
                f"aorte < {SEUIL_PROCHE:.2f} × AP, repère indicatif)".replace(".", ","), fontsize=11)
    h.grid(axis="y", color="#dddddd", lw=0.6)
    fig.legend(handles=h.get_legend_handles_labels()[0] + [Patch(color=c, label=f"cas {d}") for d, c in FOND_DRAPEAU.items()],
               loc="lower center", fontsize=9, ncol=5)                # sous la figure : ne cache aucun point
    pts = [(i, l["aorte_sur_AP"]) for i, l in zip(x, lignes) if l["aorte_sur_AP"] is not None]
    b.plot([p for p, _ in pts], [r for _, r in pts], "o", ms=8, color="#52514e")
    b.axhline(1.0, color="#0b0b0b", lw=1)
    b.axhline(SEUIL_PROCHE, color="#0b0b0b", lw=1, ls="--")
    b.text(len(lignes) - 0.5, 1.0, " 1 : aorte = AP", va="center", fontsize=8)
    b.text(len(lignes) - 0.5, SEUIL_PROCHE, f" {SEUIL_PROCHE:.2f} : repère « proche »".replace(".", ","), va="center", fontsize=8)
    b.set_ylabel("rapport aorte / AP")
    b.grid(axis="y", color="#dddddd", lw=0.6)
    b.set_xticks(list(x))
    b.set_xticklabels([f"{l['cas'][-4:]}\n{l['epaisseur_coupe_mm']} mm" + (f"\n({l['groupe_patient']})"
                       if l["groupe_patient"] else "") for l in lignes], fontsize=8)
    b.set_xlim(-0.6, len(lignes) + 1.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(chemin, dpi=90)
    plt.close(fig)


# --- Affichage et écriture --------------------------------------------------------------------
COLONNES = (("cas", "cas", 10), ("épais.", "epaisseur_coupe_mm", 6), ("groupe", "groupe_patient", 9),
            ("aorte", "aorte_mediane", 5), ("AP", "AP_mediane", 5), ("VG", "VG_mediane", 5), ("VD", "VD_mediane", 5),
            ("ao/AP", "aorte_sur_AP", 5), ("drapeau", "drapeau", 9), ("|", "|", 1),
            ("ao réf", "aorte_ref", 6), ("AP réf", "AP_ref", 6), ("VG réf", "VG_ref", 6), ("VD réf", "VD_ref", 6),
            ("ao−AP", "aorte_moins_AP_ref", 5), ("ao−VG", "aorte_moins_VG_ref", 5), ("AP−VD", "AP_moins_VD_ref", 5),
            ("ao/AP", "aorte_sur_AP_ref", 5), ("drapeau réf", "drapeau_ref", 11), ("repli", "ref_repli_region_entiere", 6))
STATS = (("aorte_moins_VG", "aorte − VG (HU)"), ("AP_moins_VD", "AP − VD (HU)"), ("aorte_moins_AP", "aorte − AP (HU)"),
         ("aorte_sur_AP", "aorte / AP"), ("VG", "VG (HU)"))            # VG : contraste présent (règle R7)


def afficher(lignes, rep):
    print("\n=== Densités médianes (HU) d'après la vérité terrain, triées par rapport aorte/AP croissant ===")
    print("  À gauche de « | » : région entière. À droite : RÉFÉRENCE = médiane au cœur (voxels à plus de 3 mm du bord),")
    print("  ou région entière si le cœur est vide (structures listées dans « repli »).")
    print("  ao = aorte ; AP = artère pulmonaire ; VG = cavité VG ; VD = ventricule droit ; écarts en HU")
    print("  " + " ".join(f"{t:>{w}}" for t, _, w in COLONNES))
    for l in lignes:
        print("  " + " ".join(f"{ana.txt(l.get(k, k)):>{w}}" for _, k, w in COLONNES))
    print(f"\nDrapeaux : INVERSÉ = médiane aorte < médiane AP ; proche = aorte moins de "
          f"{100 * (SEUIL_PROCHE - 1):.0f} % plus dense que l'AP (repère INDICATIF, choisi a priori, non validé).")
    print("\nRépartition sur les cas, mesure de RÉFÉRENCE (descriptif : aucun seuil n'est fixé ici) :")
    for cle, libelle in STATS:
        for groupe in ("tous", "sans_1019"):
            r = rep[groupe][cle + "_ref"]
            print(f"  {libelle:<16} {groupe:<10}" + ("aucune valeur" if r is None else
                  f"min {r['min']} ({r['cas_min'][-4:]}), médiane {r['mediane']}, max {r['max']} ({r['cas_max'][-4:]}), "
                  f"n = {r['n']}"))


def ecrire(chemin, ecrire_contenu):
    """Écriture dans un fichier temporaire puis renommage : jamais de fichier à moitié écrit."""
    with open(chemin + ".tmp", "w", encoding="utf-8", newline="") as f:
        ecrire_contenu(f)
    os.replace(chemin + ".tmp", chemin)


def main():
    a = lire_arguments()
    sortie, chemins, strates, pb_strates, groupe = verifier(a)
    anomalies = {"strates (en-têtes des 15 cas)": pb_strates} if pb_strates else {}
    lignes = []
    for cas, c in chemins.items():
        print(f"  {cas} : lecture de imagesTr et labelsTr...", flush=True)
        stats, an, epaisseur = mesurer_cas(c)
        if an:
            anomalies[cas] = an
        lignes.append(ligne(cas, stats, epaisseur, strates.get(cas), groupe.get(cas)))
    lignes.sort(key=ordre)
    cles = [c + s for c, _ in STATS for s in (("_mediane", "_ref") if c == "VG" else ("", "_ref"))]
    groupes = {"tous": lignes, "sans_1019": [l for l in lignes if l["cas"] != CAS_ETUDIE],
               "hors_inverses": [l for l in lignes if l["drapeau"] != "INVERSÉ"]}
    rep = {g: {k: repartition(ls, k) for k in cles} for g, ls in groupes.items()}
    afficher(lignes, rep)
    for qui, liste in anomalies.items():
        for x in liste:
            print(f"ANOMALIE {qui} : {x}")

    os.makedirs(sortie, exist_ok=True)                          # le seul dossier où le script écrit
    def contenu_csv(f):
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
    ecrire(os.path.join(sortie, NOM_CSV), contenu_csv)
    figure(lignes, os.path.join(sortie, NOM_FIGURE))
    synthese = {"objet": "densités HU de l'aorte, de l'AP, de la cavité VG et du VD (vérité terrain) sur les cas "
                         "d'entraînement : phase d'injection (D27), base de calibration du contrôle de plausibilité (D28)",
                "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "split_sha256": prep.sha256_fichier(a.split), "code_commit": prep.commit_git(),
                "cas_mesures": len(lignes),
                "avertissements": ([] if len(lignes) == prep.NB_CAS_ENTRAINEMENT else
                                   [f"{len(lignes)} cas sur {prep.NB_CAS_ENTRAINEMENT} : répartition INCOMPLÈTE, "
                                    "ne pas l'utiliser pour calibrer D28"]) +
                                  ["statistiques descriptives : aucun seuil n'est fixé ici (D28 : seuils figés à part, "
                                   "avant tout usage sur le test)"],
                "mesure_de_reference": {
                    "choix": "coeur", "colonnes": "suffixe _ref (structures et écarts) ; drapeau_ref",
                    "definition": f"médiane des HU au cœur de la région : voxels à plus de {ana.COEUR_MM:g} mm de son bord "
                                  "(distance euclidienne en mm ; fonction intensites() de 05d, même définition)",
                    "repli": "si le cœur est vide, médiane de la région entière ; structures concernées dans "
                             "ref_repli_region_entiere (spécification D28, section 5)",
                    "region_entiere": "colonnes sans suffixe (médiane, p5, p95) : comparaison, et drapeau principal "
                                      "(médiane aorte < médiane AP)"},
                "definitions": {"INVERSE": "médiane aorte < médiane AP",
                                "proche": f"médiane aorte < {SEUIL_PROCHE} × médiane AP (repère indicatif, non validé)",
                                "tous": "tous les cas mesurés",
                                "sans_1019": f"tous les cas sauf {CAS_ETUDIE} (calibrage de R6 sans 1019, spécification D28 § 6.4)",
                                "hors_inverses": "cas sans drapeau INVERSÉ (calculé sur les médianes entières)",
                                "statistiques": "n, min, médiane, max, cas du min et du max ; écarts signés (R4 et R5 "
                                                "portent sur la valeur absolue : regarder le min ET le max)"},
                "seuil_proche_indicatif": SEUIL_PROCHE, "repartition": rep, "anomalies": anomalies,
                "cas_signales": {l["cas"]: l["drapeau"] for l in lignes if l["drapeau"]},
                "lignes": lignes, "sources": chemins}
    ecrire(os.path.join(sortie, NOM_JSON), lambda f: json.dump(synthese, f, indent=2, ensure_ascii=False))
    print(f"\nÉcrit dans {sortie} : {NOM_CSV}, {NOM_JSON}, {NOM_FIGURE}")
    print("Rien n'a été écrit dans nnUNet_raw, nnUNet_preprocessed ni nnUNet_results ; aucun cas de test lu.")


if __name__ == "__main__":
    main()
