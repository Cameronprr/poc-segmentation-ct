"""
Étape 5d — Analyser l'échec du pli 0 de nnU-Net sur les vaisseaux de mmwhs_1019 (décision D26).
LECTURE SEULE : aucun fichier nnU-Net n'est modifié ; le lot de test n'est jamais lu.

Contexte : sur mmwhs_1019 (validation du pli 0), l'aorte (étiquette 6) a un Dice de 0,00 et l'artère
pulmonaire (7) de 0,46, alors que les cavités et le myocarde sont entre 0,89 et 0,96 et que l'aorte est à 0,97-0,98 sur
1001 et 1020. Ce script rassemble les indicateurs qui aident à trancher entre trois hypothèses :
  (a) cadrage du scanner : aorte coupée par le bord, champ de vue différent des autres cas ;
  (b) qualité de l'image : rehaussement du contraste trop faible, bruit trop fort ;
  (c) annotation : étiquette 6 posée sur une autre structure, aorte descendante incluse, 6 et 7 permutées...
Il NE CONCLUT PAS : il imprime les chiffres et ce que chacun suggérerait. La décision reste humaine.

Ce que fait le script, pour chaque cas demandé (défaut : 1019, et 1001 comme cas de comparaison réussi) :
  0. garde-fous : chaque cas doit être en ENTRAÎNEMENT dans le découpage figé (splits/mmwhs_split.json,
     lu seulement) ; tout cas de test (1004, 1010, 1011, 1013, 1014) est refusé ; seuls imagesTr, labelsTr
     et fold_<pli>/validation sont lus ; --sortie doit être hors du dépôt et hors des dossiers nnU-Net ;
     les noms d'étiquettes de dataset.json doivent être ceux du script 03 ;
  1. géométrie : forme, espacement, épaisseur de coupe, orientation, étendue du champ en mm ; image,
     vérité et prédiction doivent avoir la même grille (sinon arrêt, code 6) ; en plus, en-têtes SEULS des
     15 cas d'entraînement pour situer le champ de 1019 parmi eux ;
  2. aorte (6) et artère pulmonaire (7), dans la vérité ET dans la prédiction : voxels, volume, niveaux bas
     et haut, boîte englobante, distance aux bords du champ, centre de gravité, morceaux ;
  3. matrice de confusion : ce que la prédiction met à la place de la vérité (et l'inverse) ;
  4. intensités (HU) dans les vaisseaux et les cavités de la vérité, bruit, contraste ;
  5. une figure par cas (coupes axiale, coronale, sagittale par le centre de chaque vaisseau + projection) ;
  6. un récapitulatif des indicateurs, rangés par hypothèse.

Entrées :
  --raw       dossier nnUNet_raw (défaut /content/drive/MyDrive/poc_ct/nnunet_raw) ; seuls
              Dataset101_MMWHS/imagesTr, labelsTr et dataset.json sont lus
  --results   dossier nnUNet_results (défaut /content/drive/MyDrive/poc_ct/nnunet_results) ; seul
              .../nnUNetTrainer_250epochs_reprise10__nnUNetPlans__3d_fullres/fold_<pli>/validation/ est lu
  --pli       pli dont on lit la validation (défaut 0)
  --cas       numéros des cas, le premier est le cas étudié (défaut : 1019 1001)
  --sortie    dossier de sortie dans Drive (défaut /content/drive/MyDrive/poc_ct/analyses/05d_1019)
  --split     découpage figé (défaut : ../splits/mmwhs_split.json à côté du script)
Sorties (uniquement dans --sortie) :
  analyse_mmwhs_XXXX.png (une figure par cas) ; synthese_05d.json (tous les chiffres imprimés)
Codes de sortie :
  0 = analyse faite ; 1 = garde-fou (rien n'est écrit) ; 6 = image, vérité et prédiction n'ont pas la même
  forme (comparaison voxel à voxel impossible).

Vocabulaire :
  - HU (unités Hounsfield) : échelle des intensités CT. Eau = 0, air = -1000, sang injecté de produit de
    contraste : souvent 250 à 500 ;
  - centre de gravité : position moyenne des voxels d'un masque (en mm). Décalage vérité/prédiction grand
    = la prédiction est ailleurs ;
  - morceaux (composantes connexes) : blocs séparés d'un masque (connexité 26, comme le script 04b) ;
  - matrice de confusion : pour chaque étiquette de la vérité, comment ses voxels ont été prédits ;
  - bruit : écart-type des HU au cœur d'une région homogène (sang de la cavité VG, à plus de 3 mm du bord) ;
  - CNR (rapport contraste sur bruit) : (HU sang VG - HU myocarde) / bruit. Plus il est grand, plus les
    bords sont faciles à voir ;
  - repère RAS : on remet chaque volume dans le même sens (1er axe vers la droite du patient, 2e vers
    l'avant, 3e vers la tête). Les indices de coupe affichés sont dans ce repère : coupe 0 = la plus basse.
"""
import argparse, datetime, importlib.util, json, math, os, re, sys

# Fonctions communes reprises du script 05a (même dossier) plutôt que réécrites ; l'import n'exécute rien.
ICI = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("prep05a", os.path.join(ICI, "05a_preparer_nnunet.py"))
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)

# --- Paramètres fixes -------------------------------------------------------------------------
NOM_DS = "Dataset101_MMWHS"
TRAINER, CONFIGURATION = "nnUNetTrainer_250epochs_reprise10", "3d_fullres"   # comme le script 05c
ETIQUETTES = {"background": 0, "myocarde_VG": 1, "oreillette_gauche": 2, "cavite_VG": 3,   # script 03
              "oreillette_droite": 4, "ventricule_droit": 5, "aorte_ascendante": 6, "artere_pulmonaire": 7}
NOMS = {v: k for k, v in ETIQUETTES.items()}
VAISSEAUX = (6, 7)
REFERENCES = (3, 1, 2, 4, 5)            # cavité VG, myocarde, OG, OD, VD : repères d'intensité et de position
FENETRE = (-200, 600)                   # fenêtre d'affichage en HU, adaptée au CT injecté
COEUR_MM = 3.0                          # « cœur » d'une région = voxels à plus de 3 mm de son bord
BORD_COUPES = 1                         # à 0 ou 1 coupe du bord : le masque « touche » le bord du champ
COULEURS = {("verite", 6): "#FF3B30", ("pred", 6): "#FFE600",     # aorte : rouge (vérité), jaune (prédite)
            ("verite", 7): "#2F80FF", ("pred", 7): "#FF40FF"}     # AP : bleu (vérité), magenta (prédite)
NOM_FIGURE, NOM_SYNTHESE = "analyse_mmwhs_{n}.png", "synthese_05d.json"
DOSSIERS_INTERDITS = re.compile(r"^(nnunet_raw|nnunet_preprocessed|nnunet_results|Dataset\d{3}_.*|"
                                r"imagesTr|labelsTr|imagesTs|labelsTs)$", re.I)


def arret(code, message):
    print(f"ARRÊT : {message}")
    sys.exit(code)


def lire_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=f"{prep.DRIVE}/nnunet_raw")
    p.add_argument("--results", default=f"{prep.DRIVE}/nnunet_results")
    p.add_argument("--pli", type=int, default=0, choices=range(prep.NB_PLIS))
    p.add_argument("--cas", nargs="+", default=["1019", "1001"])
    p.add_argument("--sortie", default=f"{prep.DRIVE}/analyses/05d_1019")
    p.add_argument("--split", default=os.path.join(prep.DEPOT, "splits", "mmwhs_split.json"))
    return p.parse_args()


# --- 0. Garde-fous (rien n'est écrit avant qu'ils soient passés) -------------------------------
def verifier(a):
    raw, res, sortie = (os.path.abspath(x) for x in (a.raw, a.results, a.sortie))
    interdits = [d for d in sortie.split(os.sep) if DOSSIERS_INTERDITS.match(d)]
    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if os.environ.get(var) and prep.dans(sortie, os.environ[var]):
            interdits.append(f"${var}")
    if prep.dans(sortie, prep.DEPOT) or prep.dans(sortie, raw) or prep.dans(sortie, res) or interdits:
        arret(1, f"--sortie ({sortie}) est dans le dépôt ou dans un dossier nnU-Net ({interdits}). "
                 "Choisir par exemple /content/drive/MyDrive/poc_ct/analyses/05d_1019.")
    try:
        entr, test, _, _ = prep.lire_decoupage(a.split)     # lecture seule du découpage figé
    except (OSError, ValueError, KeyError) as err:
        arret(1, f"découpage figé illisible ou incohérent ({a.split}) : {err}")
    ds = os.path.join(raw, NOM_DS)
    validation = os.path.join(res, NOM_DS, f"{TRAINER}__nnUNetPlans__{CONFIGURATION}", f"fold_{a.pli}", "validation")
    chemins = {}
    for n in a.cas:
        cas = f"mmwhs_{n}"
        if not re.fullmatch(r"\d{4}", n):
            arret(1, f"numéro de cas invalide : {n} (attendu : 4 chiffres, ex. 1019).")
        if cas in test:
            arret(1, f"{cas} fait partie du LOT DE TEST : refusé (le test ne sert qu'à l'étape 6, D18).")
        if cas not in entr:
            arret(1, f"{cas} n'est pas dans le lot d'entraînement du découpage figé : refusé.")
        c = {"image": os.path.join(ds, "imagesTr", f"{cas}_0000.nii.gz"),
             "verite": os.path.join(ds, "labelsTr", f"{cas}.nii.gz"),
             "pred": os.path.join(validation, f"{cas}.nii.gz")}
        # Ceinture et bretelles : aucun chemin lu ne peut passer par un dossier de test.
        if any(d in ("imagesTs", "labelsTs") for x in c.values() for d in x.split(os.sep)):
            arret(1, "un chemin passe par imagesTs ou labelsTs : refusé.")
        manquants = [x for x in c.values() if not os.path.isfile(x)]
        if manquants:
            arret(1, f"fichiers introuvables (Drive relié ? cas absent de la validation du pli {a.pli} ?) : {manquants}")
        chemins[n] = c
    with open(os.path.join(ds, "dataset.json"), encoding="utf-8") as f:
        etiquettes_ds = json.load(f).get("labels")
    if etiquettes_ds != ETIQUETTES:
        arret(1, f"les étiquettes de dataset.json ({etiquettes_ds}) ne sont pas celles attendues ({ETIQUETTES}).")
    print(f"Cas vérifiés : {', '.join('mmwhs_' + n for n in a.cas)} sont en ENTRAÎNEMENT (aucun cas de test).")
    print(f"Étiquettes de dataset.json conformes : 3 = cavite_VG, 6 = aorte_ascendante, 7 = artere_pulmonaire.")
    return ds, sortie, chemins, entr


# --- 1. Lecture et géométrie ------------------------------------------------------------------
def charger(c):
    """Lit image, vérité et prédiction. Les remet dans le repère RAS avec LA MÊME transformation (celle de
    l'image) : si les en-têtes différaient, les trois tableaux resteraient alignés voxel à voxel."""
    import nibabel as nib
    import numpy as np
    from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation
    imgs = {k: nib.load(c[k]) for k in ("image", "verite", "pred")}
    ref = imgs["image"]
    for k in ("verite", "pred"):
        if imgs[k].shape[:3] != ref.shape[:3]:
            arret(6, f"{c[k]} : forme {imgs[k].shape} différente de l'image {ref.shape}.")
    anomalies = [f"affine de {k} différente de celle de l'image (écart max "
                 f"{np.abs(imgs[k].affine - ref.affine).max():.4f}) : comparaison faite voxel à voxel"
                 for k in ("verite", "pred") if not np.allclose(imgs[k].affine, ref.affine, atol=1e-3)]
    ornt = io_orientation(ref.affine)
    affine = ref.affine @ inv_ornt_aff(ornt, ref.shape[:3])
    ct = apply_orientation(ref.get_fdata(dtype=np.float32), ornt)        # pente/ordonnée de l'en-tête appliquées
    ver = apply_orientation(np.asanyarray(imgs["verite"].dataobj), ornt).astype(np.uint8)
    pred = apply_orientation(np.asanyarray(imgs["pred"].dataobj), ornt).astype(np.uint8)
    esp = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))                     # mm le long de chaque axe RAS
    h = ref.header
    geo = {"forme_fichier": [int(x) for x in ref.shape[:3]],
           "espacement_fichier_mm": [round(float(x), 4) for x in h.get_zooms()[:3]],
           "epaisseur_coupe_mm": round(float(h.get_zooms()[2]), 4),     # définition des scripts 03b et 05a
           "orientation_fichier": "".join(nib.aff2axcodes(ref.affine)),
           "forme_RAS": [int(x) for x in ct.shape],
           "espacement_RAS_mm": [round(float(x), 4) for x in esp],
           "etendue_mm_RL_AP_IS": [round(float(s * e), 1) for s, e in zip(ct.shape, esp)],
           "codes_sform_qform": [int(h["sform_code"]), int(h["qform_code"])],
           "hu_p1_mediane_p99": [round(float(x)) for x in np.percentile(ct[::4, ::4, ::4], [1, 50, 99])],
           "hu_min_max": [round(float(ct.min())), round(float(ct.max()))],
           "etiquettes_verite": sorted(int(x) for x in np.unique(ver)),
           "etiquettes_prediction": sorted(int(x) for x in np.unique(pred)),
           "anomalies": anomalies}
    return ct, ver, pred, affine, esp, geo


def champs_entrainement(ds, entr):
    """En-têtes SEULS (aucune donnée) des cas d'entraînement : forme, espacement, étendue en mm."""
    import nibabel as nib
    import numpy as np
    lignes = {}
    for cas in sorted(entr):
        img = nib.load(os.path.join(ds, "imagesTr", f"{cas}_0000.nii.gz"))
        z = np.array(img.header.get_zooms()[:3], dtype=float)
        lignes[cas] = {"forme": [int(x) for x in img.shape[:3]], "espacement_mm": [round(float(x), 3) for x in z],
                       "etendue_mm": [round(float(x), 1) for x in np.array(img.shape[:3]) * z],
                       "orientation": "".join(nib.aff2axcodes(img.affine))}
    return lignes


# --- 2. Mesures d'un masque -------------------------------------------------------------------
def mm(affine, ijk):
    """Indices de voxel (repère RAS) → coordonnées en mm."""
    return [float(x) for x in (affine @ [*ijk, 1])[:3]]


def mesurer(m, affine, esp):
    import numpy as np
    from scipy import ndimage
    n = int(m.sum())
    if n == 0:
        return {"voxels": 0}
    bornes = [np.where(m.any(axis=tuple(x for x in range(3) if x != ax)))[0] for ax in range(3)]
    (i0, i1), (j0, j1), (k0, k1) = [(int(b.min()), int(b.max())) for b in bornes]
    centre = [float(x) for x in ndimage.center_of_mass(m)]
    boite = m[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1]                         # recadré : moins de mémoire
    etiq, nb = ndimage.label(boite, structure=np.ones((3, 3, 3)))     # connexité 26
    tailles = sorted(np.bincount(etiq.ravel())[1:].tolist(), reverse=True)
    vox_ml = float(np.prod(esp)) / 1000
    nx, ny, nz = m.shape
    return {"voxels": n, "volume_ml": round(n * vox_ml, 1),
            "coupe_basse": k0, "coupe_haute": k1,
            "niveau_bas_mm": round(mm(affine, [centre[0], centre[1], k0])[2], 1),
            "niveau_haut_mm": round(mm(affine, [centre[0], centre[1], k1])[2], 1),
            "hauteur_mm": round((k1 - k0 + 1) * float(esp[2]), 1),
            "boite_indices": [[i0, i1], [j0, j1], [k0, k1]],
            "boite_mm": [round((b1 - b0 + 1) * float(e), 1) for (b0, b1), e in zip(((i0, i1), (j0, j1), (k0, k1)), esp)],
            "bords_coupes": {"bas": k0, "haut": nz - 1 - k1, "droite": nx - 1 - i1, "gauche": i0,
                             "avant": ny - 1 - j1, "arriere": j0},
            "bord_bas_mm": round(k0 * float(esp[2]), 1), "bord_haut_mm": round((nz - 1 - k1) * float(esp[2]), 1),
            "centre_indices": [round(x, 1) for x in centre],
            "centre_mm": [round(x, 1) for x in mm(affine, centre)],
            "morceaux": int(nb), "morceaux_3_plus_gros_ml": [round(t * vox_ml, 2) for t in tailles[:3]]}


def confusion(ver, pred):
    """Matrice 8 × 8 : ligne = étiquette de la vérité, colonne = étiquette prédite (nombre de voxels).
    Calculée sur les voxels non nuls d'au moins un des deux masques (le fond/fond est déduit)."""
    import numpy as np
    sel = (ver > 0) | (pred > 0)
    cm = np.bincount(ver[sel].astype(np.int64) * 8 + pred[sel], minlength=64).reshape(8, 8)
    cm[0, 0] = ver.size - int(sel.sum())
    return cm


def intensites(ct, m, esp):
    """Statistiques HU d'un masque, puis de son « cœur » (voxels à plus de COEUR_MM du bord)."""
    import numpy as np
    from scipy import ndimage
    if not m.any():
        return None
    v = ct[m]
    r = {"mediane": round(float(np.median(v))), "p5": round(float(np.percentile(v, 5))),
         "p95": round(float(np.percentile(v, 95))), "ecart_type": round(float(v.std()), 1)}
    idx = np.where(m)
    sl = tuple(slice(max(int(x.min()) - 1, 0), int(x.max()) + 2) for x in idx)
    coeur = ndimage.distance_transform_edt(m[sl], sampling=esp) > COEUR_MM
    vc = ct[sl][coeur]
    r.update({"coeur_voxels": int(vc.size),
              "coeur_mediane": round(float(np.median(vc))) if vc.size else None,
              "coeur_ecart_type": round(float(vc.std()), 1) if vc.size else None})
    return r


def analyser(n, c, pli):
    """Toutes les mesures d'un cas. Renvoie (résultats, tableaux utiles pour la figure)."""
    import numpy as np
    ct, ver, pred, affine, esp, geo = charger(c)
    r = {"cas": f"mmwhs_{n}", "pli": pli, "geometrie": geo, "structures": {}, "intensites": {}}
    for lab in VAISSEAUX:
        mv, mp = mesurer(ver == lab, affine, esp), mesurer(pred == lab, affine, esp)
        dec = (None if not (mv["voxels"] and mp["voxels"]) else
               round(float(np.linalg.norm(np.subtract(mp["centre_mm"], mv["centre_mm"]))), 1))
        r["structures"][NOMS[lab]] = {"verite": mv, "prediction": mp, "decalage_centres_mm": dec}
    for lab in REFERENCES:
        r["structures"][NOMS[lab]] = {"verite": mesurer(ver == lab, affine, esp)}
    r["structures"]["coeur_entier"] = {"verite": mesurer(ver > 0, affine, esp)}   # union des 7 étiquettes
    cm = confusion(ver, pred)
    r["confusion_verite_vers_prediction"] = cm.tolist()
    for lab in VAISSEAUX + REFERENCES:
        r["intensites"][f"verite_{lab}_{NOMS[lab]}"] = intensites(ct, ver == lab, esp)
    for lab in VAISSEAUX:
        r["intensites"][f"prediction_{lab}_{NOMS[lab]}"] = intensites(ct, pred == lab, esp)
    return r, (ct, ver, pred, esp)


# --- 5. Figure --------------------------------------------------------------------------------
def vue(t):
    """Sens des radiologues pour imshow(origin='lower') (comme le script 04b) : axiale = droite du patient
    à gauche, avant en haut ; coronale = droite à gauche, tête en haut ; sagittale = avant à gauche."""
    import numpy as np
    return np.flip(t, 0).T


def coupe_utile(m, ax, centre):
    """Indice de coupe le long de l'axe ax : le centre, ou la coupe la plus remplie si le centre est vide."""
    import numpy as np
    autres = tuple(x for x in range(3) if x != ax)
    if np.take(m, centre, axis=ax).any():
        return centre
    return int(np.argmax(m.sum(axis=autres)))


def figure(n, r, tableaux, chemin):
    import matplotlib
    matplotlib.use("Agg")                                   # figure enregistrée en PNG
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    ct, ver, pred, esp = tableaux
    geo = r["geometrie"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 12.5))
    fig.suptitle(f"mmwhs_{n} (entraînement, validation du pli {r['pli']}, coupes de {geo['epaisseur_coupe_mm']} mm) — "
                 f"vérité contre prédiction nnU-Net — fenêtre {FENETRE[0]}/{FENETRE[1]} HU — "
                 "image entière : les bords de l'image sont les bords du champ", fontsize=13)
    for ligne, lab in zip(axes, VAISSEAUX):
        mv = ver == lab
        base = mv if mv.any() else (pred == lab)
        if not base.any():
            for ax in ligne:
                ax.axis("off")
            ligne[0].set_title(f"{NOMS[lab]} absente de la vérité et de la prédiction")
            continue
        centre = [int(round(x)) for x in np.argwhere(base).mean(axis=0)]
        i, j, k = (coupe_utile(base, ax_, centre[ax_]) for ax_ in range(3))
        quelle = "vérité" if mv.any() else "prédiction (vérité vide)"
        vues = [(ct[:, :, k], ver[:, :, k], pred[:, :, k], esp[1] / esp[0], f"axiale, coupe {k}", ("D", "G", "A", "P"), None),
                (ct[:, j, :], ver[:, j, :], pred[:, j, :], esp[2] / esp[0], "coronale", ("D", "G", "H", "B"), k),
                (ct[i, :, :], ver[i, :, :], pred[i, :, :], esp[2] / esp[1], "sagittale", ("A", "P", "H", "B"), k)]
        for ax, (fond, v2, p2, aspect, titre, lettres, niveau) in zip(ligne[:3], vues):
            ax.imshow(vue(fond), cmap="gray", vmin=FENETRE[0], vmax=FENETRE[1], origin="lower", aspect=aspect)
            dessiner_contours(ax, {(s, l): t == l for s, t in (("verite", v2), ("pred", p2)) for l in VAISSEAUX})
            if niveau is not None:
                ax.axhline(niveau, color="white", ls=":", lw=0.8)
            lettres_bords(ax, lettres)
            ax.set_title(f"{NOMS[lab]} — {titre} (centre de la {quelle})", fontsize=10)
        # Projection de face (comme une ombre) : tout le cœur de la vérité en gris, contours des projections
        # des deux vaisseaux. Le cadre de l'image est le bord du champ : on voit si un vaisseau le touche.
        ax = ligne[3]
        ax.imshow(vue((ver > 0).any(axis=1).astype(np.float32)), cmap="gray", vmin=0, vmax=3,
                  origin="lower", aspect=esp[2] / esp[0])
        dessiner_contours(ax, {(s, l): (t == l).any(axis=1) for s, t in (("verite", ver), ("pred", pred))
                               for l in VAISSEAUX})
        ax.axhline(k, color="white", ls=":", lw=0.8)
        lettres_bords(ax, ("D", "G", "H", "B"))
        ax.set_title("projection de face (gris = cœur de la vérité) ; cadre = bords du champ", fontsize=10)
    fig.legend(handles=[Line2D([], [], color=COULEURS[(s, l)], lw=2.4 if s == "verite" else 1.3,
                               label=f"{'vérité' if s == 'verite' else 'prédiction'} {NOMS[l]}")
                        for l in VAISSEAUX for s in ("verite", "pred")]
               + [Line2D([], [], color="black", ls=":", label="niveau de la coupe axiale (pointillé blanc)")],
               loc="lower center", ncol=5, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=3)
    fig.savefig(chemin, dpi=80)
    plt.close(fig)


def dessiner_contours(ax, masques):
    """Contours vérité (trait épais, dessous) puis prédiction (trait fin, dessus). masques : {(source, étiquette):
    masque 2D}. Là où les deux se confondent, on voit un liseré de la couleur vérité autour du trait fin."""
    import numpy as np
    for (source, lab), m in sorted(masques.items(), key=lambda x: x[0][0] != "verite"):
        if m.any():
            ax.contour(vue(m).astype(np.float32), levels=[0.5], colors=[COULEURS[(source, lab)]],
                       linewidths=2.4 if source == "verite" else 1.3)


def lettres_bords(ax, lettres):
    for (x, y), t in zip(((0.02, 0.5), (0.98, 0.5), (0.5, 0.97), (0.5, 0.03)), lettres):
        ax.text(x, y, t, transform=ax.transAxes, color="yellow", fontsize=9, ha="center", va="center")
    ax.set_xticks([]); ax.set_yticks([])


# --- Affichage lisible ------------------------------------------------------------------------
def val(r, *chemin):
    """r[a][b][c]... ou None si une clé manque (structure absente, région vide)."""
    for cle in chemin:
        if isinstance(r, dict) and cle in r:
            r = r[cle]
        elif isinstance(r, list) and isinstance(cle, int) and cle < len(r):
            r = r[cle]
        else:
            return None
    return r


def txt(x, fmt="{}"):
    return "—" if x is None else fmt.format(x)


def afficher_cas(r):
    g = r["geometrie"]
    print(f"\n=== {r['cas']} — 1. géométrie ===")
    for libelle, cle in (("forme (fichier)", "forme_fichier"), ("espacement (mm, fichier)", "espacement_fichier_mm"),
                         ("épaisseur de coupe (mm)", "epaisseur_coupe_mm"), ("orientation (fichier)", "orientation_fichier"),
                         ("étendue du champ (mm) D-G / A-P / bas-haut", "etendue_mm_RL_AP_IS"),
                         ("codes sform / qform", "codes_sform_qform"), ("HU p1 / médiane / p99", "hu_p1_mediane_p99"),
                         ("HU min / max", "hu_min_max"), ("étiquettes de la vérité", "etiquettes_verite"),
                         ("étiquettes prédites", "etiquettes_prediction")):
        print(f"  {libelle:<44}{g[cle]}")
    for an in g["anomalies"]:
        print(f"  ANOMALIE : {an}")

    print(f"\n=== {r['cas']} — 2. aorte (6) et artère pulmonaire (7) : vérité | prédiction ===")
    print("  (coupes comptées dans le repère RAS : coupe 0 = la plus basse ; niveaux en mm, plus grand = vers la tête)")
    for lab in VAISSEAUX:
        s = r["structures"][NOMS[lab]]
        print(f"  -- {lab} {NOMS[lab]} (décalage des centres de gravité vérité/prédiction : "
              f"{txt(s['decalage_centres_mm'], '{} mm')})")
        for libelle, extraire in LIGNES_STRUCTURE:
            cellules = [(extraire(s[src]) if s[src]["voxels"] else ("absente" if libelle == "voxels" else "—"))
                        for src in ("verite", "prediction")]
            print(f"    {libelle:<48}{str(cellules[0]):<32}| {cellules[1]}")

    cm = r["confusion_verite_vers_prediction"]
    print(f"\n=== {r['cas']} — 3. matrice de confusion, en % (0 fond, 1 myo VG, 2 OG, 3 VG, 4 OD, 5 VD, 6 aorte, 7 AP) ===")
    print("  Ce que devient la VÉRITÉ (ligne) dans la prédiction (colonnes) :")
    print("    " + " " * 23 + "".join(f"{'→' + str(k):>7}" for k in range(8)))
    for lab in VAISSEAUX:
        tot = sum(cm[lab])
        print(f"    vérité {lab} {NOMS[lab]:<14}" + ("".join(f"{100 * x / tot:7.1f}" for x in cm[lab]) if tot
                                                    else "   aucun voxel dans la vérité"))
    print("  Sur quoi tombe la PRÉDICTION (ligne) dans la vérité (colonnes = étiquette de la vérité) :")
    for lab in VAISSEAUX:
        col = [cm[v][lab] for v in range(8)]
        tot = sum(col)
        print(f"    prédit {lab} {NOMS[lab]:<14}" + ("".join(f"{100 * x / tot:7.1f}" for x in col) if tot
                                                    else "   aucun voxel prédit"))

    print(f"\n=== {r['cas']} — 4. intensités (HU) ; « cœur » = voxels à plus de {COEUR_MM:g} mm du bord ===")
    print(f"    {'région':<34}{'médiane':>8}{'p5':>7}{'p95':>7}{'σ':>7}{'méd. cœur':>10}{'σ cœur':>8}{'vox. cœur':>11}")
    for nom, h in r["intensites"].items():
        if h is None:
            print(f"    {nom:<34}région vide")
            continue
        print(f"    {nom:<34}{h['mediane']:>8}{h['p5']:>7}{h['p95']:>7}{h['ecart_type']:>7}"
              f"{txt(h['coeur_mediane']):>10}{txt(h['coeur_ecart_type']):>8}{h['coeur_voxels']:>11}")


# Lignes du tableau 2 : libellé, fonction qui extrait la valeur d'une mesure (vérité ou prédiction).
LIGNES_STRUCTURE = (
    ("voxels", lambda m: m["voxels"]),
    ("volume (mL)", lambda m: m["volume_ml"]),
    ("coupes basse - haute", lambda m: f"{m['coupe_basse']} - {m['coupe_haute']}"),
    ("niveaux bas / haut (mm)", lambda m: f"{m['niveau_bas_mm']} / {m['niveau_haut_mm']}"),
    ("hauteur (mm)", lambda m: m["hauteur_mm"]),
    ("boîte englobante, indices [D-G, A-P, bas-haut]", lambda m: m["boite_indices"]),
    ("boîte englobante, taille (mm)", lambda m: m["boite_mm"]),
    ("coupes jusqu'au bord bas / haut", lambda m: f"{m['bords_coupes']['bas']} / {m['bords_coupes']['haut']}"),
    ("mm jusqu'au bord bas / haut", lambda m: f"{m['bord_bas_mm']} / {m['bord_haut_mm']}"),
    ("coupes jusqu'au bord D / G / avant / arrière",
     lambda m: "{droite} / {gauche} / {avant} / {arriere}".format(**m["bords_coupes"])),
    ("centre de gravité (mm, RAS)", lambda m: m["centre_mm"]),
    ("morceaux [3 plus gros, mL]", lambda m: f"{m['morceaux']} {m['morceaux_3_plus_gros_ml']}"))


def afficher_champs(champs, etudies):
    """Champ de vue des 15 cas d'entraînement (en-têtes seuls) : 1019 est-il à part ?"""
    import numpy as np
    print("\n=== Champ de vue des 15 cas d'entraînement (en-têtes seuls, aucune donnée lue) ===")
    print(f"    {'cas':<12}{'forme':<18}{'espacement (mm)':<26}{'étendue (mm)':<26}orientation")
    for cas, l in champs.items():
        marque = "  <== analysé" if cas in etudies else ""
        print(f"    {cas:<12}{str(l['forme']):<18}{str(l['espacement_mm']):<26}{str(l['etendue_mm']):<26}"
              f"{l['orientation']}{marque}")
    hauteurs = np.array([l["etendue_mm"][2] for l in champs.values()])
    print(f"    étendue le long du 3e axe (axe des coupes) : min {hauteurs.min():.0f}, médiane {np.median(hauteurs):.0f}, "
          f"max {hauteurs.max():.0f} mm")


# --- 6. Récapitulatif par hypothèse (le script ne conclut pas) --------------------------------
def indicateurs(r):
    """Indicateurs d'un cas, rangés par hypothèse."""
    S, I, cm = r["structures"], r["intensites"], r["confusion_verite_vers_prediction"]
    ao, ap = S["aorte_ascendante"]["verite"], S["artere_pulmonaire"]["verite"]
    og, od, coeur = S["oreillette_gauche"]["verite"], S["oreillette_droite"]["verite"], S["coeur_entier"]["verite"]

    def pct(lig, cols):                       # % de la vérité « lig » prédite dans « cols »
        tot = sum(cm[lig])
        return None if not tot else round(100 * sum(cm[lig][c] for c in cols) / tot, 1)

    def pct_pred(col, lignes):                # % de la prédiction « col » posée sur la vérité « lignes »
        tot = sum(cm[v][col] for v in range(8))
        return None if not tot else round(100 * sum(cm[v][col] for v in lignes) / tot, 1)

    def diff(x, y):
        return None if x is None or y is None else round(x - y, 1)

    lateraux = None if not ao["voxels"] else min(ao["bords_coupes"][k] for k in ("droite", "gauche", "avant", "arriere"))
    hauts = [x for x in (val(og, "niveau_haut_mm"), val(od, "niveau_haut_mm")) if x is not None]
    bas = [x for x in (val(og, "niveau_bas_mm"), val(od, "niveau_bas_mm")) if x is not None]
    vg, myo = I.get("verite_3_cavite_VG") or {}, I.get("verite_1_myocarde_VG") or {}
    bruit = vg.get("coeur_ecart_type")
    cnr = (None if vg.get("coeur_mediane") is None or myo.get("mediane") is None or not bruit
           else round((vg["coeur_mediane"] - myo["mediane"]) / bruit, 1))
    return {
        "a": {"etendue_haut_bas_mm": r["geometrie"]["etendue_mm_RL_AP_IS"][2],
              "aorte_verite_coupes_au_bord_haut": val(ao, "bords_coupes", "haut"),
              "aorte_verite_coupes_au_bord_bas": val(ao, "bords_coupes", "bas"),
              "aorte_verite_coupes_au_bord_lateral_le_plus_proche": lateraux,
              "ap_verite_coupes_au_bord_haut": val(ap, "bords_coupes", "haut"),
              "coeur_mm_au_bord_haut": val(coeur, "bord_haut_mm"),
              "coeur_mm_au_bord_bas": val(coeur, "bord_bas_mm")},
        "b": {"hu_aorte_verite": val(I, "verite_6_aorte_ascendante", "mediane"),
              "hu_cavite_VG": vg.get("mediane"),
              "hu_aorte_moins_VG": diff(val(I, "verite_6_aorte_ascendante", "mediane"), vg.get("mediane")),
              "hu_AP_moins_VD": diff(val(I, "verite_7_artere_pulmonaire", "mediane"),
                                     val(I, "verite_5_ventricule_droit", "mediane")),
              "bruit_sigma_coeur_VG": bruit, "cnr_VG_myocarde": cnr},
        "c": {"pct_aorte_verite_predite_fond": pct(6, [0]), "pct_aorte_verite_predite_AP": pct(6, [7]),
              "pct_aorte_verite_predite_cavites_myocarde": pct(6, [1, 2, 3, 4, 5]),
              "pct_aorte_verite_predite_aorte": pct(6, [6]),
              "pct_AP_verite_predite_aorte": pct(7, [6]), "pct_AP_verite_predite_fond": pct(7, [0]),
              "pct_aorte_predite_sur_verite_aorte": pct_pred(6, [6]),
              "pct_aorte_predite_sur_verite_AP": pct_pred(6, [7]),
              "pct_aorte_predite_sur_fond_verite": pct_pred(6, [0]),
              "volume_aorte_verite_ml": val(ao, "volume_ml"), "volume_AP_verite_ml": val(ap, "volume_ml"),
              "morceaux_aorte_verite": val(ao, "morceaux"),
              "aorte_moins_OG_avant_arriere_mm": diff(val(ao, "centre_mm", 1), val(og, "centre_mm", 1)),
              "aorte_moins_AP_droite_gauche_mm": diff(val(ao, "centre_mm", 0), val(ap, "centre_mm", 0)),
              "AP_moins_aorte_avant_arriere_mm": diff(val(ap, "centre_mm", 1), val(ao, "centre_mm", 1)),
              "haut_aorte_moins_haut_oreillettes_mm": diff(val(ao, "niveau_haut_mm"), max(hauts, default=None)),
              "bas_aorte_moins_bas_oreillettes_mm": diff(val(ao, "niveau_bas_mm"), min(bas, default=None))}}


# Pour chaque indicateur : libellé, et ce qu'une valeur inhabituelle suggérerait (sans conclure).
LECTURE = {
    "a": ("(a) CADRAGE DU SCANNER", {
        "etendue_haut_bas_mm": ("étendue du champ, bas-haut (mm)", "très différente des cas réussis : champ inhabituel"),
        "aorte_verite_coupes_au_bord_haut": ("vérité aorte : coupes jusqu'au bord haut",
                                             f"≤ {BORD_COUPES} : l'aorte est coupée par le haut du champ"),
        "aorte_verite_coupes_au_bord_bas": ("vérité aorte : coupes jusqu'au bord bas",
                                            f"≤ {BORD_COUPES} : l'aorte est coupée par le bas du champ"),
        "aorte_verite_coupes_au_bord_lateral_le_plus_proche": ("vérité aorte : coupes jusqu'au bord latéral",
                                                               f"≤ {BORD_COUPES} : l'aorte est coupée sur le côté"),
        "ap_verite_coupes_au_bord_haut": ("vérité AP : coupes jusqu'au bord haut", f"≤ {BORD_COUPES} : l'AP est coupée par le haut"),
        "coeur_mm_au_bord_haut": ("cœur entier (vérité) : mm jusqu'au bord haut", "petit : peu de marge au-dessus du cœur"),
        "coeur_mm_au_bord_bas": ("cœur entier (vérité) : mm jusqu'au bord bas", "petit : peu de marge sous le cœur")}),
    "b": ("(b) QUALITÉ DE L'IMAGE", {
        "hu_aorte_verite": ("HU médiane, vérité aorte", "basse (moins de 150 environ) : peu de contraste dans ce qui est annoté"),
        "hu_cavite_VG": ("HU médiane, cavité VG", "basse : injection globalement faible"),
        "hu_aorte_moins_VG": ("HU aorte − HU cavité VG", "loin de 0 : l'aorte annotée ne contient pas le même sang que le VG "
                              "(attendu : proche de 0, même sang artériel) ; voir aussi (c)"),
        "hu_AP_moins_VD": ("HU AP − HU VD", "loin de 0 : l'AP annotée ne contient pas le même sang que le VD"),
        "bruit_sigma_coeur_VG": ("bruit : σ des HU au cœur du VG", "nettement plus grand que sur les cas réussis : image bruitée"),
        "cnr_VG_myocarde": ("CNR = (VG − myocarde) / bruit", "nettement plus petit que sur les cas réussis : bords peu visibles")}),
    "c": ("(c) ANNOTATION (ou confusion du modèle)", {
        "pct_aorte_verite_predite_fond": ("% de la vérité aorte prédite « fond »",
                                          "grand : le modèle ne voit pas d'aorte là où elle est annotée (structure non "
                                          "apprise, ex. aorte descendante ou crosse ; ou étiquette posée sur autre chose)"),
        "pct_aorte_verite_predite_AP": ("% de la vérité aorte prédite « AP »", "grand : confusion d'étiquettes aorte / AP"),
        "pct_aorte_verite_predite_cavites_myocarde": ("% de la vérité aorte prédite cavité ou myocarde",
                                                      "grand : limite aorte / cœur placée ailleurs"),
        "pct_aorte_verite_predite_aorte": ("% de la vérité aorte prédite « aorte »", "proche de 0 : aucune superposition (Dice 0)"),
        "pct_AP_verite_predite_aorte": ("% de la vérité AP prédite « aorte »", "grand, avec « aorte prédite AP » : permutation 6/7"),
        "pct_AP_verite_predite_fond": ("% de la vérité AP prédite « fond »", "grand : AP annotée plus loin que ce que le modèle a appris"),
        "pct_aorte_predite_sur_verite_aorte": ("% de l'aorte prédite posée sur la vérité aorte",
                                               "proche de 0 : le modèle met son « aorte » ailleurs"),
        "pct_aorte_predite_sur_verite_AP": ("% de l'aorte prédite posée sur la vérité AP",
                                            "grand : le modèle appelle « aorte » ce qui est annoté AP"),
        "pct_aorte_predite_sur_fond_verite": ("% de l'aorte prédite posée sur le fond de la vérité",
                                              "grand : le modèle voit une aorte là où rien n'est annoté"),
        "volume_aorte_verite_ml": ("volume vérité aorte (mL)", "très différent des cas réussis : définition différente ?"),
        "volume_AP_verite_ml": ("volume vérité AP (mL)", "très différent des cas réussis : définition différente ?"),
        "morceaux_aorte_verite": ("morceaux de la vérité aorte", "2 gros morceaux ou plus : par ex. ascendante + descendante"),
        "aorte_moins_OG_avant_arriere_mm": ("centre aorte − centre OG, avant-arrière (mm, + = devant)",
                                            "négatif : aorte DERRIÈRE l'OG, place de l'aorte descendante (l'ascendante est devant)"),
        "aorte_moins_AP_droite_gauche_mm": ("centre aorte − centre AP, droite-gauche (mm, + = aorte à droite)",
                                            "négatif : aorte à gauche de l'AP, inverse de l'anatomie habituelle : permutation 6/7 ?"),
        "AP_moins_aorte_avant_arriere_mm": ("centre AP − centre aorte, avant-arrière (mm, + = AP devant)",
                                            "négatif : AP derrière l'aorte, inverse de l'anatomie habituelle : permutation 6/7 ?"),
        "haut_aorte_moins_haut_oreillettes_mm": ("haut de l'aorte − haut des oreillettes (mm)",
                                                 "grand et positif : aorte annotée au-delà de la définition MM-WHS "
                                                 "(de la valve au bord supérieur des oreillettes)"),
        "bas_aorte_moins_bas_oreillettes_mm": ("bas de l'aorte − bas des oreillettes (mm)",
                                               "très négatif : l'aorte descend sous les oreillettes (aorte descendante incluse ?)")})}


def recapitulatif(resultats):
    tous = [(r["cas"], indicateurs(r)) for r in resultats]
    print("\n" + "=" * 110)
    print(f"6. RÉCAPITULATIF — {tous[0][0]} (cas étudié) comparé à {', '.join(c for c, _ in tous[1:]) or 'aucun cas'}")
    print("Le script NE CONCLUT PAS. Pour chaque indicateur : les valeurs de chaque cas, puis ce qu'une valeur "
          "inhabituelle suggérerait. Croiser avec les figures avant de décider.")
    for h, (titre, libelles) in LECTURE.items():
        print(f"\n{titre}")
        for cle, (libelle, si) in libelles.items():
            valeurs = " | ".join(f"{c[-4:]} : {txt(ind[h][cle])}" for c, ind in tous)
            print(f"  - {libelle:<66}{valeurs}")
            print(f"      si {si}")
    print("\nPoints de repère pour lire ce tableau :")
    print("  - les cavités et le myocarde de 1019 sont bien segmentés (Dice 0,89 à 0,96) : un problème d'IMAGE global (b) devrait "
          "aussi les toucher ;")
    print("  - vérité aorte qui touche le bord du champ sur 1019 mais pas sur les cas réussis : regarder (a) ;")
    print("  - vérité aorte surtout prédite « fond », derrière l'OG ou sous les oreillettes : regarder (c) ;")
    print("  - vérité aorte prédite « AP » et vérité AP prédite « aorte », positions inversées : regarder (c), permutation ;")
    print("  - aucun seuil n'est validé : ce sont des repères. La décision se prend sur les figures, puis se trace "
          "dans le journal (suite de D26).")
    return {c: ind for c, ind in tous}


# --- Programme principal ----------------------------------------------------------------------
def main():
    a = lire_arguments()
    ds, sortie, chemins, entr = verifier(a)
    champs = champs_entrainement(ds, entr)
    os.makedirs(sortie, exist_ok=True)                      # le seul dossier où le script écrit
    resultats, figures = [], {}
    for n, c in chemins.items():
        print(f"\n{'#' * 110}\n# mmwhs_{n} : image et vérité de imagesTr / labelsTr, prédiction de "
              f"fold_{a.pli}/validation\n{'#' * 110}", flush=True)
        r, tableaux = analyser(n, c, a.pli)
        afficher_cas(r)
        figures[r["cas"]] = os.path.join(sortie, NOM_FIGURE.format(n=n))
        figure(n, r, tableaux, figures[r["cas"]])
        print(f"\nFigure : {figures[r['cas']]}", flush=True)
        del tableaux                                        # un seul cas en mémoire à la fois
        resultats.append(r)
    afficher_champs(champs, {r["cas"] for r in resultats})
    ind = recapitulatif(resultats)

    synthese = {"objet": "analyse de l'échec du pli 0 sur les vaisseaux de mmwhs_1019 (D26), LECTURE SEULE",
                "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "pli": a.pli, "trainer": TRAINER, "configuration": CONFIGURATION,
                "split_sha256": prep.sha256_fichier(a.split), "code_commit": prep.commit_git(),
                "sources": chemins, "champs_entrainement": champs, "cas": resultats, "indicateurs": ind,
                "figures": figures,
                "lecture": {h: {k: {"libelle": l, "si": s} for k, (l, s) in d.items()} for h, (_, d) in LECTURE.items()}}
    chemin = os.path.join(sortie, NOM_SYNTHESE)
    with open(chemin + ".tmp", "w", encoding="utf-8") as f:
        json.dump(synthese, f, indent=2, ensure_ascii=False)
    os.replace(chemin + ".tmp", chemin)                     # jamais de fichier à moitié écrit
    print(f"\nSynthèse : {chemin}")
    print("Rien n'a été écrit dans nnUNet_raw, nnUNet_preprocessed ni nnUNet_results ; aucun cas de test lu.")


if __name__ == "__main__":
    main()
