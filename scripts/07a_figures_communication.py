"""
Étape 7 — Figures de communication pour le cas de test 1010 (décision D48). AFFICHAGE SEULEMENT.

Ce que fait le script :
  1. vérifie les garde-fous : cas demandé = 1010 (seul cas couvert par D48), dossier de sortie hors du dépôt
     et hors des dossiers de données lus (nnunet_raw, etape6, predictions) ;
  2. lit avec nibabel, en lecture seule, 4 fichiers NIfTI du cas 1010 : le scanner (imagesTs), la vérité
     terrain (labelsTs), la prédiction A de la phase B (nnU-Net, ensemble des 5 plis + post-traitement D33) et
     la prédiction T (TotalSegmentator, refaite selon D44) ;
  3. contrôle chaque fichier : même forme et même affine que le scanner (sinon ARRÊT, aucun rééchantillonnage),
     étiquettes entières de 0 à 7 (sinon ARRÊT) ; puis réoriente tout en orientation canonique RAS ;
  4. dessine 3 coupes (axiale, coronale, sagittale) passant par le centre du cœur (barycentre de la vérité),
     avec, côte à côte : vérité terrain, A et T ; mêmes couleurs par structure ; légende ;
  5. écrit la figure en PNG et en PDF dans --sortie, sans écraser un fichier existant (sauf --ecraser).

Ce que le script NE fait PAS (D48) :
  - aucune métrique (ni Dice, ni distance), aucun volume, aucun chiffre calculé ni affiché sur la figure ;
  - aucun fichier de résultats de l'étape 6 n'est relu ni réécrit (résultats, registre, manifeste, verrou) ;
    les prédictions A et T sont seulement lues ; etape6_commun.py n'est pas importé : son garde-fou
    (verifier_chemin) bloquerait la lecture de labelsTs (code 18), et D48 dit de ne ni l'appeler ni le contourner. Ce script est donc autonome.

Rappel D48 : le protocole figé (docs/protocole_etape6.md, R-O1, § 11) interdit de lire labelsTs hors du script
d'évaluation. L'exception D48 permet, APRÈS la clôture de l'étape 6, de lire la vérité du SEUL cas 1010, pour
affichage uniquement.

LICENCE : les images viennent de MM-WHS. Figures NON DIFFUSABLES avant vérification de la licence MM-WHS.
Elles restent dans Drive (poc_ct/communication/), JAMAIS dans le dépôt.

Entrées (paramètres) :
  --donnees  racine des données dans Drive (défaut /content/drive/MyDrive/poc_ct)
  --sortie   dossier des figures (défaut /content/drive/MyDrive/poc_ct/communication)
  --cas      cas à afficher (défaut et seule valeur autorisée : 1010)
  --ecraser  autorise le remplacement d'une figure existante

Sorties (dans --sortie uniquement) :
  figure_communication_mmwhs_1010.png  (3 lignes × 3 colonnes, 200 points par pouce)
  figure_communication_mmwhs_1010.pdf  (même figure, format vectoriel pour un document)
"""
import argparse, io, os, sys

# --- Constantes -----------------------------------------------------------------------------------
DEPOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DRIVE = "/content/drive/MyDrive/poc_ct"                  # comme scripts/05a_preparer_nnunet.py (DRIVE)
CAS_AUTORISES = ("1010",)                                 # D48 : le seul cas couvert par l'exception
NOM_DS = "Dataset101_MMWHS"                               # scripts/06c_predire_test.py (NOM_DS)

# Chemins relatifs à --donnees. Recopiés depuis le dépôt (etape6_commun.py n'est PAS importé, voir en tête) :
CHEMINS = {
    # scripts/06d_evaluer_test.py, fonction image() : os.path.join(a.raw, DS, "imagesTs", f"mmwhs_{c}_0000.nii.gz")
    "scanner": ("nnunet_raw", NOM_DS, "imagesTs", "mmwhs_{cas}_0000.nii.gz"),
    # scripts/06d_evaluer_test.py, fonction calculer() : os.path.join(a.raw, DS, "labelsTs", f"mmwhs_{c}.nii.gz")
    "verite": ("nnunet_raw", NOM_DS, "labelsTs", "mmwhs_{cas}.nii.gz"),
    # scripts/06c_predire_test.py (sous-commande nnunet) : os.path.join(a.sortie, "predictions", "A"), avec
    # --sortie = etape6_commun.DOSSIERS_PHASES["B"] = <DRIVE>/etape6/phaseB (carnet, section 28 a, SORTIE)
    "A": ("etape6", "phaseB", "predictions", "A", "mmwhs_{cas}.nii.gz"),
    # scripts/etape6_commun.py : PREDICTIONS_TOTALSEG = f"{DRIVE}/predictions/totalsegmentator_d44" (D44),
    # NOM_PREDICTION = "mmwhs_{cas}.nii.gz"
    "T": ("predictions", "totalsegmentator_d44", "mmwhs_{cas}.nii.gz"),
}
# Dossiers lus : une sortie à l'intérieur est refusée (rien n'y est écrit, en particulier dans etape6).
DOSSIERS_LUS = ("nnunet_raw", "etape6", "predictions")

# Tolérance sur l'affine : celle de la règle R0 (scripts/controle_plausibilite.py, TOLERANCE_R0_MM = 0.01).
TOLERANCE_MM = 0.01

# Étiquettes MM-WHS renumérotées (scripts/03_preparer.py, CORRESPONDANCE ; noms du rapport v1, § 2,
# et du protocole § 3.2).
STRUCTURES = {1: "myocarde VG", 2: "oreillette gauche (OG)", 3: "cavité VG", 4: "oreillette droite (OD)",
              5: "ventricule droit (VD)", 6: "aorte ascendante", 7: "artère pulmonaire (AP)"}
# Correspondance TotalSegmentator → MM-WHS : docs/correspondance_etiquettes.md, « Règle : l'étiquette k de
# TotalSegmentator correspond à notre étiquette k, pour k = 1 à 7. Aucune renumérotation n'est nécessaire. »
# (cohérent avec scripts/04_totalsegmentator.py, CORRESPONDANCE_MMWHS, composé avec 03_preparer.CORRESPONDANCE).
CORRESPONDANCE_T = {k: k for k in range(1, 8)}
# T n'est comparé que sur les structures 1 à 5 (D23 ; scripts/etape6_metriques.py, ETIQUETTES_C) : aorte et AP
# sont définies autrement par TotalSegmentator ; elles ne sont pas affichées pour T.
ETIQUETTES_T_AFFICHEES = (1, 2, 3, 4, 5)

# Palette d'Okabe et Ito : 7 couleurs distinguables par la plupart des personnes daltoniennes. Les structures
# voisines (myocarde / cavité VG, OD / VD, aorte / AP) reçoivent des teintes éloignées.
COULEURS = {1: "#E69F00", 2: "#56B4E9", 3: "#CC79A7", 4: "#009E73", 5: "#0072B2", 6: "#D55E00", 7: "#F0E442"}
OPACITE = 0.6

# Fenêtre d'affichage CT : niveau 200 HU, largeur 800 HU (de -200 à +600 HU). Fenêtre usuelle des scanners
# cardiaques injectés : le sang opacifié (environ 190 à 560 HU dans ce projet : D27, D46) reste distinct du myocarde
# (environ 100 HU) au lieu d'être saturé en blanc, ce que ferait la fenêtre « tissus mous » 40 / 400.
NIVEAU_HU, LARGEUR_HU = 200, 800
MARGE_MM = 20                                             # marge autour du cœur pour recadrer les coupes

NOM_FIGURE = "figure_communication_mmwhs_{cas}"


def arret(message, code=1):
    print(f"ARRÊT : {message}", flush=True)
    sys.exit(code)


def dans(chemin, dossier):
    """Vrai si chemin est dossier ou l'un de ses sous-dossiers (même logique que 05a_preparer_nnunet.dans)."""
    chemin, dossier = os.path.abspath(chemin), os.path.abspath(dossier)
    try:
        return os.path.commonpath([chemin, dossier]) == dossier
    except ValueError:                                    # lecteurs différents sous Windows
        return False


def lire_arguments(argv=None):
    p = argparse.ArgumentParser(description="Figures de communication du cas de test 1010 (D48) : vérité, A et T "
                                            "côte à côte. Affichage seulement, aucun calcul de métrique.")
    p.add_argument("--donnees", default=DRIVE, help=f"racine des données dans Drive (défaut {DRIVE})")
    p.add_argument("--sortie", default=f"{DRIVE}/communication", help="dossier des figures, hors du dépôt")
    p.add_argument("--cas", default="1010", help="cas à afficher ; D48 n'autorise que 1010")
    p.add_argument("--ecraser", action="store_true", help="remplacer une figure existante")
    a = p.parse_args(argv)
    a.donnees, a.sortie = os.path.abspath(a.donnees), os.path.abspath(a.sortie)
    # Garde-fous AVANT toute lecture.
    if a.cas not in CAS_AUTORISES:
        arret(f"cas {a.cas!r} refusé. L'exception D48 ne couvre que le cas 1010 : la vérité terrain des autres cas "
              "de test (labelsTs) n'est pas lue. Rien n'a été lu ni écrit.")
    if dans(a.sortie, DEPOT):
        arret(f"--sortie ({a.sortie}) est dans le dépôt : les figures (images MM-WHS) n'y vont jamais. "
              "Rien n'a été lu ni écrit.")
    for d in DOSSIERS_LUS:
        if dans(a.sortie, os.path.join(a.donnees, d)):
            arret(f"--sortie ({a.sortie}) est dans un dossier de données lu ({d}) : refusé, rien de l'étape 6 ni "
                  "des données n'est réécrit. Rien n'a été lu ni écrit.")
    return a


def charger(chemins):
    """Lit les 4 fichiers (nibabel, lecture seule), contrôle la géométrie et les étiquettes, réoriente en RAS."""
    import numpy as np
    import nibabel as nib
    images = {nom: nib.load(ch) for nom, ch in chemins.items()}
    ref = images["scanner"]
    print(f"Scanner : forme {tuple(ref.shape)}, espacement {tuple(round(float(z), 4) for z in ref.header.get_zooms()[:3])} mm, "
          f"orientation d'origine {''.join(nib.aff2axcodes(ref.affine))}")
    for nom in ("verite", "A", "T"):
        img = images[nom]
        if tuple(img.shape) != tuple(ref.shape):
            arret(f"{nom} : forme {tuple(img.shape)} différente du scanner {tuple(ref.shape)}. Aucun rééchantillonnage.")
        if not np.allclose(img.affine, ref.affine, rtol=0, atol=TOLERANCE_MM):
            arret(f"{nom} : affine différente du scanner (tolérance {TOLERANCE_MM} mm). Aucun rééchantillonnage.")
    # Réorientation canonique RAS (axe 1 vers la droite du patient, axe 2 vers l'avant, axe 3 vers le haut).
    canon = {nom: nib.as_closest_canonical(img) for nom, img in images.items()}
    ras = canon["scanner"].affine
    for nom in ("verite", "A", "T"):
        if canon[nom].shape != canon["scanner"].shape or not np.allclose(canon[nom].affine, ras, rtol=0, atol=TOLERANCE_MM):
            arret(f"{nom} : géométrie différente du scanner après réorientation RAS.")
    donnees = {"scanner": np.asanyarray(canon["scanner"].dataobj, dtype=np.float32)}
    for nom in ("verite", "A", "T"):
        m = np.asanyarray(canon[nom].dataobj)
        valeurs = np.unique(m)
        if not np.array_equal(valeurs, np.round(valeurs)) or valeurs.min() < 0 or valeurs.max() > 7:
            arret(f"{nom} : valeurs hors des étiquettes entières 0 à 7 : rien n'est corrigé, à examiner.")
        donnees[nom] = m.astype(np.uint8)
    # T : passage dans l'espace MM-WHS (identité, voir CORRESPONDANCE_T), puis structures 1 à 5 seulement (D23).
    t = np.zeros_like(donnees["T"])
    for k_ts, k_mmwhs in CORRESPONDANCE_T.items():
        if k_mmwhs in ETIQUETTES_T_AFFICHEES:
            t[donnees["T"] == k_ts] = k_mmwhs
    donnees["T"] = t
    if not (donnees["verite"] > 0).any():
        arret("vérité terrain vide : impossible de centrer les coupes. Rien n'est écrit.")
    espacement = np.sqrt((ras[:3, :3] ** 2).sum(axis=0))   # mm par voxel, dans l'ordre RAS
    return donnees, espacement


def coupes(vol, centre, boite):
    """3 coupes 2D en convention radiologique, recadrées sur la boîte (min, max) par axe.
    Axiale : droite du patient à gauche de l'image, avant en haut. Coronale : droite du patient à gauche, haut en
    haut. Sagittale : avant à gauche, haut en haut. En RAS, il suffit de transposer puis de retourner les 2 axes."""
    (x0, x1), (y0, y1), (z0, z1) = boite
    i, j, k = centre
    return {"axiale": vol[x0:x1, y0:y1, k].T[::-1, ::-1],
            "coronale": vol[x0:x1, j, z0:z1].T[::-1, ::-1],
            "sagittale": vol[i, y0:y1, z0:z1].T[::-1, ::-1]}


def dessiner(donnees, espacement, cas):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, to_rgba
    from matplotlib.patches import Patch

    # Centre du cœur : barycentre des voxels de la vérité (position seulement, aucun chiffre affiché).
    voxels = np.argwhere(donnees["verite"] > 0)
    centre = tuple(int(round(c)) for c in voxels.mean(axis=0))
    marge = np.ceil(MARGE_MM / espacement).astype(int)
    forme = donnees["verite"].shape
    boite = [(max(0, int(voxels[:, a].min()) - marge[a]), min(forme[a], int(voxels[:, a].max()) + 1 + marge[a]))
             for a in range(3)]
    # Rapport d'aspect = hauteur d'un pixel / largeur d'un pixel (en mm), pour ne pas déformer l'anatomie.
    sx, sy, sz = espacement
    aspects = {"axiale": sy / sx, "coronale": sz / sx, "sagittale": sz / sy}
    reperes = {"axiale": ("D", "G", "A", "P"), "coronale": ("D", "G", "S", "I"), "sagittale": ("A", "P", "S", "I")}

    couleurs = [(0, 0, 0, 0)] + [to_rgba(COULEURS[k], OPACITE) for k in range(1, 8)]
    cmap_masque = ListedColormap(couleurs)
    colonnes = [("verite", "Vérité terrain (annotation MM-WHS)"),
                ("A", "A : nnU-Net\n(ensemble 5 plis + post-traitement)"),
                ("T", "T : TotalSegmentator\n(structures 1 à 5 seulement)")]
    scanner = coupes(donnees["scanner"], centre, boite)
    vmin, vmax = NIVEAU_HU - LARGEUR_HU / 2, NIVEAU_HU + LARGEUR_HU / 2

    fig, axes = plt.subplots(3, 3, figsize=(12, 13.5), facecolor="black")
    for ligne, plan in enumerate(("axiale", "coronale", "sagittale")):
        for col, (nom, titre) in enumerate(colonnes):
            ax = axes[ligne, col]
            masque = coupes(donnees[nom], centre, boite)[plan]
            ax.imshow(scanner[plan], cmap="gray", vmin=vmin, vmax=vmax, aspect=aspects[plan], interpolation="nearest")
            ax.imshow(masque, cmap=cmap_masque, vmin=0, vmax=7, aspect=aspects[plan], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            gauche, droite, haut, bas = reperes[plan]
            style = dict(color="white", fontsize=10, fontweight="bold", transform=ax.transAxes)
            ax.text(0.02, 0.5, gauche, ha="left", va="center", **style)
            ax.text(0.98, 0.5, droite, ha="right", va="center", **style)
            ax.text(0.5, 0.98, haut, ha="center", va="top", **style)
            ax.text(0.5, 0.02, bas, ha="center", va="bottom", **style)
            if ligne == 0:
                ax.set_title(titre, color="white", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"Coupe {plan}", color="white", fontsize=11)

    legende = [Patch(facecolor=to_rgba(COULEURS[k], OPACITE), edgecolor="none", label=f"{k}  {STRUCTURES[k]}") for k in range(1, 8)]
    fig.legend(handles=legende, loc="lower center", ncol=4, frameon=False, fontsize=10, labelcolor="white",
               bbox_to_anchor=(0.5, 0.045))
    fig.suptitle(f"MM-WHS, cas de test {cas} : vérité terrain, A (nnU-Net) et T (TotalSegmentator)",
                 color="white", fontsize=13, y=0.985)
    fig.text(0.5, 0.012,
             "T : aorte et artère pulmonaire non affichées (définitions différentes de MM-WHS, D23). "
             "Repères : D/G droite/gauche du patient, A/P avant/arrière, S/I haut/bas (convention radiologique).\n"
             f"Cadrage sur la vérité terrain (+ {MARGE_MM} mm) : une détection de A ou de T loin du cœur peut ne pas apparaître.\n"
             f"Fenêtre CT : niveau {NIVEAU_HU} HU, largeur {LARGEUR_HU} HU. "
             "Figure non diffusable avant vérification de la licence MM-WHS.",
             ha="center", va="bottom", color="lightgray", fontsize=8.5)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.11, wspace=0.03, hspace=0.06)
    return fig


def ecrire(fig, chemin, format_, ecraser):
    """Écrit la figure sans métadonnées superflues (ni logiciel ni date) ; refuse d'écraser sans --ecraser."""
    meta = {"png": {"Software": None}, "pdf": {"Creator": None, "Producer": None, "CreationDate": None}}[format_]
    tampon = io.BytesIO()
    fig.savefig(tampon, format=format_, dpi=200, facecolor=fig.get_facecolor(), metadata=meta)
    with open(chemin, "wb" if ecraser else "xb") as f:      # "xb" : échoue si le fichier existe déjà
        f.write(tampon.getvalue())


def main(argv=None):
    a = lire_arguments(argv)
    chemins = {nom: os.path.join(a.donnees, *[p.format(cas=a.cas) for p in parties]) for nom, parties in CHEMINS.items()}
    manquants = [ch for ch in chemins.values() if not os.path.isfile(ch)]
    if manquants:
        arret("fichier(s) introuvable(s) : " + " ; ".join(manquants))
    sorties = [os.path.join(a.sortie, NOM_FIGURE.format(cas=a.cas) + ext) for ext in (".png", ".pdf")]
    existants = [s for s in sorties if os.path.exists(s)]
    if existants and not a.ecraser:
        arret("figure(s) déjà présente(s), non écrasée(s) (ajouter --ecraser pour les remplacer) : " + " ; ".join(existants))

    print(f"Cas {a.cas} (D48) : lecture pour AFFICHAGE seulement, aucune métrique ni volume calculé.")
    donnees, espacement = charger(chemins)
    print("Contrôles : formes et affines identiques au scanner, étiquettes entières 0 à 7 : conformes.")
    fig = dessiner(donnees, espacement, a.cas)
    os.makedirs(a.sortie, exist_ok=True)
    for s in sorties:
        ecrire(fig, s, s.rsplit(".", 1)[1], a.ecraser)

    print("\nFichiers lus (lecture seule) :")
    for nom, ch in chemins.items():
        print(f"  {nom:8s} {ch}")
    print("Fichiers écrits :")
    for s in sorties:
        print(f"  {s}")
    print("RAPPEL : figures non diffusables avant vérification de la licence MM-WHS ; jamais dans le dépôt.")


if __name__ == "__main__":
    main()
