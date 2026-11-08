"""
Avant l'étape 6 — Comparer les DÉFINITIONS des structures (TotalSegmentator contre MM-WHS)
sur DEUX cas d'ENTRAÎNEMENT. Le lot de test n'est jamais lu.

Pourquoi : les étiquettes 1 à 7 de TotalSegmentator et les nôtres portent les mêmes numéros, mais pas
forcément la même définition (ex. MM-WHS ne garde que l'aorte ascendante). Pour fixer les règles de
comparaison de l'étape 6 SANS regarder le test, on regarde les écarts sur deux cas d'entraînement, un de
chaque famille d'examens : 1003 (coupes de 0,45 mm) et 1002 (0,625 mm).

Ce que fait le script :
  1. garde-fous : les cas demandés sont en ENTRAÎNEMENT dans le découpage figé (splits/mmwhs_split.json,
     lu seulement) et absents du test ; seuls imagesTr et labelsTr sont lus ; --sortie est hors du dépôt,
     hors de tout dossier nnU-Net et ne contient pas les prédictions de test de l'étape 4 ;
  2. lance TotalSegmentator (mêmes réglages que le script 04 : tâche heartchambers_highres, licence lue
     dans TOTALSEG_LICENCE, statistiques d'usage coupées, GPU obligatoire, 1 processus enfant).
     Si les prédictions existent déjà (avec leur traçabilité), elles sont RÉUTILISÉES sans relancer
     TotalSegmentator (--ecraser pour les refaire) : on peut ainsi refaire les figures en quelques minutes ;
  3. contrôle que chaque prédiction a la même grille de voxels que le scanner et que la vérité terrain ;
  4. pour chaque cas et chaque structure 1 à 7 : volume (mL), niveaux le plus bas et le plus haut (mm),
     nombre de morceaux, et une figure (contours superposés sur le scanner + projections des masques).
  Le script ne calcule AUCUN score (ni Dice, ni HD95) : ce n'est pas l'étape 6.

Entrées :
  --dataset   dossier Dataset101_MMWHS dans Drive (seuls imagesTr/ et labelsTr/ sont lus)
  --sortie    dossier de sortie dans Drive, HORS de nnunet_raw (règle 5 : ces masques ne doivent jamais
              pouvoir servir à l'entraînement). Ex. /content/drive/MyDrive/poc_ct/predictions/totalsegmentator_definitions
  --cas       numéros des cas d'entraînement (défaut : 1003 1002)
  --split, --tache, --appareil, --processus, --ecraser, --statistiques-usage : comme le script 04
  Variable d'environnement TOTALSEG_LICENCE : clé de licence (jamais affichée ni écrite).
Sorties (uniquement dans --sortie) :
  mmwhs_XXXX_totalsegmentator.nii.gz (prédiction ; nom volontairement différent d'un masque nnU-Net) ;
  tracabilite_definitions.json ; chiffres_definitions.csv ; definitions_mmwhs_XXXX.png (une figure par cas)
Codes de sortie :
  0 = figures et chiffres écrits ; 1 = garde-fou (rien n'est écrit) ;
  2 = prédictions partielles déjà présentes sans traçabilité : relancer avec --ecraser ;
  3 = erreur pendant la prédiction ; 4 = licence absente ou refusée ; 5 = pas de GPU ;
  6 = géométrie différente entre scanner, vérité terrain et prédiction : la figure du cas fautif et le
      fichier CSV ne sont pas faits (la figure d'un cas traité avant peut déjà exister).

Vocabulaire :
  - définition d'une structure : où l'annotateur s'arrête (ex. jusqu'où on suit l'aorte vers le haut et
    vers le bas). Deux définitions différentes font baisser un score sans que le modèle se trompe ;
  - contour : la ligne qui entoure un masque sur une coupe ;
  - projection (d'intensité maximale) d'un masque : on « écrase » le masque 3D sur un plan, comme une ombre.
    Un point du plan est coloré si la structure est présente n'importe où sur la ligne de vue. Elle montre
    toute l'étendue de la structure en une seule image ;
  - niveau (mm) : hauteur d'une coupe axiale dans le repère du scanner (plus grand = plus vers la tête).
    La valeur absolue dépend du repère de chaque scanner : seuls les ÉCARTS TotalSegmentator − MM-WHS,
    calculés dans le même repère pour les deux masques, sont à interpréter ;
  - morceaux (composantes connexes) : nombre de blocs séparés d'un masque (deux voxels qui se touchent
    par une face, une arête ou un coin sont dans le même bloc). Une aorte en 2 morceaux peut
    signaler, par exemple, l'aorte ascendante et une partie de l'aorte descendante ;
  - processus enfant, mode spawn : voir le script 04 (c'est pourquoi tout est rangé dans main()).
"""
# Au niveau du module : seulement des imports légers, des constantes et des fonctions.
import argparse, csv, datetime, importlib, json, math, os, re, sys, time

# Fonctions communes reprises du script 04 (même dossier) plutôt que réécrites. Importer 04 ne lance
# rien : son traitement est protégé par if __name__ == "__main__".
etape4 = importlib.import_module("04_totalsegmentator")
arret, sans_cle, sha256_fichier = etape4.arret, etape4.sans_cle, etape4.sha256_fichier
DEPOT, DOSSIERS_INTERDITS = etape4.DEPOT, etape4.DOSSIERS_INTERDITS

# --- Paramètres fixes -------------------------------------------------------------------------
# Étiquette → (notre nom dans dataset.json, nom TotalSegmentator 2.18.0). Voir docs/correspondance_etiquettes.md.
STRUCTURES = {1: ("myocarde_VG", "heart_myocardium"), 2: ("oreillette_gauche", "heart_atrium_left"),
              3: ("cavite_VG", "heart_ventricle_left"), 4: ("oreillette_droite", "heart_atrium_right"),
              5: ("ventricule_droit", "heart_ventricle_right"), 6: ("aorte_ascendante", "aorta"),
              7: ("artere_pulmonaire", "pulmonary_artery")}
NOM_PRED = "mmwhs_{n}_totalsegmentator.nii.gz"
FICHIER_TRACE, FICHIER_CSV, NOM_FIGURE = "tracabilite_definitions.json", "chiffres_definitions.csv", "definitions_mmwhs_{n}.png"
FENETRE = (-200, 800)      # fenêtre d'affichage en HU (comme la section 8 et le script 03)
MARGE_MM = 20              # marge autour de la structure dans les vues recadrées
BLEU, ORANGE, GRIS = "#33BBFF", "#FF8C00", "#E6E6E6"   # vérité MM-WHS, TotalSegmentator, les deux


def lire_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--sortie", required=True)
    p.add_argument("--cas", nargs="+", default=["1003", "1002"])
    p.add_argument("--split", default=os.path.join(DEPOT, "splits", "mmwhs_split.json"))
    p.add_argument("--tache", default="heartchambers_highres")
    p.add_argument("--appareil", default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--processus", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--ecraser", action="store_true")
    p.add_argument("--statistiques-usage", action="store_true")
    return p.parse_args()


def verifier(a):
    """Garde-fous. Renvoie (dataset, sortie, chemins par cas). Rien n'est écrit avant qu'ils soient passés."""
    dataset, sortie = os.path.abspath(a.dataset), os.path.abspath(a.sortie)
    if not re.match(r"^Dataset\d{3}_", os.path.basename(dataset)):
        arret(1, f"--dataset doit être le dossier Dataset1xx_... (reçu : {dataset}).")
    if os.path.commonpath([sortie, DEPOT]) == DEPOT:
        arret(1, "--sortie est dans le dépôt. Les prédictions doivent rester dans Google Drive.")
    interdits = [d for d in sortie.split(os.sep) if DOSSIERS_INTERDITS.match(d)]
    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        racine = os.environ.get(var)
        if racine and os.path.commonpath([sortie, os.path.abspath(racine)]) == os.path.abspath(racine):
            interdits.append(f"${var}")
    if interdits or os.path.commonpath([sortie, dataset]) == dataset:
        arret(1, f"--sortie ({sortie}) est dans un dossier de données d'entraînement ({interdits or dataset}). "
                 "Choisir par exemple /content/drive/MyDrive/poc_ct/predictions/totalsegmentator_definitions.")
    if os.path.exists(os.path.join(sortie, etape4.FICHIER_TRACE)):
        arret(1, f"{sortie} contient les prédictions de TEST de l'étape 4 : choisir un autre dossier.")

    with open(a.split, encoding="utf-8") as f:              # lecture seule du découpage figé
        split = json.load(f)
    test, entr = set(split["test_sous_cle"]), set(split["entrainement_validation_croisee"])
    chemins = {}
    for n in a.cas:
        nom = f"ct_train_{n}"
        if not re.fullmatch(r"\d{4}", n):
            arret(1, f"numéro de cas invalide : {n} (attendu : 4 chiffres, ex. 1003).")
        if nom in test:
            arret(1, f"{nom} fait partie du LOT DE TEST : refusé (le test ne sert qu'à l'étape 6, D18).")
        if nom not in entr:
            arret(1, f"{nom} n'est pas dans le lot d'entraînement du découpage figé.")
        c = {"image": os.path.join(dataset, "imagesTr", f"mmwhs_{n}_0000.nii.gz"),
             "verite": os.path.join(dataset, "labelsTr", f"mmwhs_{n}.nii.gz"),
             "pred": os.path.join(sortie, NOM_PRED.format(n=n))}
        manquants = [x for x in (c["image"], c["verite"]) if not os.path.exists(x)]
        if manquants:
            arret(1, f"fichiers introuvables (Drive relié ? section 1) : {manquants}")
        chemins[n] = c
    print(f"Cas vérifiés : {', '.join('mmwhs_' + n for n in a.cas)} sont en ENTRAÎNEMENT (aucun cas de test).")
    return dataset, sortie, chemins


def predictions_reutilisables(sortie, chemins):
    """True si toutes les prédictions et leur traçabilité existent, pour les MÊMES scanners (empreintes)."""
    trace = os.path.join(sortie, FICHIER_TRACE)
    if not (os.path.exists(trace) and all(os.path.exists(c["pred"]) for c in chemins.values())):
        return False
    with open(trace, encoding="utf-8") as f:
        faits = {r["cas"]: r["scanner_sha256"] for r in json.load(f)["cas"]}
    return all(faits.get(f"mmwhs_{n}") == sha256_fichier(c["image"]) for n, c in chemins.items())


def predire(a, cle, sortie, chemins):
    """Lance TotalSegmentator sur chaque cas, un à la fois. Mêmes réglages et même gestion de la licence
    que le script 04 (section 2 de son main(), reprise ici)."""
    os.environ["nnUNet_def_n_proc"] = str(a.processus)     # à fixer AVANT d'importer totalsegmentator
    import torch
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.config import (setup_totalseg, set_config_key, set_license_number,
                                         has_valid_license_offline, get_version)
    from totalsegmentator.map_to_binary import commercial_models

    if a.appareil == "gpu" and not torch.cuda.is_available():
        arret(5, "aucun GPU visible. Menu Exécution → Modifier le type d'exécution → GPU T4, puis relancer.")
    setup_totalseg()
    set_config_key("send_usage_stats", bool(a.statistiques_usage))
    print(f"Statistiques d'usage envoyées à TotalSegmentator : {'OUI (demandé)' if a.statistiques_usage else 'non'}")
    if a.tache in commercial_models:
        if cle:
            print(f"Clé de licence reçue par TOTALSEG_LICENCE : {cle[:4]}… ({len(cle)} caractères, masquée)")
            try:
                set_license_number(cle)
            except SystemExit:
                arret(4, "clé de licence refusée par TotalSegmentator (voir le message juste au-dessus).", cle)
        etat, _ = has_valid_license_offline()              # message jamais affiché : il peut contenir la clé
        if etat != "yes":
            arret(4, f"la tâche {a.tache} demande une licence (état : {etat}). Voir la section 10 du notebook.", cle)

    os.makedirs(sortie, exist_ok=True)
    trace = os.path.join(sortie, FICHIER_TRACE)
    if os.path.exists(trace):
        os.remove(trace)                                   # une relance qui échoue ne laisse pas de trace trompeuse
    options = {"task": a.tache, "ml": True, "device": a.appareil,
               "nr_thr_resamp": a.processus, "nr_thr_saving": a.processus}   # comme le script 04
    resultats = []
    for n, c in chemins.items():
        print(f"\n=== mmwhs_{n} (entraînement) : prédiction en cours ===", flush=True)
        t0 = time.time()
        try:
            empreinte = sha256_fichier(c["image"])
            totalsegmentator(c["image"], c["pred"], **options)
        except (Exception, SystemExit) as err:
            print(f"\nERREUR PENDANT LA PRÉDICTION (mmwhs_{n}) : {sans_cle(repr(err), cle)}")
            print("Corriger la cause, puis relancer avec --ecraser.")
            sys.exit(3)
        resultats.append({"cas": f"mmwhs_{n}", "scanner_sha256": empreinte, "duree_s": round(time.time() - t0, 1)})
        print(f"mmwhs_{n} : {resultats[-1]['duree_s']} s")
    with open(trace, "w", encoding="utf-8") as f:
        json.dump({"objet": "comparaison des définitions avant l'étape 6 (cas d'ENTRAÎNEMENT seulement)",
                   "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                   "totalsegmentator_version": get_version(),
                   "nnunetv2_version": etape4.version_paquet("nnunetv2"), "torch_version": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "aucun",
                   "code_commit": etape4.commande("git", "rev-parse", "HEAD") or "inconnu",
                   "code_modifie_localement": bool(etape4.commande("git", "status", "--porcelain")),
                   "options": options, "statistiques_usage_envoyees": bool(a.statistiques_usage),
                   "split_sha256": sha256_fichier(a.split), "cas": resultats,
                   "rappel": "Prédictions TotalSegmentator : ne jamais s'en servir pour entraîner (règle 5)."},
                  f, indent=2, ensure_ascii=False)


# --- Mesures et figure d'un cas (aucun score) -------------------------------------------------
def charger(c):
    """Lit scanner, vérité et prédiction ; vérifie qu'ils partagent la même grille ; les remet en RAS
    (1er axe vers la droite du patient, 2e vers l'avant, 3e vers le haut)."""
    import nibabel as nib
    import numpy as np
    imgs = {k: nib.load(c[k]) for k in ("image", "verite", "pred")}
    ref = imgs["image"]
    for k in ("verite", "pred"):
        if imgs[k].shape[:3] != ref.shape[:3] or not np.allclose(imgs[k].affine, ref.affine, atol=1e-3):
            return None
    ras = {k: nib.as_closest_canonical(v) for k, v in imgs.items()}
    ct = np.asanyarray(ras["image"].dataobj).astype(np.float32)
    ver = np.asanyarray(ras["verite"].dataobj).astype(np.uint8)
    pred = np.asanyarray(ras["pred"].dataobj).astype(np.uint8)
    esp = np.sqrt((ras["image"].affine[:3, :3] ** 2).sum(axis=0))   # espacement (mm) le long de chaque axe
    return ct, ver, pred, ras["image"].affine, esp, float(ref.header.get_zooms()[2])


def vue(t):
    """Sens des radiologues pour imshow(origin='lower') : 1er axe retourné puis transposé.
    Axiale : droite du patient à gauche, avant en haut. Coronale : droite à gauche, tête en haut.
    Sagittale : avant à gauche, tête en haut."""
    import numpy as np
    return np.flip(t, 0).T


def mesurer(m, affine, ic, jc, vol_voxel_ml):
    """Volume (mL), niveaux bas/haut (mm) et nombre de morceaux d'un masque (déjà recadré : m, k0)."""
    import numpy as np
    from scipy import ndimage
    masque, k0 = m
    if not masque.any():
        return {"vol": 0.0, "bas": None, "haut": None, "morceaux": 0}
    ks = np.where(masque.any(axis=(0, 1)))[0] + k0
    z = [float((affine @ [ic, jc, k, 1])[2]) for k in (ks.min(), ks.max())]
    return {"vol": float(masque.sum()) * vol_voxel_ml, "bas": min(z), "haut": max(z),
            # structure 3x3x3 = connexité 26 : face, arête ou coin (sinon de petits îlots gonfleraient le compte)
            "morceaux": int(ndimage.label(masque, structure=np.ones((3, 3, 3)))[1])}


def panneau(ax, fond, v, p, aspect, titre, lettres, niveaux=()):
    """Une vue : scanner en gris + contours (fond donné), ou projection colorée des masques (fond None)."""
    import numpy as np
    if fond is not None:
        ax.imshow(vue(fond), cmap="gray", vmin=FENETRE[0], vmax=FENETRE[1], origin="lower", aspect=aspect)
        # Vérité en trait épais dessous, TotalSegmentator en trait fin dessus : là où les deux contours
        # se confondent, on voit un liseré bleu autour du trait orange.
        for masque, couleur, epais in ((v, BLEU, 2.6), (p, ORANGE, 1.1)):
            if masque.any():
                ax.contour(vue(masque).astype(np.float32), levels=[0.5], colors=[couleur], linewidths=epais)
    else:
        rgb = np.zeros(v.shape + (3,), np.float32)
        for zone, couleur in ((v & ~p, BLEU), (~v & p, ORANGE), (v & p, GRIS)):
            rgb[zone] = [int(couleur[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        ax.imshow(np.flip(rgb, 0).transpose(1, 0, 2), origin="lower", aspect=aspect)
        for z, couleur in niveaux:                          # niveaux haut et bas de chaque source
            ax.axhline(z, color=couleur, ls="--", lw=0.9)
    gauche, droite, haut, bas = lettres
    for x, y, t in ((0.02, 0.5, gauche), (0.98, 0.5, droite), (0.5, 0.97, haut), (0.5, 0.02, bas)):
        ax.text(x, y, t, transform=ax.transAxes, color="yellow", fontsize=9, ha="center", va="center")
    ax.set_title(titre, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def analyser(n, c, chemin_figure):
    """Chiffres et figure d'un cas. Un seul cas en mémoire à la fois."""
    import matplotlib
    matplotlib.use("Agg")                                   # figure enregistrée en PNG (pas d'écran ici)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np
    from scipy import ndimage
    lu = charger(c)
    if lu is None:
        return None
    ct, ver, pred, affine, esp, epaisseur = lu
    vol_voxel_ml = float(np.prod(esp)) / 1000
    marge = [int(math.ceil(MARGE_MM / e)) for e in esp]
    lignes = []
    fig = plt.figure(figsize=(19, 3.4 * len(STRUCTURES) + 1.2))
    # Une bande en haut (titre et légende), puis une rangée de 5 vues par structure.
    entete, *sous_figures = fig.subfigures(len(STRUCTURES) + 1, 1, height_ratios=[0.35] + [1] * len(STRUCTURES))
    entete.suptitle(f"mmwhs_{n} (ENTRAÎNEMENT, coupes de {epaisseur:.3g} mm) — vérité MM-WHS contre "
                 "TotalSegmentator — droite du patient à gauche de l'image", fontsize=14)
    entete.legend(handles=[Line2D([], [], color=BLEU, lw=2, label="contour vérité MM-WHS"),
                        Line2D([], [], color=ORANGE, lw=2, label="contour TotalSegmentator"),
                        Patch(color=BLEU, label="projection : MM-WHS seul"),
                        Patch(color=ORANGE, label="projection : TotalSegmentator seul"),
                        Patch(color=GRIS, label="projection : les deux")],
               loc="lower center", ncol=5, fontsize=10)
    for sf, (lab, (nom, nom_ts)) in zip(sous_figures, STRUCTURES.items()):
        v_tout, p_tout = ver == lab, pred == lab
        union = v_tout | p_tout
        axes = sf.subplots(1, 5)
        if not union.any():
            sf.suptitle(f"{lab} — {nom} / {nom_ts} : absente des deux masques", fontsize=11)
            for ax in axes:
                ax.axis("off")
            lignes.append({"cas": f"mmwhs_{n}", "etiquette": lab, "structure": nom})
            continue
        # Boîte englobante de l'union + marge, puis recadrage (moins de mémoire, vues lisibles).
        bornes = [np.where(union.any(axis=tuple(a for a in range(3) if a != ax_)))[0] for ax_ in range(3)]
        (x0, x1), (y0, y1), (z0, z1) = [(max(b.min() - m, 0), min(b.max() + m + 1, s))
                                        for b, m, s in zip(bornes, marge, union.shape)]
        v, p, u = (t[x0:x1, y0:y1, z0:z1] for t in (v_tout, p_tout, union))
        f = ct[x0:x1, y0:y1, z0:z1]
        # Coupes : on part du centre de la vérité MM-WHS (de l'union si la vérité est vide). Le centre de
        # l'union tomberait, pour une aorte de TotalSegmentator qui descend loin, ENTRE l'aorte ascendante
        # et l'aorte descendante. On prend ensuite la coupe la plus proche de ce centre qui contient les
        # DEUX masques ; à défaut (une source vide), la coupe où l'union est la plus étendue.
        centre = [int(round(x)) for x in ndimage.center_of_mass(v if v.any() else u)]
        for ax_ in range(3):
            autres = tuple(a for a in range(3) if a != ax_)
            communes = np.where(v.any(axis=autres) & p.any(axis=autres))[0]
            if communes.size:
                centre[ax_] = int(communes[np.argmin(np.abs(communes - centre[ax_]))])
            elif not np.take(u, centre[ax_], axis=ax_).any():
                centre[ax_] = int(np.argmax(u.sum(axis=autres)))
        i, j, k = centre
        ic, jc = i + x0, j + y0
        mv = mesurer((v, z0), affine, ic, jc, vol_voxel_ml)
        mp = mesurer((p, z0), affine, ic, jc, vol_voxel_ml)
        z_mm = float((affine @ [ic, jc, k + z0, 1])[2])
        sf.suptitle(f"{lab} — {nom} (MM-WHS) / {nom_ts} (TotalSegmentator) — volume : MM-WHS "
                    f"{mv['vol']:.1f} mL, TotalSegmentator {mp['vol']:.1f} mL", fontsize=11)
        niveaux = []
        for mes, couleur in ((mv, BLEU), (mp, ORANGE)):
            if mes["bas"] is not None:   # niveau (mm) → indice de coupe dans le recadrage
                niveaux += [((z - affine[2, 3] - affine[2, 0] * ic - affine[2, 1] * jc) / affine[2, 2] - z0, couleur)
                            for z in (mes["bas"], mes["haut"])]
        panneau(axes[0], f[:, :, k], v[:, :, k], p[:, :, k], esp[1] / esp[0],
                f"axiale (niveau {z_mm:.0f} mm)", ("D", "G", "A", "P"))
        panneau(axes[1], f[:, j, :], v[:, j, :], p[:, j, :], esp[2] / esp[0], "coronale", ("D", "G", "H", "B"))
        panneau(axes[2], f[i, :, :], v[i, :, :], p[i, :, :], esp[2] / esp[1], "sagittale", ("A", "P", "H", "B"))
        panneau(axes[3], None, v.any(axis=1), p.any(axis=1), esp[2] / esp[0],
                "projection coronale (de face)", ("D", "G", "H", "B"), niveaux)
        panneau(axes[4], None, v.any(axis=0), p.any(axis=0), esp[2] / esp[1],
                "projection sagittale (de profil)", ("A", "P", "H", "B"), niveaux)
        ligne = {"cas": f"mmwhs_{n}", "etiquette": lab, "structure": nom}
        for cle_, mes in (("mmwhs", mv), ("ts", mp)):
            ligne.update({f"vol_ml_{cle_}": round(mes["vol"], 1), f"morceaux_{cle_}": mes["morceaux"],
                          f"bas_mm_{cle_}": None if mes["bas"] is None else round(mes["bas"], 1),
                          f"haut_mm_{cle_}": None if mes["haut"] is None else round(mes["haut"], 1)})
        lignes.append(ligne)
    fig.savefig(chemin_figure, dpi=80)
    plt.close(fig)
    del ct, ver, pred
    return lignes


def afficher(lignes):
    """Tableau lisible des chiffres d'un cas. Écart = TotalSegmentator − MM-WHS, en mm."""
    def f(x, fmt="{:7.1f}"):
        return fmt.format(x) if x is not None else "      —"
    print(f"\n{lignes[0]['cas']} — volumes (mL) et niveaux (mm, plus grand = vers la tête). Aucun score.")
    print(f"{'structure':<19}{'vol MM':>8}{'vol TS':>8} | {'bas MM':>7}{'bas TS':>7}{'écart':>7} | "
          f"{'haut MM':>8}{'haut TS':>8}{'écart':>7} | morceaux MM/TS")
    for l in lignes:
        if "vol_ml_mmwhs" not in l:
            print(f"{l['etiquette']} {l['structure']:<17}absente des deux masques")
            continue
        eb = None if None in (l["bas_mm_ts"], l["bas_mm_mmwhs"]) else l["bas_mm_ts"] - l["bas_mm_mmwhs"]
        eh = None if None in (l["haut_mm_ts"], l["haut_mm_mmwhs"]) else l["haut_mm_ts"] - l["haut_mm_mmwhs"]
        l["ecart_bas_mm"], l["ecart_haut_mm"] = (None if e is None else round(e, 1) for e in (eb, eh))
        print(f"{l['etiquette']} {l['structure']:<17}{f(l['vol_ml_mmwhs'], '{:8.1f}')}{f(l['vol_ml_ts'], '{:8.1f}')} | "
              f"{f(l['bas_mm_mmwhs'])}{f(l['bas_mm_ts'])}{f(eb, '{:+7.1f}')} | "
              f"{f(l['haut_mm_mmwhs'], '{:8.1f}')}{f(l['haut_mm_ts'], '{:8.1f}')}{f(eh, '{:+7.1f}')} | "
              f"{l['morceaux_mmwhs']}/{l['morceaux_ts']}")


# --- Programme principal ----------------------------------------------------------------------
def main():
    a = lire_arguments()
    cle = (os.environ.get("TOTALSEG_LICENCE") or "").strip()   # jamais affichée
    dataset, sortie, chemins = verifier(a)

    if not a.ecraser and predictions_reutilisables(sortie, chemins):
        print(f"Prédictions déjà présentes dans {sortie} (mêmes scanners) : RÉUTILISÉES, TotalSegmentator "
              "n'est pas relancé (--ecraser pour les refaire).")
    else:
        deja = [c["pred"] for c in chemins.values() if os.path.exists(c["pred"])]
        if deja and not a.ecraser:
            print(f"ARRÊT : {len(deja)} prédiction(s) déjà dans {sortie}, mais incomplètes, sans traçabilité "
                  "ou faites sur d'autres scanners. Rien n'a été écrasé : relancer avec --ecraser.")
            sys.exit(2)
        predire(a, cle, sortie, chemins)

    toutes = []
    for n, c in chemins.items():
        print(f"\n=== mmwhs_{n} : mesures et figure ===", flush=True)
        figure = os.path.join(sortie, NOM_FIGURE.format(n=n))
        lignes = analyser(n, c, figure)
        if lignes is None:
            arret(6, f"mmwhs_{n} : scanner, vérité terrain et prédiction n'ont pas la même grille de voxels.")
        afficher(lignes)
        print(f"Figure : {figure}")
        toutes += lignes
    colonnes = list(dict.fromkeys(k for l in toutes for k in l))
    with open(os.path.join(sortie, FICHIER_CSV), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=colonnes)
        w.writeheader()
        w.writerows(toutes)
    print(f"\nChiffres : {os.path.join(sortie, FICHIER_CSV)}")
    print("Aucun score (Dice, HD95) n'a été calculé, aucun fichier du lot de test n'a été lu.")


# Garde indispensable : les processus enfants de nnU-Net (mode spawn) relisent ce fichier.
if __name__ == "__main__":
    main()
