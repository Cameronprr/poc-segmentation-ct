"""
Étape 5g — Examiner n'importe quel cas de validation de nnU-Net, structure par structure (05d généralisé).
LECTURE SEULE : aucun fichier nnU-Net n'est modifié ; le lot de test n'est jamais lu.

Contexte : 05d a été écrit pour UN échec (vaisseaux de 1019, pli 0). Ce script fait le même examen, avec les
mêmes figures, pour N'IMPORTE QUEL cas de validation des 5 plis et N'IMPORTE QUELLES structures (1 à 7). Exemple :
1015 (pli 1), VD à 0,83 et artère pulmonaire (AP) à 0,65, avec à la fois des oublis et des ajouts. Le pli de
chaque cas est retrouvé dans splits_final.json. Les indicateurs aident à trancher entre :
  (a) FRONTIÈRE : la limite entre deux structures voisines (ex. VD / AP au niveau de la valve pulmonaire) est
      placée ailleurs par le modèle que par l'annotateur : les oublis de l'une sont prédits comme l'autre et
      les ajouts de l'autre sont posés sur l'une (erreurs « croisées »). Petit décalage : liseré, près du bord ;
      grand décalage : erreurs loin et au-delà de la boîte englobante, mais TOUJOURS croisées ;
  (b) ÉTENDUE : la vérité ou la prédiction va plus loin (branches de l'AP, haut de l'aorte...) : erreurs
      loin et au-delà de la boîte englobante (au-dessus, en dessous, sur les côtés), prédites « fond » (oublis)
      ou posées sur le fond de la vérité (ajouts) ;
  « l'autre masque » = la même structure dans l'autre carte : la prédiction pour un oubli, la vérité pour un ajout ;
  (c) CONFUSION OU OUBLI FRANC : un morceau entier de la structure est pris pour une autre, ou pour le fond ;
  (d) EXAMEN INHABITUEL : phase d'injection, bruit, contraste, cadrage (comme 1019, D27).
Il NE CONCLUT PAS : il imprime les chiffres et ce que chacun suggérerait. La décision reste humaine.

Ce que fait le script, pour chaque cas demandé (le premier = cas étudié, les suivants = comparaison) :
  0. garde-fous : splits_final.json vérifié comme dans 05b/05c ; chaque cas doit être en ENTRAÎNEMENT dans le
     découpage figé (lu seulement), tout cas de test est refusé ; seuls imagesTr, labelsTr, dataset.json et
     fold_<pli>/validation sont lus ; --sortie hors du dépôt et des dossiers nnU-Net ; étiquettes de
     dataset.json = celles du script 03 ;
  1. géométrie (fonction charger() de 05d : image, vérité et prédiction sur la même grille, sinon code 6) ;
  2. Dice des 7 structures, recalculé et comparé au summary.json du pli (écart = ligne ANOMALIE) ;
  3. structures choisies (--structures ; par défaut les 2 plus faibles du cas étudié) : mesures de 05d
     (volume, niveaux, boîte englobante, bords du champ, centre de gravité, morceaux) vérité ET prédiction ;
  4. ERREURS de chaque structure choisie :
       - oublis (vérité non prédite) : ce que la prédiction y met à la place (fond, autre structure) ;
       - ajouts (prédit hors de la vérité) : ce que la vérité y met (fond, autre structure) ;
       - distance de chaque erreur à l'autre masque : PRÈS (≤ 2 mm) = liseré de frontière, LOIN (> 5 mm) =
         morceau entier (étendue ou confusion) ;
       - part des erreurs AU-DELÀ de la boîte englobante de l'autre masque (au-dessus, en dessous, côtés) ;
       - niveaux, morceaux et densité (HU) des erreurs ;
  5. matrice de confusion 8 × 8, en % ;
  6. densités (HU) des 7 structures de la vérité ; phase d'injection (rapport aorte / AP, comme 05e) ;
     bruit et CNR (comme 05d) ;
  7. une figure par cas, mise en page de 05d : une rangée par structure choisie (coupes axiale, coronale,
     sagittale par son centre, et projection de face) ; contours de toutes les structures choisies ;
  8. récapitulatif rangé par hypothèse, cas côte à côte.

Entrées :
  --cas          numéros des cas, OBLIGATOIRE ; le premier est le cas étudié (ex. : 1015 1005)
  --structures   étiquettes à examiner, 1 à 4 parmi 1 à 7 (1 myocarde VG, 2 OG, 3 cavité VG, 4 OD, 5 VD,
                 6 aorte, 7 AP) ; défaut : les 2 au Dice le plus bas sur le cas étudié
  --raw          dossier nnUNet_raw (défaut /content/drive/MyDrive/poc_ct/nnunet_raw) ; seuls
                 Dataset101_MMWHS/imagesTr, labelsTr et dataset.json sont lus
  --preprocessed dossier nnUNet_preprocessed : splits_final.json et ses fiches de 05a (lus seulement)
  --results      dossier nnUNet_results ; seul .../fold_<pli>/validation/ est lu (prédiction et summary.json)
  --sortie       dossier de sortie (défaut /content/drive/MyDrive/poc_ct/analyses/05g_<cas étudié>)
  --split        découpage figé (défaut : ../splits/mmwhs_split.json à côté du script)
Sorties (uniquement dans --sortie) :
  examen_mmwhs_XXXX.png (une figure par cas) ; synthese_05g.json (tous les chiffres imprimés)
Codes de sortie :
  0 = examen fait (lire les lignes ANOMALIE) ; 1 = garde-fou (rien n'est écrit) ; 6 = image, vérité et
  prédiction n'ont pas la même forme (comparaison voxel à voxel impossible).

Vocabulaire (en plus de celui de 05d) :
  - oubli (faux négatif) : voxel de la vérité que le modèle ne met pas dans la structure ;
  - ajout (faux positif) : voxel que le modèle met dans la structure alors que la vérité ne l'y met pas ;
  - boîte englobante : le plus petit pavé qui contient un masque. Un oubli « au-dessus de la boîte prédite »
    est plus haut que tout ce que le modèle a prédit pour cette structure : la vérité monte plus haut ;
  - liseré : bande fine le long d'un bord. Des erreurs en liseré signalent un désaccord sur la position du
    bord, pas une structure manquée ;
  - p90 : 90 % des valeurs sont en dessous.
"""
import argparse, datetime, importlib.util, json, os, re, sys

ICI = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("bilan05f", os.path.join(ICI, "05f_bilan_validation_croisee.py"))
bilan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bilan)                 # 05f n'exécute rien à l'import ; il apporte 05a, 05b et 05d
ana, prep = bilan.ana, bilan.prep

# --- Paramètres fixes -------------------------------------------------------------------------
NOMS = ana.NOMS                                 # noms des étiquettes (dataset.json vérifié par rapport à eux)
COURTS = bilan.STRUCTURES                       # noms courts : 1 myo_VG ... 7 AP
FENETRE = ana.FENETRE
PRES_MM, LOIN_MM = 2.0, 5.0                     # repères INDICATIFS pour lire les distances des erreurs
SEUIL_PROCHE = 1.20                             # même repère que 05e : aorte < 1,20 × AP → « proche »
TOLERANCE_DICE = 1e-3                           # écart toléré entre notre Dice et celui de summary.json
# Couleurs (vérité, prédiction) par position dans --structures : les deux premières paires sont celles de 05d.
COULEURS = (("#FF3B30", "#FFE600"), ("#2F80FF", "#FF40FF"), ("#00C853", "#00E5FF"), ("#FF9100", "#FFFFFF"))
NOM_FIGURE, NOM_SYNTHESE = "examen_mmwhs_{n}.png", "synthese_05g.json"


def lire_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--cas", nargs="+", required=True)
    p.add_argument("--structures", nargs="+", type=int, default=None, choices=range(1, 8))
    p.add_argument("--raw", default=f"{prep.DRIVE}/nnunet_raw")
    p.add_argument("--preprocessed", default=f"{prep.DRIVE}/nnunet_preprocessed")
    p.add_argument("--results", default=f"{prep.DRIVE}/nnunet_results")
    p.add_argument("--sortie", default=None)
    p.add_argument("--split", default=os.path.join(prep.DEPOT, "splits", "mmwhs_split.json"))
    a = p.parse_args()
    if a.structures is not None and not 1 <= len(set(a.structures)) <= len(COULEURS):
        p.error(f"--structures : de 1 à {len(COULEURS)} étiquettes différentes")
    if a.structures is not None:
        a.structures = sorted(set(a.structures))
    if len(set(a.cas)) != len(a.cas):
        p.error("--cas : un même cas est donné deux fois")
    a.sortie = a.sortie or f"{prep.DRIVE}/analyses/05g_{a.cas[0]}"
    return a


# --- 0. Garde-fous (rien n'est écrit avant qu'ils soient passés) -------------------------------
def verifier(a):
    raw, pre, res = (os.path.abspath(x) for x in (a.raw, a.preprocessed, a.results))
    sortie = bilan.verifier_sortie(a.sortie, raw, pre, res)
    plis, strates, _, cas_entr, cas_test = bilan.lire_plis(raw, pre, a.split)
    pli_de = {c: k for k, p in enumerate(plis) for c in p["val"]}
    ds = os.path.join(raw, ana.NOM_DS)
    chemins = {}
    for n in a.cas:
        cas = f"mmwhs_{n}"
        if not re.fullmatch(r"\d{4}", n):
            bilan.arret(1, f"numéro de cas invalide : {n} (attendu : 4 chiffres, ex. 1015).")
        if cas in cas_test:
            bilan.arret(1, f"{cas} fait partie du LOT DE TEST : refusé (le test ne sert qu'à l'étape 6, D18).")
        if cas not in cas_entr or cas not in pli_de:
            bilan.arret(1, f"{cas} n'est pas un cas de validation du découpage figé : refusé.")
        k = pli_de[cas]
        validation = os.path.join(bilan.dossier_pli(res, k), "validation")
        c = {"image": os.path.join(ds, "imagesTr", f"{cas}_0000.nii.gz"),
             "verite": os.path.join(ds, "labelsTr", f"{cas}.nii.gz"),
             "pred": os.path.join(validation, f"{cas}.nii.gz"),
             "summary": os.path.join(validation, "summary.json")}
        if any(d in ("imagesTs", "labelsTs") for x in c.values() for d in x.split(os.sep)):
            bilan.arret(1, "un chemin passe par imagesTs ou labelsTs : refusé.")
        manquants = [x for x in c.values() if not os.path.isfile(x)]
        if manquants:
            bilan.arret(1, f"fichiers introuvables (Drive relié ? pli {k} terminé ?) : {manquants}")
        chemins[n] = {**c, "pli": k, "epaisseur_mm": strates.get(cas)}
    with open(os.path.join(ds, "dataset.json"), encoding="utf-8") as f:
        etiquettes_ds = json.load(f).get("labels")
    if etiquettes_ds != ana.ETIQUETTES:
        bilan.arret(1, f"les étiquettes de dataset.json ({etiquettes_ds}) ne sont pas celles attendues ({ana.ETIQUETTES}).")
    liste = ", ".join(f"mmwhs_{n} (pli {c['pli']})" for n, c in chemins.items())
    print(f"Cas vérifiés : {liste} "
          "sont des cas de validation (ENTRAÎNEMENT, aucun cas de test).")
    print("Étiquettes de dataset.json conformes (1 myo VG, 2 OG, 3 cavité VG, 4 OD, 5 VD, 6 aorte, 7 AP).")
    return ds, sortie, chemins, cas_entr, cas_test


# --- 2. Dice ----------------------------------------------------------------------------------
def dices(ver, pred):
    """Dice des 7 structures (None si la structure est vide dans la vérité ET la prédiction)."""
    import numpy as np
    r = {}
    for e in COURTS:
        v, p = ver == e, pred == e
        s = int(v.sum()) + int(p.sum())
        r[e] = None if s == 0 else 2 * int(np.logical_and(v, p).sum()) / s
    return r


def comparer_summary(n, c, d, cas_test):
    """Compare notre Dice à celui écrit par nnU-Net pour ce cas (summary.json de son pli)."""
    # Les « problèmes » renvoyés (le summary contient les 3 cas du pli, pas un seul) sont ignorés ici :
    # la cohérence complète des summary.json avec les plis est contrôlée par 05f.
    cas_pli, _, _ = bilan.lire_summary(c["summary"], [f"mmwhs_{n}"], cas_test)
    ecrit = cas_pli.get(f"mmwhs_{n}", {})
    anomalies = []
    for e in COURTS:
        x = (ecrit.get(e) or {}).get("dice")
        if (x is None) != (d[e] is None) or (x is not None and abs(x - d[e]) > TOLERANCE_DICE):
            anomalies.append(f"Dice {COURTS[e]} recalculé {bilan.f3(d[e])} ≠ summary.json {bilan.f3(x)}")
    if f"mmwhs_{n}" not in cas_pli:
        anomalies.append("cas absent du summary.json de son pli")
    return {e: (ecrit.get(e) or {}).get("dice") for e in COURTS}, anomalies


# --- 4. Erreurs -------------------------------------------------------------------------------
def distances_mm(source, cible, esp):
    """Pour chaque voxel de source : distance (mm) au voxel de cible le plus proche. Calcul recadré sur la
    boîte qui contient les deux masques ; c'est exact, car le voxel de cible le plus proche est forcément
    dans la boîte de cible, donc dans le recadrage."""
    import numpy as np
    from scipy import ndimage
    if not source.any() or not cible.any():
        return None
    idx = np.where(source | cible)
    sl = tuple(slice(int(x.min()), int(x.max()) + 1) for x in idx)
    return ndimage.distance_transform_edt(~cible[sl], sampling=esp)[source[sl]]


def decrire_erreur(err, total, etiquettes_autre, masque_autre, ct, affine, esp):
    """Erreurs d'un type (oublis ou ajouts) d'une structure.
    etiquettes_autre : l'autre carte d'étiquettes (prédiction pour les oublis, vérité pour les ajouts) ;
    masque_autre : la même structure dans l'autre carte (prédiction pour les oublis, vérité pour les ajouts)."""
    import numpy as np
    n = int(err.sum())
    if n == 0:
        return {"voxels": 0, "part": 0.0 if total else None}
    r = {"voxels": n, "part": round(n / total, 4) if total else None}
    rep = np.bincount(etiquettes_autre[err].astype(np.int64), minlength=8)[:8]
    r["repartition_pct"] = {int(e): round(100 * int(x) / n, 1) for e, x in enumerate(rep)}
    d = distances_mm(err, masque_autre, esp)
    if d is not None:
        r.update({"distance_mediane_mm": round(float(np.median(d)), 1),
                  "distance_p90_mm": round(float(np.percentile(d, 90)), 1),
                  "pct_pres": round(100 * float((d <= PRES_MM).mean()), 1),
                  "pct_loin": round(100 * float((d > LOIN_MM).mean()), 1)})
    if masque_autre.any():
        b = [np.where(masque_autre.any(axis=tuple(x for x in range(3) if x != ax)))[0] for ax in range(3)]
        (i0, i1), (j0, j1), (k0, k1) = [(int(x.min()), int(x.max())) for x in b]
        i, j, k = np.where(err)
        haut, bas = k > k1, k < k0
        cotes = ~haut & ~bas & ((i < i0) | (i > i1) | (j < j0) | (j > j1))
        r["au_dela_boite_pct"] = {"au_dessus": round(100 * float(haut.mean()), 1),
                                  "en_dessous": round(100 * float(bas.mean()), 1),
                                  "sur_les_cotes": round(100 * float(cotes.mean()), 1)}
    m = ana.mesurer(err, affine, esp)
    r.update({"niveau_bas_mm": m["niveau_bas_mm"], "niveau_haut_mm": m["niveau_haut_mm"],
              "centre_mm": m["centre_mm"], "morceaux": m["morceaux"],
              "morceaux_3_plus_gros_ml": m["morceaux_3_plus_gros_ml"]})
    h = ana.intensites(ct, err, esp)
    r["hu_mediane"] = h["mediane"] if h else None
    return r


# --- Analyse d'un cas -------------------------------------------------------------------------
def analyser(n, c, structures, ct, ver, pred, affine, esp, geo):
    import numpy as np
    r = {"cas": f"mmwhs_{n}", "pli": c["pli"], "epaisseur_mm": c["epaisseur_mm"], "geometrie": geo,
         "structures": {}, "intensites": {}}
    for lab in structures:
        v, p = ver == lab, pred == lab
        mv, mp = ana.mesurer(v, affine, esp), ana.mesurer(p, affine, esp)
        dec = (None if not (mv["voxels"] and mp["voxels"]) else
               round(float(np.linalg.norm(np.subtract(mp["centre_mm"], mv["centre_mm"]))), 1))
        r["structures"][COURTS[lab]] = {
            "etiquette": lab, "verite": mv, "prediction": mp, "decalage_centres_mm": dec,
            "oublis": decrire_erreur(v & ~p, mv["voxels"], pred, p, ct, affine, esp),
            "ajouts": decrire_erreur(p & ~v, mp["voxels"], ver, v, ct, affine, esp)}
    r["structures"]["coeur_entier"] = {"verite": ana.mesurer(ver > 0, affine, esp)}
    r["confusion_verite_vers_prediction"] = ana.confusion(ver, pred).tolist()
    for lab in COURTS:
        r["intensites"][f"verite_{lab}_{NOMS[lab]}"] = ana.intensites(ct, ver == lab, esp)
    for lab in structures:
        r["intensites"][f"prediction_{lab}_{NOMS[lab]}"] = ana.intensites(ct, pred == lab, esp)
    return r


# --- 7. Figure (mise en page de 05d, pour n'importe quelles structures) -------------------------
def figure(n, r, ct, ver, pred, esp, structures, chemin):
    import matplotlib
    matplotlib.use("Agg")                                   # figure enregistrée en PNG
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    couleurs = {lab: COULEURS[i] for i, lab in enumerate(structures)}
    geo = r["geometrie"]
    fig, axes = plt.subplots(len(structures), 4, figsize=(22, 6.25 * len(structures) + 0.5), squeeze=False)
    dices_txt = ", ".join(f"{COURTS[l]} {bilan.f3(r['dice'][l])}" for l in structures)
    fig.suptitle(f"mmwhs_{n} (validation du pli {r['pli']}, coupes de {geo['epaisseur_coupe_mm']} mm ; Dice {dices_txt}) — "
                 f"vérité contre prédiction nnU-Net — fenêtre {FENETRE[0]}/{FENETRE[1]} HU — "
                 "bords de l'image = bords du champ", fontsize=13)

    def contours(ax, v2, p2):
        # Vérité (trait épais) d'abord, prédiction (trait fin) par-dessus, comme 05d.
        for source, t, epais in (("verite", v2, 2.4), ("pred", p2, 1.3)):
            for l in structures:
                m = t == l
                if m.any():
                    ax.contour(ana.vue(m).astype(np.float32), levels=[0.5],
                               colors=[couleurs[l][0 if source == "verite" else 1]], linewidths=epais)

    for ligne, lab in zip(axes, structures):
        mv = ver == lab
        base = mv if mv.any() else (pred == lab)
        if not base.any():
            for ax in ligne:
                ax.axis("off")
            ligne[0].set_title(f"{NOMS[lab]} absente de la vérité et de la prédiction")
            continue
        centre = [int(round(x)) for x in np.argwhere(base).mean(axis=0)]
        i, j, k = (ana.coupe_utile(base, ax_, centre[ax_]) for ax_ in range(3))
        quelle = "vérité" if mv.any() else "prédiction (vérité vide)"
        vues = [(ct[:, :, k], ver[:, :, k], pred[:, :, k], esp[1] / esp[0], f"axiale, coupe {k}", ("D", "G", "A", "P"), None),
                (ct[:, j, :], ver[:, j, :], pred[:, j, :], esp[2] / esp[0], "coronale", ("D", "G", "H", "B"), k),
                (ct[i, :, :], ver[i, :, :], pred[i, :, :], esp[2] / esp[1], "sagittale", ("A", "P", "H", "B"), k)]
        for ax, (fond, v2, p2, aspect, titre, lettres, niveau) in zip(ligne[:3], vues):
            ax.imshow(ana.vue(fond), cmap="gray", vmin=FENETRE[0], vmax=FENETRE[1], origin="lower", aspect=aspect)
            contours(ax, v2, p2)
            if niveau is not None:
                ax.axhline(niveau, color="white", ls=":", lw=0.8)
            ana.lettres_bords(ax, lettres)
            ax.set_title(f"{NOMS[lab]} — {titre} (centre de la {quelle})", fontsize=10)
        # Projection de face (comme une ombre) : tout le cœur de la vérité en gris, contours des projections.
        ax = ligne[3]
        ax.imshow(ana.vue((ver > 0).any(axis=1).astype(np.float32)), cmap="gray", vmin=0, vmax=3,
                  origin="lower", aspect=esp[2] / esp[0])
        for source, t, epais in (("verite", ver, 2.4), ("pred", pred, 1.3)):
            for l in structures:
                m = (t == l).any(axis=1)
                if m.any():
                    ax.contour(ana.vue(m).astype(np.float32), levels=[0.5],
                               colors=[couleurs[l][0 if source == "verite" else 1]], linewidths=epais)
        ax.axhline(k, color="white", ls=":", lw=0.8)
        ana.lettres_bords(ax, ("D", "G", "H", "B"))
        ax.set_title("projection de face (gris = cœur de la vérité) ; cadre = bords du champ", fontsize=10)
    fig.legend(handles=[Line2D([], [], color=couleurs[l][i], lw=2.4 if i == 0 else 1.3,
                               label=f"{'vérité' if i == 0 else 'prédiction'} {NOMS[l]}")
                        for l in structures for i in (0, 1)]
               + [Line2D([], [], color="black", ls=":", label="niveau de la coupe axiale (pointillé blanc)")],
               loc="lower center", ncol=min(5, 2 * len(structures) + 1), fontsize=11)
    fig.tight_layout(rect=(0, 0.05 if len(structures) <= 2 else 0.03, 1, 0.95), h_pad=3)
    fig.savefig(chemin, dpi=80)
    plt.close(fig)


# --- Affichage lisible ------------------------------------------------------------------------
def afficher_cas(r, structures):
    g = r["geometrie"]
    print(f"\n=== {r['cas']} — 1. géométrie ===")
    for libelle, cle in (("forme (fichier)", "forme_fichier"), ("espacement (mm, fichier)", "espacement_fichier_mm"),
                         ("épaisseur de coupe (mm)", "epaisseur_coupe_mm"), ("orientation (fichier)", "orientation_fichier"),
                         ("étendue du champ (mm) D-G / A-P / bas-haut", "etendue_mm_RL_AP_IS"),
                         ("HU p1 / médiane / p99", "hu_p1_mediane_p99"), ("étiquettes de la vérité", "etiquettes_verite"),
                         ("étiquettes prédites", "etiquettes_prediction")):
        print(f"  {libelle:<44}{g[cle]}")
    for an in g["anomalies"] + r["anomalies_dice"]:
        print(f"  ANOMALIE : {an}")

    print(f"\n=== {r['cas']} — 2. Dice des 7 structures (recalculé | summary.json de nnU-Net, pli {r['pli']}) ===")
    print("    " + "".join(f"{COURTS[e]:>9}" for e in COURTS))
    print("    " + "".join(f"{bilan.f3(r['dice'][e]):>9}" for e in COURTS) + "   recalculé")
    print("    " + "".join(f"{bilan.f3(r['dice_summary'][e]):>9}" for e in COURTS) + "   summary.json")

    print(f"\n=== {r['cas']} — 3. structures examinées : vérité | prédiction ===")
    print("  (coupes comptées dans le repère RAS : coupe 0 = la plus basse ; niveaux en mm, plus grand = vers la tête)")
    for lab in structures:
        s = r["structures"][COURTS[lab]]
        print(f"  -- {lab} {NOMS[lab]} (décalage des centres de gravité : {ana.txt(s['decalage_centres_mm'], '{} mm')})")
        for libelle, extraire in ana.LIGNES_STRUCTURE:
            cellules = [(extraire(s[src]) if s[src]["voxels"] else ("absente" if libelle == "voxels" else "—"))
                        for src in ("verite", "prediction")]
            print(f"    {libelle:<48}{str(cellules[0]):<32}| {cellules[1]}")

    print(f"\n=== {r['cas']} — 4. erreurs : oublis (vérité non prédite) | ajouts (prédit hors vérité) ===")
    print(f"  (près = à {PRES_MM:g} mm ou moins de l'autre masque : liseré ; loin = à plus de {LOIN_MM:g} mm : "
          "morceau entier)")
    for lab in structures:
        s = r["structures"][COURTS[lab]]
        print(f"  -- {lab} {NOMS[lab]}")
        for libelle, extraire in LIGNES_ERREUR:
            cellules = [(extraire(s[t]) if s[t]["voxels"] else ("aucun" if libelle == "voxels" else "—"))
                        for t in ("oublis", "ajouts")]
            print(f"    {libelle:<52}{str(cellules[0]):<36}| {cellules[1]}")

    cm = r["confusion_verite_vers_prediction"]
    print(f"\n=== {r['cas']} — 5. matrice de confusion, en % de chaque ligne (0 fond, 1 myo VG, 2 OG, 3 VG, 4 OD, "
          "5 VD, 6 aorte, 7 AP) ===")
    print("    vérité (ligne) → prédiction (colonnes)" + "".join(f"{'→' + str(k):>7}" for k in range(8)))
    for lab in range(1, 8):
        tot = sum(cm[lab])
        marque = "  <==" if lab in structures else ""
        print(f"    vérité {lab} {NOMS[lab]:<30}" + ("".join(f"{100 * x / tot:7.1f}" for x in cm[lab]) if tot
                                                 else "   aucun voxel dans la vérité") + marque)

    print(f"\n=== {r['cas']} — 6. densités de la vérité (HU) ; « cœur » = voxels à plus de {ana.COEUR_MM:g} mm du bord ===")
    print(f"    {'région':<40}{'médiane':>8}{'p5':>7}{'p95':>7}{'σ':>7}{'méd. cœur':>10}{'σ cœur':>8}")
    for nom, h in r["intensites"].items():
        if h is None:
            print(f"    {nom:<40}région vide")
            continue
        print(f"    {nom:<40}{h['mediane']:>8}{h['p5']:>7}{h['p95']:>7}{h['ecart_type']:>7}"
              f"{ana.txt(h['coeur_mediane']):>10}{ana.txt(h['coeur_ecart_type']):>8}")


def repartition_txt(rep, lab):
    """« fond 40 % ; VD 35 % » : où vont les erreurs (hors la structure elle-même), 3 premières."""
    if not rep:
        return "—"
    tri = sorted(((e, x) for e, x in rep.items() if e != lab and x >= 1), key=lambda t: -t[1])[:3]
    return " ; ".join(f"{'fond' if e == 0 else COURTS[e]} {x:.0f} %" for e, x in tri) or "—"


LIGNES_ERREUR = (
    ("voxels", lambda m: m["voxels"]),
    ("part (de la vérité | de la prédiction)", lambda m: bilan.pct(m["part"])),
    ("où (oublis : prédits comme | ajouts : vérité =)", lambda m: repartition_txt(m.get("repartition_pct"), None)),
    ("distance à l'autre masque : médiane / p90 (mm)",
     lambda m: f"{m.get('distance_mediane_mm', '—')} / {m.get('distance_p90_mm', '—')}"),
    (f"% près (≤ {PRES_MM:g} mm) / % loin (> {LOIN_MM:g} mm)", lambda m: f"{m.get('pct_pres', '—')} / {m.get('pct_loin', '—')}"),
    ("% au-delà de la boîte de l'autre : haut/bas/côtés",
     lambda m: "{au_dessus} / {en_dessous} / {sur_les_cotes}".format(**m["au_dela_boite_pct"]) if "au_dela_boite_pct" in m else "—"),
    ("niveaux bas / haut (mm)", lambda m: f"{m['niveau_bas_mm']} / {m['niveau_haut_mm']}"),
    ("morceaux [3 plus gros, mL]", lambda m: f"{m['morceaux']} {m['morceaux_3_plus_gros_ml']}"),
    ("HU médiane", lambda m: m["hu_mediane"]))


# --- 8. Récapitulatif par hypothèse (le script ne conclut pas) --------------------------------
def ref_hu(I, lab):
    """Mesure de référence de 05e : médiane au cœur, sinon médiane de la région entière."""
    h = I.get(f"verite_{lab}_{NOMS[lab]}")
    if not h:
        return None
    return h["coeur_mediane"] if h["coeur_mediane"] is not None else h["mediane"]


def indicateurs(r, structures):
    S, I = r["structures"], r["intensites"]
    v = ana.val

    def diff(x, y):
        return None if x is None or y is None else round(x - y, 1)

    ind = {}
    for lab in structures:
        s = S[COURTS[lab]]
        ve, pr, ou, aj = s["verite"], s["prediction"], s["oublis"], s["ajouts"]
        ind[COURTS[lab]] = {
            "dice": None if r["dice"][lab] is None else round(r["dice"][lab], 3),
            "volume_predit_sur_verite_pct": (round(100 * pr["voxels"] / ve["voxels"]) if ve["voxels"] else None),
            "part_oublis_pct": None if ou.get("part") is None else round(100 * ou["part"], 1),
            "part_ajouts_pct": None if aj.get("part") is None else round(100 * aj["part"], 1),
            "oublis_predits_fond_pct": v(ou, "repartition_pct", 0),
            "oublis_predits_autre": repartition_txt({e: x for e, x in (ou.get("repartition_pct") or {}).items() if e}, lab),
            "ajouts_sur_fond_pct": v(aj, "repartition_pct", 0),
            "ajouts_sur_autre": repartition_txt({e: x for e, x in (aj.get("repartition_pct") or {}).items() if e}, lab),
            "oublis_pct_pres": ou.get("pct_pres"), "oublis_pct_loin": ou.get("pct_loin"),
            "ajouts_pct_pres": aj.get("pct_pres"), "ajouts_pct_loin": aj.get("pct_loin"),
            "oublis_au_dessus_pct": v(ou, "au_dela_boite_pct", "au_dessus"),
            "oublis_en_dessous_pct": v(ou, "au_dela_boite_pct", "en_dessous"),
            "oublis_cotes_pct": v(ou, "au_dela_boite_pct", "sur_les_cotes"),
            "ajouts_au_dessus_pct": v(aj, "au_dela_boite_pct", "au_dessus"),
            "ajouts_en_dessous_pct": v(aj, "au_dela_boite_pct", "en_dessous"),
            "ajouts_cotes_pct": v(aj, "au_dela_boite_pct", "sur_les_cotes"),
            "haut_predit_moins_haut_verite_mm": diff(v(pr, "niveau_haut_mm"), v(ve, "niveau_haut_mm")),
            "bas_predit_moins_bas_verite_mm": diff(v(pr, "niveau_bas_mm"), v(ve, "niveau_bas_mm")),
            "morceaux_verite_prediction": f"{ve.get('morceaux', 0)} / {pr.get('morceaux', 0)}",
            "hu_oublis_moins_hu_verite": diff(ou.get("hu_mediane"), v(I, f"verite_{lab}_{NOMS[lab]}", "mediane"))}
    ao, ap, vg, vd = (ref_hu(I, l) for l in (6, 7, 3, 5))
    rapport = None if ao is None or not ap else round(ao / ap, 2)
    vgI, myo = I.get("verite_3_cavite_VG") or {}, I.get("verite_1_myocarde_VG") or {}
    bruit = vgI.get("coeur_ecart_type")
    ind["examen"] = {
        "epaisseur_coupe_mm": r["geometrie"]["epaisseur_coupe_mm"],
        "etendue_haut_bas_mm": r["geometrie"]["etendue_mm_RL_AP_IS"][2],
        "coeur_mm_au_bord_haut": v(S, "coeur_entier", "verite", "bord_haut_mm"),
        "coeur_mm_au_bord_bas": v(S, "coeur_entier", "verite", "bord_bas_mm"),
        "rapport_aorte_sur_AP": rapport,
        "phase": (None if rapport is None else "INVERSÉ" if rapport < 1 else "proche" if rapport < SEUIL_PROCHE
                  else "habituelle"),
        "hu_aorte_moins_VG": diff(ao, vg), "hu_AP_moins_VD": diff(ap, vd),
        "bruit_sigma_coeur_VG": bruit,
        "cnr_VG_myocarde": (None if vgI.get("coeur_mediane") is None or myo.get("mediane") is None or not bruit
                            else round((vgI["coeur_mediane"] - myo["mediane"]) / bruit, 1))}
    return ind


# Pour chaque indicateur : hypothèse, libellé, et ce qu'une valeur inhabituelle suggérerait (sans conclure).
LECTURE_STRUCTURE = (
    ("—", "dice", "Dice", "repère : comparer aux cas de comparaison"),
    ("—", "volume_predit_sur_verite_pct", "volume prédit / volume vérité (%)",
     "< 100 : le modèle en met moins que l'annotation ; > 100 : plus"),
    ("—", "part_oublis_pct", "% de la vérité oubliée", "grand : voir où vont les oublis"),
    ("—", "part_ajouts_pct", "% de la prédiction ajoutée", "grand : voir d'où viennent les ajouts"),
    ("(a)", "oublis_predits_autre", "oublis prédits comme (autre structure)",
     "une structure voisine domine : la frontière entre les deux est placée ailleurs (ou confusion, voir (c))"),
    ("(a)", "ajouts_sur_autre", "ajouts posés sur (autre structure de la vérité)",
     "une structure voisine domine : frontière déplacée vers elle"),
    ("(a)", "oublis_pct_pres", f"% des oublis à ≤ {PRES_MM:g} mm de la prédiction",
     "grand : liseré le long du bord, désaccord de frontière plutôt que structure manquée"),
    ("(a)", "ajouts_pct_pres", f"% des ajouts à ≤ {PRES_MM:g} mm de la vérité", "grand : liseré, désaccord de frontière"),
    ("(b)", "oublis_pct_loin", f"% des oublis à > {LOIN_MM:g} mm de la prédiction",
     "grand : des morceaux entiers de la vérité ne sont pas prédits (étendue plus grande dans l'annotation, ex. branches)"),
    ("(b)", "ajouts_pct_loin", f"% des ajouts à > {LOIN_MM:g} mm de la vérité",
     "grand : le modèle prolonge la structure plus loin que l'annotation"),
    ("(b)", "oublis_au_dessus_pct", "% des oublis au-dessus de la boîte prédite",
     "grand : la vérité monte plus haut que la prédiction"),
    ("(b)", "oublis_en_dessous_pct", "% des oublis en dessous de la boîte prédite",
     "grand : la vérité descend plus bas que la prédiction"),
    ("(b)", "oublis_cotes_pct", "% des oublis sur les côtés de la boîte prédite",
     "grand : la vérité s'étend plus loin latéralement (ex. branches droite/gauche de l'AP)"),
    ("(b)", "ajouts_au_dessus_pct", "% des ajouts au-dessus de la boîte de la vérité", "grand : le modèle monte plus haut"),
    ("(b)", "ajouts_en_dessous_pct", "% des ajouts en dessous de la boîte de la vérité", "grand : le modèle descend plus bas"),
    ("(b)", "ajouts_cotes_pct", "% des ajouts sur les côtés de la boîte de la vérité",
     "grand : le modèle s'étend plus loin latéralement"),
    ("(b)", "haut_predit_moins_haut_verite_mm", "haut prédit − haut vérité (mm)",
     "loin de 0 : la prédiction s'arrête plus haut (+) ou plus bas (−) que l'annotation"),
    ("(b)", "bas_predit_moins_bas_verite_mm", "bas prédit − bas vérité (mm)",
     "loin de 0 : limite basse différente (ex. niveau de la valve)"),
    ("(c)", "oublis_predits_fond_pct", "% des oublis prédits « fond »",
     "grand et loin : morceau non reconnu du tout (structure non apprise sous cette forme)"),
    ("(c)", "ajouts_sur_fond_pct", "% des ajouts posés sur le fond de la vérité",
     "grand : le modèle voit la structure là où rien n'est annoté (vaisseau voisin, branche non annotée ?)"),
    ("(c)", "morceaux_verite_prediction", "morceaux vérité / prédiction",
     "très différents : l'une des deux est fragmentée"),
    ("(d)", "hu_oublis_moins_hu_verite", "HU des oublis − HU de la vérité",
     "très négatif : les oublis sont dans une zone peu rehaussée (sombre)"))

LECTURE_EXAMEN = (
    ("epaisseur_coupe_mm", "épaisseur de coupe (mm)", "famille d'examens : comparer avec des cas de la même famille"),
    ("etendue_haut_bas_mm", "étendue du champ, bas-haut (mm)", "très différente : cadrage inhabituel"),
    ("coeur_mm_au_bord_haut", "cœur (vérité) : mm jusqu'au bord haut", "petit : structures hautes coupées par le champ ?"),
    ("coeur_mm_au_bord_bas", "cœur (vérité) : mm jusqu'au bord bas", "petit : peu de marge sous le cœur"),
    ("rapport_aorte_sur_AP", "HU aorte / HU AP (référence de 05e)", "< 1 : phase inversée (comme 1019) ; < 1,20 : proche"),
    ("phase", "phase d'injection (repère de 05e)", "INVERSÉ ou proche : examen hors de l'habitude, voir D27"),
    ("hu_aorte_moins_VG", "HU aorte − HU cavité VG", "loin de 0 : aorte annotée ≠ sang du VG"),
    ("hu_AP_moins_VD", "HU AP − HU VD", "loin de 0 : AP annotée ≠ sang du VD (mélange, ou frontière VD/AP ailleurs ?)"),
    ("bruit_sigma_coeur_VG", "bruit : σ des HU au cœur du VG", "nettement plus grand : image bruitée"),
    ("cnr_VG_myocarde", "CNR = (VG − myocarde) / bruit", "nettement plus petit : bords peu visibles"))


def recapitulatif(resultats, structures):
    tous = [(r["cas"], indicateurs(r, structures)) for r in resultats]
    print("\n" + "=" * 110)
    print(f"8. RÉCAPITULATIF — {tous[0][0]} (cas étudié) comparé à {', '.join(c for c, _ in tous[1:]) or 'aucun cas'}")
    print("Le script NE CONCLUT PAS. Hypothèses : (a) frontière, (b) étendue, (c) confusion ou oubli franc, "
          "(d) examen inhabituel. Croiser avec les figures avant de décider.")
    for lab in structures:
        print(f"\n-- {lab} {NOMS[lab]}")
        for h, cle, libelle, si in LECTURE_STRUCTURE:
            valeurs = " | ".join(f"{c[-4:]} : {ana.txt(ind[COURTS[lab]][cle])}" for c, ind in tous)
            print(f"  {h:<4}{libelle:<52}{valeurs}")
            print(f"        si {si}")
    print("\n(d) EXAMEN")
    for cle, libelle, si in LECTURE_EXAMEN:
        valeurs = " | ".join(f"{c[-4:]} : {ana.txt(ind['examen'][cle])}" for c, ind in tous)
        print(f"  {libelle:<56}{valeurs}")
        print(f"        si {si}")
    print("\nPoints de repère pour lire ce tableau :")
    print("  - oublis de l'une prédits comme l'autre ET ajouts de l'autre posés sur l'une (ex. VD et AP croisés) : frontière (a),")
    print("    MÊME si ces erreurs sont « loin » ou au-delà de la boîte (frontière décalée de plusieurs mm) ;")
    print("  - oublis « loin », au-delà de la boîte prédite, prédits « fond » : la vérité s'étend plus loin (b) ;")
    print("  - ajouts surtout « loin », posés sur le fond : le modèle prolonge plus loin que l'annotation (b) ;")
    print("  - morceau entier pris pour une autre structure : confusion (c) ;")
    print("  - phase INVERSÉ / proche, bruit fort, CNR faible ou champ inhabituel : examen (d) ;")
    print(f"  - repères {PRES_MM:g} et {LOIN_MM:g} mm et 1,20 indicatifs, non validés. Toujours comparer à un cas réussi de la "
          "même épaisseur de coupe.")
    return {c: ind for c, ind in tous}


# --- Programme principal ----------------------------------------------------------------------
def main():
    a = lire_arguments()
    ds, sortie, chemins, cas_entr, cas_test = verifier(a)
    champs = ana.champs_entrainement(ds, cas_entr)          # en-têtes seuls des 15 cas d'entraînement
    os.makedirs(sortie, exist_ok=True)                      # le seul dossier où le script écrit
    structures = a.structures
    resultats, figures = [], {}
    for n, c in chemins.items():
        print(f"\n{'#' * 110}\n# mmwhs_{n} : image et vérité de imagesTr / labelsTr, prédiction de "
              f"fold_{c['pli']}/validation\n{'#' * 110}", flush=True)
        ct, ver, pred, affine, esp, geo = ana.charger(c)
        d = dices(ver, pred)
        d_summary, anomalies = comparer_summary(n, c, d, cas_test)
        if structures is None:                              # défaut : les 2 plus faibles du cas étudié
            structures = sorted(sorted((e for e in d if d[e] is not None), key=lambda e: d[e])[:2])
            print(f"Structures examinées (les 2 au Dice le plus bas sur mmwhs_{n}) : "
                  + ", ".join(f"{e} {NOMS[e]}" for e in structures))
        r = analyser(n, c, structures, ct, ver, pred, affine, esp, geo)
        r.update({"dice": d, "dice_summary": d_summary, "anomalies_dice": anomalies})
        afficher_cas(r, structures)
        figures[r["cas"]] = os.path.join(sortie, NOM_FIGURE.format(n=n))
        figure(n, r, ct, ver, pred, esp, structures, figures[r["cas"]])
        print(f"\nFigure : {figures[r['cas']]}", flush=True)
        del ct, ver, pred                                   # un seul cas en mémoire à la fois
        resultats.append(r)
    ana.afficher_champs(champs, {r["cas"] for r in resultats})
    ind = recapitulatif(resultats, structures)

    synthese = {"objet": f"examen de mmwhs_{a.cas[0]} (validation hors pli, PAS le test), LECTURE SEULE",
                "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "structures": structures, "trainer": ana.TRAINER, "configuration": ana.CONFIGURATION,
                "split_sha256": prep.sha256_fichier(a.split), "code_commit": prep.commit_git(),
                "reperes_indicatifs": {"pres_mm": PRES_MM, "loin_mm": LOIN_MM, "rapport_aorte_AP_proche": SEUIL_PROCHE},
                "sources": chemins, "champs_entrainement": champs, "cas": resultats, "indicateurs": ind,
                "figures": figures,
                "lecture": {"structure": {cle: {"hypothese": h, "libelle": l, "si": s} for h, cle, l, s in LECTURE_STRUCTURE},
                            "examen": {cle: {"libelle": l, "si": s} for cle, l, s in LECTURE_EXAMEN}}}
    chemin = os.path.join(sortie, NOM_SYNTHESE)
    with open(chemin + ".tmp", "w", encoding="utf-8") as f:
        json.dump(synthese, f, indent=2, ensure_ascii=False)
    os.replace(chemin + ".tmp", chemin)                     # jamais de fichier à moitié écrit
    print(f"\nSynthèse : {chemin}")
    print("Rien n'a été écrit dans nnUNet_raw, nnUNet_preprocessed ni nnUNet_results ; aucun cas de test lu.")


if __name__ == "__main__":
    main()
