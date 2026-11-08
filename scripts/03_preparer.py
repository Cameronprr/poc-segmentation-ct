"""
Étape 3 — Contrôle qualité détaillé et conversion au format nnU-Net v2.

Ce que fait le script :
  1. charge chaque cas MM-WHS (scanner + masque) avec MONAI, UN cas à la fois (mémoire limitée dans Colab) ;
  2. contrôle chaque cas : présence des deux fichiers, même forme, même affine, espacement, orientation,
     étiquettes présentes ;
  3. cherche des doublons entre le lot de test et le lot d'entraînement : doublon exact (empreinte)
     = bloquant ; quasi-doublon (miniature) = simple avertissement, à examiner à la main ;
  4. écrit le rapport de contrôle `rapport_qc_etape3.csv` dans --sortie ;
  5. si tout est conforme, et seulement dans ce cas, écrit le jeu au format nnU-Net v2 dans
     --sortie/Dataset101_MMWHS/ (images copiées à l'identique, masques renumérotés 0..7, dataset.json).
  Le découpage figé (splits/mmwhs_split.json) est seulement LU, jamais modifié.

Entrées :
  --donnees     dossier où MM-WHS a été téléchargé (dans Google Drive), cherché récursivement
  --sortie      dossier nnUNet_raw dans Google Drive (le rapport et le jeu nnU-Net y sont écrits)
  --split       fichier de découpage (défaut : ../splits/mmwhs_split.json à côté du script)
  --id-dataset  numéro du jeu pour nnU-Net (défaut : 101 → dossier Dataset101_MMWHS)
  --ecraser     autorise à remplacer un dossier Dataset déjà rempli (sinon le script s'arrête)
  --seuil-quasi-doublon  corrélation au-delà de laquelle un cas de test est signalé comme
                quasi-doublon d'un cas d'entraînement (défaut : 0.98 ; avertissement seulement)
Sorties (uniquement dans --sortie, jamais dans le dépôt) :
  rapport_qc_etape3.csv ; Dataset101_MMWHS/{imagesTr, labelsTr, imagesTs, labelsTs, dataset.json}
Codes de sortie :
  0 = conforme et converti ;
  1 = NON CONFORME, découpage incohérent ou --sortie placé dans le dépôt :
      rien n'a été écrit au format nnU-Net ;
  2 = le dossier Dataset existe déjà (relancer avec --ecraser) ;
  3 = ERREUR PENDANT LA CONVERSION : le dossier Dataset est incomplet,
      il faut le relancer avec --ecraser après avoir corrigé la cause.

Vocabulaire :
  - format nnU-Net : l'organisation de dossiers et de noms de fichiers qu'attend nnU-Net v2.
    imagesTr = scanners d'entraînement, labelsTr = leurs masques, imagesTs = scanners de test.
    Le suffixe « _0000 » d'une image désigne le canal n°0 (ici un seul canal : le CT) ;
  - dataset.json : petit fichier qui décrit le jeu à nnU-Net (canaux, nom des étiquettes, nombre de cas) ;
  - affine : matrice 4x4 qui relie chaque voxel à sa position réelle dans le corps du patient (en mm).
    Image et masque doivent avoir la même, sinon le masque est « décalé » par rapport au scanner ;
  - orientation : sens des axes de l'image, résumé par 3 lettres. Chaque lettre indique vers où pointe
    l'axe dans le corps du patient : « RAS » = 1er axe vers la droite (Right), 2e vers l'avant
    (Anterior), 3e vers le haut (Superior). (Les logiciels DICOM/ITK utilisent une autre convention,
    « LPS » : ne pas mélanger les deux.) ;
  - doublon : le même scanner présent deux fois. Un doublon entre test et entraînement fausserait le
    verdict (le modèle aurait « vu » la copie de l'examen). Même taille et même espacement ne suffisent
    PAS à prouver un doublon : beaucoup de scanners partagent ces réglages ;
  - empreinte (hash SHA-256) : une suite de 64 caractères calculée à partir de tous les voxels.
    Deux volumes identiques au voxel près ont la même empreinte ; la moindre différence la change ;
  - miniature : le volume réduit à 64x64x64 voxels, pour repérer des volumes presque identiques
    (ex. même scanner ré-enregistré avec un léger changement) via une corrélation.
    Avant réduction, les intensités sont écrêtées entre -200 et 800 HU ;
  - HU (unités Hounsfield) : l'échelle des intensités CT (air ≈ -1000, eau = 0, os > 400).
    Écrêter = ramener toute valeur hors de l'intervalle à sa borne. Cela supprime le « disque » de
    remplissage (-2048 ou -3024) présent autour de l'image dans beaucoup de scanners : sinon ce disque,
    commun à tous les examens, ferait croire que deux patients différents se ressemblent.
"""
import argparse, csv, glob, hashlib, json, os, shutil, sys
import nibabel as nib
import numpy as np
from monai.transforms import LoadImage, Resize

# --- Paramètres fixes -------------------------------------------------------------------------
# Table de correspondance : étiquette MM-WHS d'origine → étiquette nnU-Net (0..7, consécutives).
CORRESPONDANCE = {
    0: 0,    # fond
    205: 1,  # myocarde du ventricule gauche (VG)
    420: 2,  # oreillette gauche
    500: 3,  # cavité du ventricule gauche
    550: 4,  # oreillette droite
    600: 5,  # ventricule droit
    820: 6,  # aorte ascendante
    850: 7,  # artère pulmonaire
}
# Noms des étiquettes pour dataset.json (sans espaces ni accents, comme le veut nnU-Net).
NOMS = {"background": 0, "myocarde_VG": 1, "oreillette_gauche": 2, "cavite_VG": 3,
        "oreillette_droite": 4, "ventricule_droit": 5, "aorte_ascendante": 6, "artere_pulmonaire": 7}
TOLERANCE_AFFINE = 1e-3   # écart toléré entre l'affine de l'image et celle du masque
TAILLE_MINIATURE = (64, 64, 64)
FENETRE_HU = (-200, 800)  # écrêtage avant miniature : tissus mous, cavités injectées, os
# Le seuil de quasi-doublon (corrélation de Pearson entre miniatures, 1 = identiques) est un paramètre
# (--seuil-quasi-doublon). Deux scanners cardiaques différents se ressemblent (même anatomie, même
# cadrage) : leur corrélation peut être élevée. Le dépasser ne bloque pas : c'est un signal à vérifier.

# --- Arguments --------------------------------------------------------------------------------
depot = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
p = argparse.ArgumentParser()
p.add_argument("--donnees", required=True)
p.add_argument("--sortie", required=True)
p.add_argument("--split", default=os.path.join(depot, "splits", "mmwhs_split.json"))
p.add_argument("--id-dataset", type=int, default=101)
p.add_argument("--ecraser", action="store_true")
p.add_argument("--seuil-quasi-doublon", type=float, default=0.98)
a = p.parse_args()

sortie = os.path.abspath(a.sortie)
if os.path.commonpath([sortie, depot]) == depot:
    sys.exit("ARRÊT : --sortie est dans le dépôt. Les données doivent rester dans Google Drive.")
dossier_ds = os.path.join(sortie, f"Dataset{a.id_dataset:03d}_MMWHS")
# Vérification faite dès le début pour ne pas attendre la fin du contrôle avant de s'arrêter.
if os.path.isdir(dossier_ds) and os.listdir(dossier_ds) and not a.ecraser:
    print(f"ARRÊT : {dossier_ds} existe déjà et n'est pas vide. Rien n'a été écrasé.")
    print("Relancer avec --ecraser pour le remplacer volontairement.")
    sys.exit(2)

with open(a.split, encoding="utf-8") as f:          # lecture seule du découpage figé
    split = json.load(f)
cas_tr, cas_ts = split["entrainement_validation_croisee"], split["test_sous_cle"]

# Cohérence du découpage : un cas présent dans les deux lots fuirait du test vers l'entraînement.
communs = set(cas_tr) & set(cas_ts)
if communs or len(set(cas_tr)) != 15 or len(set(cas_ts)) != 5 \
        or len(cas_tr) != len(set(cas_tr)) or len(cas_ts) != len(set(cas_ts)):
    sys.exit(f"ARRÊT : découpage incohérent (cas dans les deux lots : {sorted(communs)}, "
             f"entraînement = {len(cas_tr)} cas, test = {len(cas_ts)} cas, attendu 15 et 5, sans répétition).")

def trouver(nom, suffixe):
    """Recherche récursive dans --donnees. Renvoie la LISTE triée des fichiers trouvés :
    0 = manquant, 1 = normal, plus de 1 = ambigu (ex. archive décompressée deux fois)."""
    return sorted(glob.glob(os.path.join(a.donnees, "**", f"{nom}_{suffixe}.nii.gz"), recursive=True))

# dtype=None : on garde le type d'origine des voxels (pas de conversion), utile pour l'empreinte.
charger = LoadImage(image_only=False, dtype=None)
reduire = Resize(spatial_size=TAILLE_MINIATURE, mode="area")  # "area" = moyenne locale, sans crénelage

def lire(chemin):
    """Renvoie (tableau numpy, affine 4x4) via MONAI."""
    donnees, meta = charger(chemin)
    affine = np.array(meta.get("original_affine", meta["affine"]), dtype=float)
    return np.asarray(donnees), affine

def empreinte(tableau):
    """SHA-256 des voxels (avec forme et type, pour ne pas confondre deux volumes de formes différentes)."""
    h = hashlib.sha256(f"{tableau.shape}{tableau.dtype}".encode())
    h.update(np.ascontiguousarray(tableau).tobytes())
    return h.hexdigest()

def miniature(tableau):
    """Volume écrêté (FENETRE_HU), réduit à 64x64x64, puis normalisé (z-score : moyenne 0, écart-type 1)."""
    ecrete = np.clip(tableau, *FENETRE_HU).astype(np.float32)
    m = np.asarray(reduire(ecrete[None]))[0].ravel()   # [None] = ajout d'un canal
    return (m - m.mean()) / (m.std() + 1e-8)

# --- 1. Contrôle qualité, un cas à la fois ----------------------------------------------------
attendues = set(CORRESPONDANCE)
lignes, bloquants, avertissements, minis = [], [], [], {}
fichiers = {}   # cas → (image, masque) retenus, réutilisés tels quels pour la conversion
for c in cas_tr + cas_ts:
    lot = "TEST" if c in cas_ts else "entr."
    imgs, labs = trouver(c, "image"), trouver(c, "label")
    l = {"cas": c, "lot": lot, "image": len(imgs), "masque": len(labs)}   # nombre de fichiers trouvés
    if len(imgs) != 1 or len(labs) != 1:
        if not imgs or not labs:
            bloquants.append(f"{c} : fichier manquant (images trouvées={len(imgs)}, masques={len(labs)})")
        if len(imgs) > 1 or len(labs) > 1:
            bloquants.append(f"{c} : plusieurs fichiers pour le même cas, impossible de choisir : "
                             f"{imgs + labs}")
        lignes.append(l)
        continue
    img, lab = imgs[0], labs[0]
    fichiers[c] = (img, lab)
    im, aff_im = lire(img)
    ma, aff_ma = lire(lab)
    valeurs = set(np.unique(ma).tolist())
    inattendues = sorted(valeurs - attendues)
    absentes = sorted(attendues - valeurs)
    espacement = np.sqrt((aff_im[:3, :3] ** 2).sum(axis=0))   # longueur de chaque colonne de l'affine
    l.update({
        "forme_image": im.shape, "forme_masque": ma.shape,
        "meme_forme": im.shape == ma.shape,
        "meme_affine": bool(np.allclose(aff_im, aff_ma, atol=TOLERANCE_AFFINE)),
        "espacement_mm": tuple(round(float(x), 3) for x in espacement),
        "orientation": "".join(nib.aff2axcodes(aff_im)),
        "etiquettes": sorted(int(v) if float(v).is_integer() else v for v in valeurs),
        "inattendues": inattendues, "absentes": absentes,
        "hash_image": empreinte(im), "hash_masque": empreinte(ma),
    })
    if not l["meme_forme"]:
        bloquants.append(f"{c} : forme image {im.shape} != forme masque {ma.shape}")
    if not l["meme_affine"]:
        bloquants.append(f"{c} : affine de l'image différente de celle du masque")
    if inattendues:
        bloquants.append(f"{c} : étiquettes inattendues {inattendues}")
    if absentes:
        avertissements.append(f"{c} : étiquettes absentes {absentes}")
    minis[c] = miniature(im)
    lignes.append(l)
    del im, ma   # libère la mémoire avant le cas suivant

# --- 2. Doublons : chaque cas de test contre chaque cas d'entraînement ------------------------
par_cas = {l["cas"]: l for l in lignes}
for t in cas_ts:
    if t not in minis:
        continue
    meilleur, r_max = None, -1.0
    for e in cas_tr:
        if e not in minis:
            continue
        if par_cas[t]["hash_image"] == par_cas[e]["hash_image"]:
            bloquants.append(f"DOUBLON EXACT (image) : {t} (test) = {e} (entraînement)")
        if par_cas[t]["hash_masque"] == par_cas[e]["hash_masque"]:
            bloquants.append(f"DOUBLON EXACT (masque) : {t} (test) = {e} (entraînement)")
        r = float(np.corrcoef(minis[t], minis[e])[0, 1])   # corrélation de Pearson
        if r > a.seuil_quasi_doublon:   # signal à vérifier à l'œil (ouvrir les deux scanners), pas un arrêt
            avertissements.append(f"QUASI-DOUBLON possible : {t} (test) ~ {e} (entraînement), "
                                  f"corrélation {r:.4f} > {a.seuil_quasi_doublon}")
        if r > r_max:
            meilleur, r_max = e, r
    par_cas[t]["plus_proche_entr"], par_cas[t]["correlation"] = meilleur, round(r_max, 4)

# --- 3. Rapport : tableau affiché + CSV -------------------------------------------------------
print(f"{'cas':<15}{'lot':<7}{'forme':<17}{'espacement (mm)':<23}{'orient.':<8}"
      f"{'forme=':<7}{'affine=':<8}étiquettes")
for l in lignes:
    if "forme_image" not in l:
        print(f"{l['cas']:<15}{l['lot']:<7}MANQUANT OU AMBIGU (images={l['image']}, masques={l['masque']})")
        continue
    print(f"{l['cas']:<15}{l['lot']:<7}{str(l['forme_image']):<17}{str(l['espacement_mm']):<23}"
          f"{l['orientation']:<8}{'oui' if l['meme_forme'] else 'NON':<7}"
          f"{'oui' if l['meme_affine'] else 'NON':<8}{l['etiquettes']}")

print(f"\nRessemblance des cas de test (seuil de quasi-doublon : {a.seuil_quasi_doublon}) :")
for t in cas_ts:
    if "plus_proche_entr" in par_cas.get(t, {}):
        print(f"  {t} : cas d'entraînement le plus ressemblant = {par_cas[t]['plus_proche_entr']}"
              f" (corrélation {par_cas[t]['correlation']})")

os.makedirs(sortie, exist_ok=True)
chemin_csv = os.path.join(sortie, "rapport_qc_etape3.csv")
colonnes = ["cas", "lot", "image", "masque", "forme_image", "forme_masque", "meme_forme", "meme_affine",
            "espacement_mm", "orientation", "etiquettes", "inattendues", "absentes",
            "hash_image", "hash_masque", "plus_proche_entr", "correlation"]
with open(chemin_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=colonnes)
    w.writeheader()
    w.writerows(lignes)
print(f"\nRapport écrit : {chemin_csv}")

for m in avertissements:
    print("ATTENTION (non bloquant) :", m)
if bloquants:
    print("\nRésultat : NON CONFORME — le jeu nnU-Net N'A PAS été écrit. Anomalies :")
    for m in bloquants:
        print("  -", m)
    sys.exit(1)
print("\nRésultat du contrôle : CONFORME")

# --- 4. Conversion au format nnU-Net v2 -------------------------------------------------------
def convertir_masque(source, destination):
    """Renumérote le masque (table CORRESPONDANCE) et l'enregistre en uint8, même affine et en-tête."""
    orig = nib.load(source)
    donnees = np.asarray(orig.dataobj)
    nouveau = np.zeros(donnees.shape, dtype=np.uint8)
    for ancienne, neuve in CORRESPONDANCE.items():
        nouveau[donnees == ancienne] = neuve
    entete = orig.header.copy()
    entete.set_data_dtype(np.uint8)
    sortie_nii = nib.Nifti1Image(nouveau, orig.affine, entete)
    sortie_nii.header.set_slope_inter(1, 0)   # pas de mise à l'échelle : les valeurs sont écrites telles quelles
    nib.save(sortie_nii, destination)
    # Vérification après coup : on relit le fichier écrit.
    ecrites = set(np.unique(np.asarray(nib.load(destination).dataobj)).tolist())
    prevues = {CORRESPONDANCE[v] for v in np.unique(donnees).tolist()}
    if ecrites != prevues or not ecrites <= set(range(8)):
        raise RuntimeError(f"{destination} contient {sorted(ecrites)} au lieu de {sorted(prevues)}")
    return ecrites

description = {
    "channel_names": {"0": "CT"},   # "CT" : nnU-Net applique sa normalisation propre aux scanners
    "labels": NOMS,
    "numTraining": len(cas_tr),
    "file_ending": ".nii.gz",
    "name": "MMWHS",
    "description": "MM-WHS CT, 20 cas annotés, 7 structures cardiaques. labelsTs/ (non lu par nnU-Net) "
                   "est réservé à l'évaluation finale de l'étape 6.",
    "reference": "Zhuang et al., Multi-Modality Whole Heart Segmentation (MM-WHS), "
                 "https://zmiclab.github.io/zxh/0/mmwhs/",
    "licence": "Usage personnel non commercial, source citée",
    "correspondance_etiquettes_origine": {str(k): v for k, v in CORRESPONDANCE.items()},
}

# Toute erreur pendant l'écriture (suppression de l'ancien dossier, création des sous-dossiers, copie,
# conversion, vérification, dataset.json) → code de sortie 3, distinct du code 1 (NON CONFORME),
# pour savoir tout de suite que le dossier Dataset est à moitié rempli.
etape = "préparation du dossier"
try:
    if os.path.isdir(dossier_ds) and a.ecraser:
        shutil.rmtree(dossier_ds)   # --ecraser demandé explicitement : on repart d'un dossier propre
    for sous in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
        os.makedirs(os.path.join(dossier_ds, sous), exist_ok=True)
    for c in cas_tr + cas_ts:
        etape = c
        n = c.split("_")[-1]   # ex. "ct_train_1001" → "1001"
        images, masques = ("imagesTr", "labelsTr") if c in cas_tr else ("imagesTs", "labelsTs")
        img, lab = fichiers[c]   # les fichiers contrôlés plus haut, pas une nouvelle recherche
        shutil.copyfile(img, os.path.join(dossier_ds, images, f"mmwhs_{n}_0000.nii.gz"))
        ecrites = convertir_masque(lab, os.path.join(dossier_ds, masques, f"mmwhs_{n}.nii.gz"))
        print(f"  {c} → {images}/mmwhs_{n}_0000.nii.gz + {masques}/mmwhs_{n}.nii.gz  étiquettes {sorted(ecrites)}")
    etape = "écriture de dataset.json"
    with open(os.path.join(dossier_ds, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(description, f, indent=2, ensure_ascii=False)
except Exception as err:
    print(f"\nERREUR PENDANT LA CONVERSION ({etape}) : {err}")
    print(f"Le dossier {dossier_ds} est INCOMPLET. Corriger la cause, puis relancer avec --ecraser.")
    sys.exit(3)

# --- 5. Récapitulatif -------------------------------------------------------------------------
print(f"\nJeu nnU-Net écrit : {dossier_ds}")
print(f"  imagesTr/labelsTr : {len(cas_tr)} cas (entraînement + validation croisée)")
print(f"  imagesTs/labelsTs : {len(cas_ts)} cas (test sous clé)")
print("Rappel : le lot de test ne sera utilisé qu'à l'étape 6, une seule fois.")
