"""
Étape 2 — Vérifier les données téléchargées.

Ce que fait le script :
  1. cherche les scanners (*_image.nii.gz) et masques (*_label.nii.gz) MM-WHS CT dans le dossier donné ;
  2. vérifie que chaque cas du découpage figé (splits/mmwhs_split.json) a bien son scanner et son masque ;
  3. affiche, pour chaque cas, la taille de l'image (en voxels) et l'espacement des voxels (en mm),
     et les valeurs d'étiquettes présentes dans le masque.

Entrée  : --donnees  dossier où MM-WHS a été téléchargé (dans Google Drive)
Sortie  : un tableau affiché à l'écran ; aucun fichier n'est modifié.

Vocabulaire :
  - voxel : le « pixel en 3D », un petit cube de l'image ;
  - espacement : la taille réelle d'un voxel en millimètres (ex. 0,44 x 0,44 x 0,625 mm) ;
  - étiquette : le nombre qui code une structure dans le masque (ex. 500 = ventricule gauche dans MM-WHS).
"""
import argparse, glob, json, os
import nibabel as nib
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--donnees", required=True)
p.add_argument("--split", default=os.path.join(os.path.dirname(__file__), "..", "splits", "mmwhs_split.json"))
a = p.parse_args()

split = json.load(open(a.split))
cas = split["entrainement_validation_croisee"] + split["test_sous_cle"]

def trouver(nom, suffixe):
    f = glob.glob(os.path.join(a.donnees, "**", f"{nom}_{suffixe}.nii.gz"), recursive=True)
    return f[0] if f else None

manquants = 0
print(f"{'cas':<15}{'lot':<8}{'taille (voxels)':<20}{'espacement (mm)':<24}étiquettes")
for c in cas:
    lot = "TEST" if c in split["test_sous_cle"] else "entr."
    img, lab = trouver(c, "image"), trouver(c, "label")
    if not img or not lab:
        print(f"{c:<15}{lot:<8}MANQUANT (image={bool(img)}, masque={bool(lab)})")
        manquants += 1
        continue
    im = nib.load(img)
    etiquettes = np.unique(np.asarray(nib.load(lab).dataobj)).astype(int).tolist()
    esp = tuple(round(float(x), 3) for x in im.header.get_zooms()[:3])
    print(f"{c:<15}{lot:<8}{str(im.shape):<20}{str(esp):<24}{etiquettes}")

print()
print("Résultat :", "CONFORME — 20 cas complets" if manquants == 0 else f"NON CONFORME — {manquants} cas incomplets")
