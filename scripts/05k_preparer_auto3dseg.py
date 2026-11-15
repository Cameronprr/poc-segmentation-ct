"""
Étape 5k — Préparer Auto3DSeg (MONAI) : liste de données des 15 cas, avec EXACTEMENT les plis de nnU-Net (D37).

Ce que fait le script :
  1. garde-fous : la sortie est hors du dépôt ; le découpage figé (splits/mmwhs_split.json) est LU SEULEMENT ;
     splits_final.json est bien celui écrit par 05a (mêmes contrôles que 05b, fonctions réutilisées :
     marqueur verification_05a.json, empreinte notée dans la fiche splits_final_05a.json, plis recalculés
     depuis le découpage figé, la graine et les épaisseurs de coupe lues dans les en-têtes de imagesTr) ;
  2. liste les fichiers de nnUNet_raw/Dataset101_MMWHS/imagesTr et labelsTr UNIQUEMENT (jamais imagesTs ni
     labelsTs, D18) : il faut exactement les 15 cas d'entraînement du découpage figé ;
  3. construit la liste de données (datalist) au format d'Auto3DSeg : une seule clé « training », un élément
     par cas : {"fold": k, "image": "imagesTr/<cas>_0000.nii.gz", "label": "labelsTr/<cas>.nii.gz"}.
     Le pli k d'un cas = le numéro du pli de splits_final.json dont la liste « val » contient ce cas ;
  4. contrôle la liste (controler_liste), puis la relit sur le disque et la recontrôle. Tout écart arrête le
     script (code 4) : 15 cas exactement, chacun dans un seul pli de validation, mêmes ensembles
     entraînement / validation que splits_final.json pour chacun des 5 plis, paires {1001, 1020} et
     {1005, 1012} dans le même pli, aucun cas du lot de test sous aucune clé, aucune clé autre que
     « training » (donc ni « testing », ni « validation »), aucun chemin vers imagesTs / labelsTs ;
  5. écrit la liste et sa fiche de traçabilité (empreintes SHA-256 de splits_final.json, du découpage figé
     et de la liste, pli de chaque cas, version du code).
  Le lot de test n'est jamais lu : l'appartenance au test est vérifiée par les seuls identifiants du
  découpage figé.

Entrées :
  --raw           dossier nnUNet_raw (défaut /content/drive/MyDrive/poc_ct/nnunet_raw)
  --preprocessed  dossier nnUNet_preprocessed, où 05a a écrit splits_final.json (défaut .../nnunet_preprocessed)
  --split         découpage figé (défaut ../splits/mmwhs_split.json à côté du script)
  --sortie        dossier Auto3DSeg dans Drive (défaut /content/drive/MyDrive/poc_ct/auto3dseg)
  --id-dataset    numéro du jeu (défaut 101)
  --ecraser       autorise à remplacer une liste existante DIFFÉRENTE
Sorties (dans --sortie) :
  datalist_mmwhs_5plis.json : la liste de données lue par Auto3DSeg (chemins relatifs au dossier du jeu) ;
  datalist_05k.json         : fiche de traçabilité (empreintes, pli de chaque cas, date, commit).
Codes de sortie :
  0 = liste prête ; 1 = garde-fou (dossier dans le dépôt, prérequis de 05a absents, fichiers inattendus) ;
  2 = une liste DIFFÉRENTE existe déjà (relancer avec --ecraser) ; 4 = contrôle de la liste en échec.

Vocabulaire :
  - Auto3DSeg : la « recette » automatique de MONAI. Elle analyse les données (DataAnalyzer), prépare des
    algorithmes (ici SegResNet seulement, D37) puis les entraîne ;
  - liste de données (datalist) : fichier JSON qui dit à Auto3DSeg quelles images et quels masques
    utiliser. Auto3DSeg ne cherche pas lui-même les fichiers : il ne voit que cette liste ;
  - champ « fold » : numéro du pli de VALIDATION du cas. Pour le pli k, Auto3DSeg valide sur les cas dont
    fold = k et entraîne sur tous les autres. Sans ce champ, Auto3DSeg tirerait des plis au hasard, cas
    par cas (et pourrait séparer les deux examens d'un même patient) ;
  - empreinte SHA-256 : « signature » calculée sur le contenu d'un fichier ; elle change si un seul octet
    change. Elle prouve plus tard que le fichier utilisé est bien celui-ci.
"""
import argparse, datetime, glob, importlib.util, json, os, sys

ICI = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("essai05b", os.path.join(ICI, "05b_essai_nnunet.py"))
essai = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(essai)                   # 05b n'exécute rien à l'import ; il apporte aussi 05a
prep = essai.prep

NOM_LISTE = "datalist_mmwhs_5plis.json"
NOM_FICHE = "datalist_05k.json"
CLES_AUTORISEES = {"training"}                    # ni « testing », ni « validation » (vérifié dans MONAI 1.6.1)
PAIRES_ATTENDUES = [{"mmwhs_1001", "mmwhs_1020"}, {"mmwhs_1005", "mmwhs_1012"}]   # D20, contrôle explicite
INTERDITS_CHEMIN = ("imagesTs", "labelsTs")


# --- Fonctions réutilisées par 05l (aucun effet à l'import) ------------------------------------
def chemins_cas(cas):
    """Chemins relatifs (au dossier du jeu) de l'image et du masque d'un cas, nommage de l'étape 3."""
    return f"imagesTr/{cas}_0000.nii.gz", f"labelsTr/{cas}.nii.gz"


def pli_de_chaque_cas(plis):
    """{cas: numéro du pli où il est en validation}. Lève ValueError si un cas est en validation 0 ou
    plusieurs fois."""
    pli = {}
    for k, p in enumerate(plis):
        for c in p["val"]:
            if c in pli:
                raise ValueError(f"{c} est en validation dans les plis {pli[c]} et {k}")
            pli[c] = k
    tous = {c for p in plis for c in p["train"] + p["val"]}
    if set(pli) != tous:
        raise ValueError(f"cas jamais en validation : {sorted(tous - set(pli))}")
    return pli


def construire_liste(plis):
    """Liste Auto3DSeg : {"training": [{"fold", "image", "label"}, ...]}, cas triés."""
    pli = pli_de_chaque_cas(plis)
    elements = []
    for c in sorted(pli):
        image, masque = chemins_cas(c)
        elements.append({"fold": pli[c], "image": image, "label": masque})
    return {"training": elements}


def _cas_du_chemin(chemin):
    """« imagesTr/mmwhs_1001_0000.nii.gz » → « mmwhs_1001 » ; « labelsTr/mmwhs_1001.nii.gz » → idem."""
    nom = os.path.basename(str(chemin))
    nom = nom[:-len(".nii.gz")] if nom.endswith(".nii.gz") else nom
    return nom[:-len("_0000")] if nom.endswith("_0000") else nom


def _chaines(objet):
    """Toutes les chaînes (clés et valeurs) contenues dans un objet JSON, à toute profondeur."""
    if isinstance(objet, dict):
        for k, v in objet.items():
            yield str(k)
            yield from _chaines(v)
    elif isinstance(objet, list):
        for v in objet:
            yield from _chaines(v)
    elif isinstance(objet, str):
        yield objet


def controler_liste(liste, plis, cas, cas_test, groupes):
    """Contrôles bloquants de la liste Auto3DSeg contre splits_final.json et le découpage figé.
    Renvoie la liste des problèmes (vide = conforme)."""
    pb = []
    if not isinstance(liste, dict):
        return ["la liste n'est pas un objet JSON"]
    # 1. Clés : uniquement « training » (pas de « testing », même vide ; pas de « validation »).
    if set(liste) != CLES_AUTORISEES:
        pb.append(f"clés de la liste {sorted(liste)} ≠ {sorted(CLES_AUTORISEES)} (aucune clé testing ni validation admise)")
    elements = liste.get("training", [])
    if not isinstance(elements, list):
        return pb + ["« training » n'est pas une liste"]
    # 2. Aucun cas de test, sous aucune clé et dans aucune chaîne ; aucun chemin vers imagesTs / labelsTs.
    numeros_test = {c.split("_")[-1] for c in cas_test}
    for s in _chaines(liste):
        if any(x in s for x in INTERDITS_CHEMIN):
            pb.append(f"chemin interdit (imagesTs / labelsTs) : {s}")
        if any(n in s for n in numeros_test):
            pb.append(f"CAS DE TEST mentionné dans la liste : {s}")
        if os.path.isabs(s) or ".." in s.replace("\\", "/").split("/"):
            pb.append(f"chemin absolu ou remontant d'un dossier : {s}")
    # 3. Chaque élément : un cas d'entraînement, image et masque à l'emplacement attendu, un pli entier.
    vus, pli = [], {}
    for i, e in enumerate(elements):
        if not isinstance(e, dict) or set(e) != {"fold", "image", "label"}:
            pb.append(f"élément {i} : champs {sorted(e) if isinstance(e, dict) else type(e).__name__} ≠ fold, image, label")
            continue
        c = _cas_du_chemin(e["image"])
        if (e["image"], e["label"]) != chemins_cas(c):
            pb.append(f"élément {i} : chemins inattendus {e['image']} / {e['label']} (attendu {chemins_cas(c)})")
        if c in cas_test:
            pb.append(f"élément {i} : CAS DE TEST {c}")
        if not isinstance(e["fold"], int) or isinstance(e["fold"], bool) or not 0 <= e["fold"] < len(plis):
            pb.append(f"élément {i} ({c}) : fold {e['fold']!r} n'est pas un entier de 0 à {len(plis) - 1}")
        vus.append(c)
        pli[c] = e["fold"]
    if len(vus) != prep.NB_CAS_ENTRAINEMENT or len(set(vus)) != len(vus):
        pb.append(f"{len(vus)} éléments, {len(set(vus))} cas distincts (attendu exactement {prep.NB_CAS_ENTRAINEMENT}, sans doublon)")
    if set(vus) != set(cas):
        pb.append(f"cas de la liste ≠ cas d'entraînement du découpage figé (en trop : {sorted(set(vus) - set(cas))}, "
                  f"manquants : {sorted(set(cas) - set(vus))})")
    # 4. Correspondance pli par pli avec splits_final.json : mêmes ensembles validation ET entraînement.
    if len(plis) != prep.NB_PLIS:
        pb.append(f"{len(plis)} plis dans splits_final.json au lieu de {prep.NB_PLIS}")
    for k, p in enumerate(plis):
        val = {c for c, f in pli.items() if f == k}
        train = set(vus) - val                     # règle d'Auto3DSeg : tout ce qui n'est pas « fold = k »
        if val != set(p["val"]) or train != set(p["train"]):
            pb.append(f"pli {k} : validation {sorted(val)} / {len(train)} en entraînement ≠ splits_final.json "
                      f"(validation {sorted(p['val'])} / {len(p['train'])} en entraînement)")
    # 5. Patients regroupés (D20) : les deux examens dans le même pli de validation.
    for g in [set(g) for g in groupes] + PAIRES_ATTENDUES:
        plis_g = {pli.get(c) for c in g}
        if len(plis_g) != 1 or None in plis_g:
            pb.append(f"le patient {sorted(g)} est SÉPARÉ ou absent (plis {sorted(str(x) for x in plis_g)})")
    return pb


def verifier_plis_05a(raw, pre, nom_ds, split):
    """Prérequis de 05a (fonction de 05b : marqueur, empreinte, plis recalculés). Arrêt (code 1) sinon.
    Renvoie (chemin de splits_final.json, fiche de 05a, plis, cas, cas_test, groupes)."""
    _, chemin_splits, _, trace, plis = essai.verifier_prerequis_05a(raw, pre, nom_ds, split, 0)
    cas, cas_test, groupes, _ = prep.lire_decoupage(split)
    return chemin_splits, trace, plis, cas, cas_test, groupes


def lire_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=f"{prep.DRIVE}/nnunet_raw")
    p.add_argument("--preprocessed", default=f"{prep.DRIVE}/nnunet_preprocessed")
    p.add_argument("--split", default=os.path.join(prep.DEPOT, "splits", "mmwhs_split.json"))
    p.add_argument("--sortie", default=f"{prep.DRIVE}/auto3dseg")
    p.add_argument("--id-dataset", type=int, default=101)
    p.add_argument("--ecraser", action="store_true")
    return p.parse_args()


# --- Programme principal ----------------------------------------------------------------------
def main():
    a = lire_arguments()
    raw, pre, sortie = (os.path.abspath(x) for x in (a.raw, a.preprocessed, a.sortie))
    nom_ds = f"Dataset{a.id_dataset:03d}_MMWHS"
    ds_raw = os.path.join(raw, nom_ds)
    for nom, d in (("--raw", raw), ("--preprocessed", pre), ("--sortie", sortie)):
        if prep.dans(d, prep.DEPOT):
            sys.exit(f"ARRÊT : {nom} est dans le dépôt. Les données et résultats restent dans Google Drive.")

    # --- 1. splits_final.json : celui de 05a (sinon arrêt, code 1) ---------------------------------
    chemin_splits, trace, plis, cas, cas_test, groupes = verifier_plis_05a(raw, pre, nom_ds, a.split)

    # --- 2. Fichiers de imagesTr et labelsTr seulement (jamais imagesTs ni labelsTs, D18) -----------
    images = {os.path.basename(x)[:-len("_0000.nii.gz")] for x in glob.glob(os.path.join(ds_raw, "imagesTr", "*_0000.nii.gz"))}
    autres = [x for x in glob.glob(os.path.join(ds_raw, "imagesTr", "*")) if not x.endswith("_0000.nii.gz")]
    masques = {os.path.basename(x)[:-len(".nii.gz")] for x in glob.glob(os.path.join(ds_raw, "labelsTr", "*.nii.gz"))}
    autres += [x for x in glob.glob(os.path.join(ds_raw, "labelsTr", "*")) if not x.endswith(".nii.gz")]
    print(f"imagesTr : {len(images)} cas ; labelsTr : {len(masques)} cas ({ds_raw})")
    if images != set(cas) or masques != set(cas) or autres:
        sys.exit(f"ARRÊT : imagesTr/labelsTr ≠ les {len(cas)} cas d'entraînement du découpage figé.\n"
                 f"  images en trop : {sorted(images - set(cas))} ; manquantes : {sorted(set(cas) - images)}\n"
                 f"  masques en trop : {sorted(masques - set(cas))} ; manquants : {sorted(set(cas) - masques)}\n"
                 f"  autres fichiers dans imagesTr ou labelsTr : {autres}")

    # --- 3 et 4. Liste, contrôle, écriture, relecture, recontrôle ----------------------------------
    try:
        liste = construire_liste(plis)
    except ValueError as err:
        print(f"ARRÊT : {err}")
        sys.exit(4)
    problemes = controler_liste(liste, plis, cas, cas_test, groupes)
    if problemes:
        print("ARRÊT : liste non conforme :\n  - " + "\n  - ".join(problemes))
        sys.exit(4)
    chemin_liste, chemin_fiche = os.path.join(sortie, NOM_LISTE), os.path.join(sortie, NOM_FICHE)
    texte = json.dumps(liste, indent=2, ensure_ascii=False) + "\n"
    if os.path.exists(chemin_liste):
        with open(chemin_liste, encoding="utf-8") as f:
            ancien = f.read()
        if ancien != texte and not a.ecraser:
            print(f"ARRÊT : {chemin_liste} existe déjà et est DIFFÉRENT. Rien n'a été modifié.\n"
                  "Relancer avec --ecraser pour le remplacer volontairement (un essai Auto3DSeg déjà fait "
                  "avec l'ancienne liste devra alors être refait).")
            sys.exit(2)
    os.makedirs(sortie, exist_ok=True)
    with open(chemin_liste + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(texte)
    os.replace(chemin_liste + ".tmp", chemin_liste)   # remplacement d'un seul coup
    with open(chemin_liste, encoding="utf-8") as f:   # contrôle de ce qui est réellement sur le disque
        relue = json.load(f)
    problemes = controler_liste(relue, plis, cas, cas_test, groupes)
    if problemes or relue != liste:
        os.remove(chemin_liste)
        print("ARRÊT : la liste relue sur le disque n'est pas conforme (fichier effacé) :\n  - " +
              "\n  - ".join(problemes or ["contenu relu ≠ contenu écrit"]))
        sys.exit(4)

    # --- 5. Fiche de traçabilité --------------------------------------------------------------------
    pli = {e["image"].split("/")[1][:-len("_0000.nii.gz")]: e["fold"] for e in relue["training"]}
    fiche = {"script": "scripts/05k_preparer_auto3dseg.py", "commit": prep.commit_git(),
             "date": datetime.datetime.now().isoformat(timespec="seconds"), "decision": "D37",
             "liste": NOM_LISTE, "sha256_liste": prep.sha256_fichier(chemin_liste),
             "splits_final": chemin_splits, "sha256_splits_final": prep.sha256_fichier(chemin_splits),
             "sha256_splits_final_note_par_05a": trace["sha256_splits_final"],
             "decoupage_fige": os.path.relpath(a.split, prep.DEPOT) if prep.dans(a.split, prep.DEPOT) else a.split,
             "sha256_decoupage_fige": prep.sha256_fichier(a.split),
             "dossier_du_jeu_lors_de_la_preparation": ds_raw, "pli_de_chaque_cas": pli,
             "groupes_patients": groupes, "nb_cas": len(pli)}
    with open(chemin_fiche + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        json.dump(fiche, f, indent=2, ensure_ascii=False)
    os.replace(chemin_fiche + ".tmp", chemin_fiche)

    print(f"\nListe Auto3DSeg : {len(pli)} cas, plis repris à l'identique de splits_final.json :")
    for k, p in enumerate(plis):
        print(f"  pli {k} : validation = {sorted(c for c, f in pli.items() if f == k)}  ({len(p['train'])} en entraînement)")
    print(f"Contrôles : CONFORME (clé unique « training », aucun cas de test, paires {[sorted(g) for g in PAIRES_ATTENDUES]} "
          "dans le même pli, correspondance pli par pli identique à splits_final.json)")
    print(f"SHA-256 de splits_final.json : {fiche['sha256_splits_final']}")
    print(f"SHA-256 du découpage figé    : {fiche['sha256_decoupage_fige']}")
    print(f"SHA-256 de la liste          : {fiche['sha256_liste']}")
    print(f"Écrit : {chemin_liste}\nÉcrit : {chemin_fiche}")
    print("Résultat : LISTE PRÊTE pour l'essai SegResNet (script 05l). Le lot de test n'a été ni lu ni utilisé.")


if __name__ == "__main__":
    main()
