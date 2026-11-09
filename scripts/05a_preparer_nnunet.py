"""
Étape 5a — Préparer nnU-Net v2 : vérification du jeu, prétraitement, plis de validation croisée par patient.

Ce que fait le script :
  1. garde-fous : les trois dossiers nnU-Net sont hors du dépôt ; le découpage figé
     (splits/mmwhs_split.json, LU SEULEMENT) est cohérent ; imagesTr et labelsTr contiennent exactement
     les 15 cas d'entraînement, et aucun cas de test. Seuls imagesTr et labelsTr sont listés : jamais
     imagesTs ni labelsTs (décision D18) ;
  2. construit 5 plis de validation croisée PAR PATIENT (groupes lus dans « groupes_patients » : 1001/1020
     et 1005/1012 toujours dans le même pli), avec une graine fixe, et les contrôle (voir controler_plis) ;
     s'arrête tout de suite si un splits_final.json DIFFÉRENT existe déjà (sauf --ecraser) ;
  3. compare lui-même, avec SimpleITK (la bibliothèque de lecture d'images de nnU-Net), chaque image de
     imagesTr à son masque de labelsTr : taille, espacement, origine, orientation (même tolérance que
     nnU-Net : np.allclose), et vérifie qu'aucun masque n'est vide. C'est CE contrôle qui fait foi pour le
     point A3 du README : il ne dépend pas de ce que nnU-Net affiche. Tout écart → arrêt (code 4) ;
  4. efface l'ancienne fiche splits_final_05a.json et l'ancien marqueur verification_05a.json, puis lance
     `nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity -c 3d_fullres`, avec les variables
     nnUNet_raw / nnUNet_preprocessed / nnUNet_results pointant vers Drive, et un nombre de processus
     limité (RAM de Colab). La sortie est recopiée dans un journal dans Drive. Le script s'arrête aussi si
     nnU-Net signale un écart (pour nnU-Net, simple « Warning » : il continuerait sans rien dire) ;
  5. écrit le marqueur verification_05a.json SEULEMENT si tout est conforme : contrôle SimpleITK, aucune
     alerte de nnU-Net, code de retour 0, ligne « verify_dataset_integrity Done ». Avec
     --sauter-pretraitement, ce marqueur est exigé (sinon arrêt) ; le script 05b l'exige aussi ;
  6. vérifie que les cas prétraités sont bien les 15 attendus, puis écrit splits_final.json et sa fiche
     de traçabilité dans nnUNet_preprocessed/Dataset101_MMWHS/.
  En cas d'arrêt pendant les étapes 3 à 6, la fiche et le marqueur sont effacés : 05b refusera de lancer.

Entrées :
  --raw           dossier nnUNet_raw (défaut /content/drive/MyDrive/poc_ct/nnunet_raw, cf. étape 3)
  --preprocessed  dossier nnUNet_preprocessed (défaut /content/drive/MyDrive/poc_ct/nnunet_preprocessed)
  --results       dossier nnUNet_results (défaut /content/drive/MyDrive/poc_ct/nnunet_results)
  --split         découpage figé (défaut : ../splits/mmwhs_split.json à côté du script)
  --id-dataset    numéro du jeu (défaut 101 → Dataset101_MMWHS)
  --graine        graine du tirage des plis (défaut : la graine du découpage figé, 42)
  --processus     processus de nnU-Net pour la vérification, l'empreinte et le prétraitement (défaut 2)
  --sauter-pretraitement  ne relance pas nnU-Net (prétraitement déjà fait) : écrit seulement les plis
  --ecraser       autorise à remplacer un splits_final.json existant DIFFÉRENT
Sorties (dans Drive) :
  nnUNet_preprocessed/Dataset101_MMWHS/ : plans, empreinte, données prétraitées 3d_fullres (par nnU-Net),
      splits_final.json, splits_final_05a.json (traçabilité : graine, groupes, empreinte SHA-256),
      verification_05a.json (marqueur : vérification conforme, date, cas vérifiés) ;
  nnUNet_preprocessed/journal_05a_plan_and_preprocess.txt (sortie complète de nnU-Net).
Codes de sortie :
  0 = prêt ; 1 = garde-fou (rien n'est lancé ; avec --sauter-pretraitement : marqueur absent ou périmé) ;
  2 = splits_final.json différent déjà présent (relancer avec --ecraser) ; 3 = nnU-Net a échoué ;
  4 = problème de géométrie, masque vide ou étiquettes signalé par SimpleITK ou par nnU-Net :
  NE PAS ENTRAÎNER, splits_final.json n'est pas écrit.

Vocabulaire :
  - prétraitement (preprocessing) : nnU-Net ré-échantillonne chaque scanner à un espacement commun et
    normalise les intensités, une fois pour toutes, avant l'entraînement ;
  - empreinte du jeu (fingerprint) : statistiques que nnU-Net calcule sur les 15 cas (tailles, espacements,
    intensités) pour choisir lui-même ses réglages (« plans ») : taille des patchs, architecture, etc. ;
  - configuration : une manière d'entraîner. 2d = coupe par coupe ; 3d_fullres = en 3D à pleine
    résolution ; 3d_lowres + cascade = d'abord en 3D à basse résolution, puis affinage ;
  - pli (fold) : sous-ensemble de cas mis de côté à tour de rôle pour valider. Avec 5 plis, on entraîne
    5 modèles ; chacun est validé sur environ 3 cas qu'il n'a pas vus ;
  - splits_final.json : le fichier où nnU-Net lit ses plis. S'il manque, nnU-Net en tire un AU HASARD,
    cas par cas, et pourrait séparer les deux examens d'un même patient ;
  - graine : nombre qui fixe le tirage au sort, pour retrouver exactement les mêmes plis à chaque fois ;
  - processus : programmes Python lancés en parallèle par nnU-Net ; chacun charge des volumes entiers
    en mémoire vive (RAM), d'où la limite.

Pourquoi seulement 3d_fullres (choix à valider) : par défaut, nnU-Net entraîne 2d, 3d_fullres et (si les
images sont grandes) 3d_lowres + cascade, 5 plis chacune, puis garde la meilleure. Sur Colab gratuit,
une seule configuration est envisageable. 3d_fullres est celle que les auteurs de nnU-Net recommandent
quand on n'en entraîne qu'une, et la plus souvent gagnante dans leurs comparaisons. Ne prétraiter
qu'elle économise aussi des heures et de la place dans Drive. Les plans des autres configurations sont
quand même calculés (c'est instantané) mais leurs données ne sont pas préparées.
"""
import argparse, datetime, glob, hashlib, json, os, random, re, shutil, subprocess, sys

# --- Paramètres fixes -------------------------------------------------------------------------
NB_PLIS = 5
NB_CAS_ENTRAINEMENT = 15          # D20 : 15 cas en entraînement / validation croisée
CONFIGURATION = "3d_fullres"
NOM_SPLITS = "splits_final.json"
NOM_TRACE = "splits_final_05a.json"
NOM_MARQUEUR = "verification_05a.json"
DEPOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DRIVE = "/content/drive/MyDrive/poc_ct"
# Messages de verify_dataset_integrity (nnunetv2 2.8.1, experiment_planning/verify_dataset_integrity.py).
# Les écarts d'origine, d'orientation (direction) et d'affine n'y sont que des « Warning » : on les traite
# ici comme bloquants, car un masque décalé par rapport à son scanner fausserait tout l'entraînement.
# « only has label 0 » (masque vide) : dans nnunetv2 2.8.1, la condition qui l'affiche ne peut jamais être
# vraie ; le motif est gardé au cas où, mais c'est le contrôle SimpleITK du script qui détecte les masques vides.
ALERTES_NNUNET = re.compile(r"(Shape|Spacing|Origin|Direction|Affine) mismatch|Unexpected labels|"
                            r"contain(s)? NaN|Unexpected number of modalities|only has label 0", re.I)


# --- Fonctions réutilisées par 05b (aucun effet à l'import) ------------------------------------
def identifiant(cas):
    """Nom du découpage figé → identifiant de cas nnU-Net. Ex. « ct_train_1001 » → « mmwhs_1001 »
    (nommage du script 03 : imagesTr/mmwhs_1001_0000.nii.gz et labelsTr/mmwhs_1001.nii.gz)."""
    return f"mmwhs_{cas.split('_')[-1]}"


def lire_decoupage(chemin):
    """Lit le découpage figé (lecture seule). Renvoie (cas d'entraînement, cas de test, groupes, graine),
    en identifiants nnU-Net. Lève ValueError si le découpage est incohérent."""
    with open(chemin, encoding="utf-8") as f:
        split = json.load(f)
    tr = [identifiant(c) for c in split["entrainement_validation_croisee"]]
    ts = [identifiant(c) for c in split["test_sous_cle"]]
    groupes = [sorted(identifiant(c) for c in g) for g in split.get("groupes_patients", [])]
    if len(tr) != NB_CAS_ENTRAINEMENT or len(set(tr)) != len(tr) or set(tr) & set(ts):
        raise ValueError(f"découpage incohérent : {len(set(tr))} cas d'entraînement distincts "
                         f"(attendu {NB_CAS_ENTRAINEMENT}), cas communs avec le test : {sorted(set(tr) & set(ts))}")
    for g in groupes:
        if not set(g) <= set(tr):
            raise ValueError(f"le groupe {g} n'est pas entièrement en entraînement (D20)")
    vus = [c for g in groupes for c in g]
    if len(vus) != len(set(vus)):
        raise ValueError("un cas apparaît dans deux groupes_patients")
    return tr, ts, groupes, split.get("graine", 42)


def construire_plis(cas, groupes, nb_plis=NB_PLIS, graine=42):
    """Répartit les cas en nb_plis plis de validation, PAR PATIENT.
    Un patient = un groupe de groupes_patients, ou un cas seul. Les patients sont mélangés avec la graine,
    puis rangés du plus grand au plus petit (les paires d'abord : plus faciles à caser), et chacun va dans
    le pli de validation le moins rempli (à égalité, le premier). Résultat ici : 5 plis de 3 cas.
    Renvoie la liste nnU-Net : [{"train": [...], "val": [...]}, ...] (listes triées)."""
    dans_un_groupe = {c for g in groupes for c in g}
    patients = [sorted(g) for g in groupes] + [[c] for c in sorted(cas) if c not in dans_un_groupe]
    random.Random(graine).shuffle(patients)
    patients.sort(key=len, reverse=True)          # tri stable : l'ordre tiré au sort est gardé à taille égale
    validation = [[] for _ in range(nb_plis)]
    for p in patients:
        k = min(range(nb_plis), key=lambda i: (len(validation[i]), i))
        validation[k] += p
    return [{"train": sorted(c for c in cas if c not in v), "val": sorted(v)} for v in validation]


def controler_plis(plis, cas, cas_test, groupes, nb_plis=NB_PLIS):
    """Contrôles bloquants. Renvoie la liste des problèmes (vide = conforme)."""
    pb, cas = [], set(cas)
    if len(cas) != NB_CAS_ENTRAINEMENT:
        pb.append(f"{len(cas)} cas d'entraînement au lieu de {NB_CAS_ENTRAINEMENT}")
    if len(plis) != nb_plis:
        pb.append(f"{len(plis)} plis au lieu de {nb_plis}")
    nb_val = {c: 0 for c in cas}
    for i, p in enumerate(plis):
        tr, val = p.get("train", []), p.get("val", [])
        if len(tr) != len(set(tr)) or len(val) != len(set(val)):
            pb.append(f"pli {i} : un cas est répété")
        if set(tr) & set(val):
            pb.append(f"pli {i} : cas à la fois en entraînement et en validation : {sorted(set(tr) & set(val))}")
        if set(tr) | set(val) != cas:
            pb.append(f"pli {i} : entraînement ∪ validation ≠ les {len(cas)} cas "
                      f"(en trop : {sorted((set(tr) | set(val)) - cas)}, manquants : {sorted(cas - set(tr) - set(val))})")
        if (set(tr) | set(val)) & set(cas_test):
            pb.append(f"pli {i} : CAS DE TEST présent : {sorted((set(tr) | set(val)) & set(cas_test))}")
        if not val:
            pb.append(f"pli {i} : validation vide")
        for c in val:
            nb_val[c] = nb_val.get(c, 0) + 1
        for g in groupes:
            if 0 < len(set(g) & set(val)) < len(g):
                pb.append(f"pli {i} : le patient {g} est SÉPARÉ entre entraînement et validation")
    for c, n in sorted(nb_val.items()):
        if n != 1:
            pb.append(f"{c} est en validation {n} fois (attendu : exactement 1)")
    tailles = [len(p.get("val", [])) for p in plis]
    if tailles and max(tailles) - min(tailles) > 1:
        pb.append(f"plis déséquilibrés : tailles de validation {tailles}")
    return pb


def sha256_fichier(chemin):
    with open(chemin, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def commit_git():
    """Version du code : identifiant du commit du dépôt, suivi de « +modifié » si des fichiers ont changé."""
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=DEPOT, capture_output=True, text=True, timeout=10).stdout.strip()
        sale = subprocess.run(["git", "status", "--porcelain"], cwd=DEPOT, capture_output=True, text=True, timeout=10).stdout.strip()
        return (c or "inconnu") + ("+modifié" if sale else "")
    except Exception:
        return "inconnu"


def environnement_nnunet(raw, preprocessed, results, processus):
    """Copie de l'environnement avec les variables que lisent les commandes nnU-Net.
    nnUNet_def_n_proc : nombre de processus par défaut de nnU-Net (8 sinon), utilisé notamment pour
    l'export des prédictions de validation."""
    env = os.environ.copy()
    env.update({"nnUNet_raw": raw, "nnUNet_preprocessed": preprocessed, "nnUNet_results": results,
                "nnUNet_def_n_proc": str(processus),
                # Sortie non mise en mémoire tampon : les messages des processus enfants de nnU-Net arrivent
                # tout de suite dans le tuyau (et dans le journal), au lieu de risquer d'être perdus.
                "PYTHONUNBUFFERED": "1"})
    return env


def verifier_geometrie_sitk(ds_raw, cas):
    """Compare, avec SimpleITK, chaque image de imagesTr à son masque de labelsTr (cas d'entraînement
    UNIQUEMENT) : taille exacte ; espacement, origine et orientation avec np.allclose (tolérance de
    nnU-Net). Vérifie aussi que le masque n'est pas vide (au moins une étiquette > 0).
    Seuls les en-têtes des images sont lus ; les masques sont lus en entier (pour le test « vide »).
    Renvoie la liste des problèmes (vide = conforme)."""
    import numpy as np
    import SimpleITK as sitk
    pb = []
    for c in sorted(cas):
        infos = []
        for chemin in (os.path.join(ds_raw, "imagesTr", f"{c}_0000.nii.gz"), os.path.join(ds_raw, "labelsTr", f"{c}.nii.gz")):
            lecteur = sitk.ImageFileReader()
            lecteur.SetFileName(chemin)
            lecteur.ReadImageInformation()            # en-tête seulement : rapide, même depuis Drive
            infos.append((lecteur.GetSize(), lecteur.GetSpacing(), lecteur.GetOrigin(), lecteur.GetDirection()))
        (t_i, e_i, o_i, d_i), (t_m, e_m, o_m, d_m) = infos
        ecarts = [nom for nom, ok in (("taille", t_i == t_m), ("espacement", np.allclose(e_i, e_m)),
                                      ("origine", np.allclose(o_i, o_m)), ("orientation", np.allclose(d_i, d_m)))
                  if not ok]
        if ecarts:
            pb.append(f"{c} : image et masque diffèrent ({', '.join(ecarts)}) : image {infos[0]} / masque {infos[1]}")
        stats = sitk.MinimumMaximumImageFilter()
        stats.Execute(sitk.ReadImage(os.path.join(ds_raw, "labelsTr", f"{c}.nii.gz")))
        if stats.GetMaximum() <= 0:
            pb.append(f"{c} : masque vide (seulement l'étiquette 0)")
        print(f"  {c} : taille {t_i}, espacement {tuple(round(x, 3) for x in e_i)} mm → "
              f"{'conforme' if not ecarts and stats.GetMaximum() > 0 else 'PROBLÈME'}", flush=True)
    return pb


def lire_marqueur(ds_pre, cas):
    """Renvoie None si le marqueur verification_05a.json prouve une vérification conforme des mêmes cas,
    sinon le message qui explique pourquoi il n'est pas valable."""
    chemin = os.path.join(ds_pre, NOM_MARQUEUR)
    if not os.path.exists(chemin):
        return f"{chemin} absent : la vérification du jeu n'a pas été faite (ou a échoué)."
    with open(chemin, encoding="utf-8") as f:
        m = json.load(f)
    if m.get("ok") is not True or sorted(m.get("cas", [])) != sorted(cas):
        return f"{chemin} ne prouve pas une vérification conforme des {len(cas)} cas attendus."
    return None


def invalider(ds_pre):
    """Efface la fiche des plis et le marqueur de vérification : 05b refusera alors de lancer."""
    for nom in (NOM_TRACE, NOM_MARQUEUR):
        if os.path.exists(os.path.join(ds_pre, nom)):
            os.remove(os.path.join(ds_pre, nom))


def lancer(commande, env, journal):
    """Lance une commande, affiche sa sortie au fil de l'eau ET la recopie dans le fichier journal.
    Renvoie (code de retour, texte complet)."""
    print("Commande :", " ".join(commande), flush=True)
    lignes = []
    with open(journal, "a", encoding="utf-8") as j:
        j.write(f"\n===== {datetime.datetime.now().isoformat(timespec='seconds')} : {' '.join(commande)}\n")
        proc = subprocess.Popen(commande, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        for ligne in proc.stdout:
            print(ligne, end="", flush=True)
            j.write(ligne)
            lignes.append(ligne)
        code = proc.wait()
    return code, "".join(lignes)


def dans(chemin, dossier):
    chemin, dossier = os.path.abspath(chemin), os.path.abspath(dossier)
    return os.path.commonpath([chemin, dossier]) == dossier


# --- Programme principal ----------------------------------------------------------------------
def lire_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=f"{DRIVE}/nnunet_raw")
    p.add_argument("--preprocessed", default=f"{DRIVE}/nnunet_preprocessed")
    p.add_argument("--results", default=f"{DRIVE}/nnunet_results")
    p.add_argument("--split", default=os.path.join(DEPOT, "splits", "mmwhs_split.json"))
    p.add_argument("--id-dataset", type=int, default=101)
    p.add_argument("--graine", type=int, default=None)
    p.add_argument("--processus", type=int, default=2, choices=[1, 2, 3, 4])
    p.add_argument("--sauter-pretraitement", action="store_true")
    p.add_argument("--ecraser", action="store_true")
    return p.parse_args()


def main():
    a = lire_arguments()
    raw, pre, res = (os.path.abspath(x) for x in (a.raw, a.preprocessed, a.results))
    nom_ds = f"Dataset{a.id_dataset:03d}_MMWHS"
    ds_raw, ds_pre = os.path.join(raw, nom_ds), os.path.join(pre, nom_ds)

    # --- 1. Garde-fous ----------------------------------------------------------------------------
    for nom, d in (("--raw", raw), ("--preprocessed", pre), ("--results", res)):
        if dans(d, DEPOT):
            sys.exit(f"ARRÊT : {nom} est dans le dépôt. Les données doivent rester dans Google Drive.")
    try:
        cas, cas_test, groupes, graine_split = lire_decoupage(a.split)
    except (ValueError, KeyError) as err:
        sys.exit(f"ARRÊT : {err}")
    graine = graine_split if a.graine is None else a.graine

    # Identifiants vus par nnU-Net : uniquement imagesTr et labelsTr (jamais imagesTs ni labelsTs, D18).
    images = {os.path.basename(x)[:-len("_0000.nii.gz")]
              for x in glob.glob(os.path.join(ds_raw, "imagesTr", "*_0000.nii.gz"))}
    autres_canaux = [x for x in glob.glob(os.path.join(ds_raw, "imagesTr", "*.nii.gz")) if not x.endswith("_0000.nii.gz")]
    masques = {os.path.basename(x)[:-len(".nii.gz")] for x in glob.glob(os.path.join(ds_raw, "labelsTr", "*.nii.gz"))}
    print(f"imagesTr : {len(images)} cas ; labelsTr : {len(masques)} cas ({ds_raw})")
    if images != masques or images != set(cas) or autres_canaux:
        sys.exit(f"ARRÊT : imagesTr/labelsTr ne correspondent pas aux {len(cas)} cas d'entraînement du découpage figé.\n"
                 f"  images sans masque : {sorted(images - masques)} ; masques sans image : {sorted(masques - images)}\n"
                 f"  absents : {sorted(set(cas) - images)} ; en trop : {sorted(images - set(cas))} ; "
                 f"fichiers d'un autre canal : {autres_canaux}")
    if (images | masques) & set(cas_test):   # déjà couvert ci-dessus, gardé explicite : le test ne doit pas fuiter
        sys.exit(f"ARRÊT : cas de test dans imagesTr/labelsTr : {sorted((images | masques) & set(cas_test))}")
    with open(os.path.join(ds_raw, "dataset.json"), encoding="utf-8") as f:
        if json.load(f).get("numTraining") != len(cas):
            sys.exit(f"ARRÊT : dataset.json n'annonce pas {len(cas)} cas d'entraînement (numTraining).")

    # --- 2. Plis par patient ----------------------------------------------------------------------
    plis = construire_plis(cas, groupes, NB_PLIS, graine)
    problemes = controler_plis(plis, cas, cas_test, groupes)
    if problemes:
        sys.exit("ARRÊT : plis non conformes :\n  - " + "\n  - ".join(problemes))
    print(f"\nPlis de validation croisée (par patient, graine {graine}) :")
    for i, p in enumerate(plis):
        print(f"  pli {i} : validation = {p['val']}  ({len(p['train'])} en entraînement)")
    print(f"  patients regroupés : {groupes}")

    chemin_splits = os.path.join(ds_pre, NOM_SPLITS)
    if os.path.exists(chemin_splits):
        with open(chemin_splits, encoding="utf-8") as f:
            existant = json.load(f)
        normalise = [{"train": sorted(p["train"]), "val": sorted(p["val"])} for p in existant]
        if normalise != plis and not a.ecraser:
            print(f"ARRÊT : {chemin_splits} existe déjà et est DIFFÉRENT (peut-être tiré au hasard par nnU-Net "
                  "lors d'un entraînement lancé trop tôt). Rien n'a été modifié.")
            print("Relancer avec --ecraser pour le remplacer volontairement.")
            sys.exit(2)

    # --- 3. Vérification indépendante avec SimpleITK (fait foi pour le point A3) -----------------
    def arret(code, message):
        """Arrêt pendant la vérification ou le prétraitement : fiche et marqueur effacés (05b bloqué)."""
        invalider(ds_pre)
        print(message)
        sys.exit(code)

    # L'ancienne fiche des plis est effacée dans tous les cas : elle ne sera réécrite qu'à la fin, si tout
    # est conforme. Le marqueur est effacé aussi, sauf avec --sauter-pretraitement (il est alors exigé).
    if os.path.exists(os.path.join(ds_pre, NOM_TRACE)):
        os.remove(os.path.join(ds_pre, NOM_TRACE))
    print("\nContrôle SimpleITK, image contre masque (imagesTr / labelsTr uniquement) :")
    ecarts = verifier_geometrie_sitk(ds_raw, cas)
    if ecarts:
        arret(4, "\nARRÊT : écarts entre images et masques :\n  - " + "\n  - ".join(ecarts) +
              "\nNE PAS ENTRAÎNER. splits_final.json n'a pas été écrit. Envoyer ce message dans la conversation.")
    print("Contrôle SimpleITK : CONFORME (taille, espacement, origine, orientation identiques ; aucun masque vide).")

    # --- 4. Vérification + empreinte + plans + prétraitement par nnU-Net -------------------------
    env = environnement_nnunet(raw, pre, res, a.processus)
    os.makedirs(pre, exist_ok=True)
    os.makedirs(res, exist_ok=True)
    if a.sauter_pretraitement:
        message = lire_marqueur(ds_pre, cas)
        if message:
            arret(1, f"ARRÊT : {message} Relancer SANS --sauter-pretraitement.")
        print(f"Marqueur {NOM_MARQUEUR} présent : la vérification nnU-Net de ces {len(cas)} cas était conforme.")
    else:
        invalider(ds_pre)
        exe = shutil.which("nnUNetv2_plan_and_preprocess")
        if exe is None:
            sys.exit("ARRÊT : nnUNetv2_plan_and_preprocess introuvable. Installer nnU-Net (notebook, section 12).")
        # -np : processus du prétraitement (défaut nnU-Net : 4 pour 3d_fullres) ; -npfp : processus de la
        # vérification et de l'empreinte (défaut 8). Colab gratuit n'a que 2 processeurs et ~12 Go de RAM :
        # au-delà de 2, on ne gagne pas de temps et on multiplie la mémoire (un volume ré-échantillonné
        # occupe plusieurs Go). Passer --processus 1 si la RAM sature.
        commande = [exe, "-d", str(a.id_dataset), "--verify_dataset_integrity", "-c", CONFIGURATION,
                    "-np", str(a.processus), "-npfp", str(a.processus), "--no_pbar"]
        journal = os.path.join(pre, "journal_05a_plan_and_preprocess.txt")
        code, sortie = lancer(commande, env, journal)
        print(f"\nJournal complet : {journal}")
        alertes = [l.strip() for l in sortie.splitlines() if ALERTES_NNUNET.search(l)]
        if alertes:
            arret(4, "\nARRÊT : nnU-Net signale un problème dans les données (géométrie image/masque ou étiquettes) :\n  - "
                  + "\n  - ".join(alertes) +
                  "\nNE PAS ENTRAÎNER. splits_final.json n'a pas été écrit. Envoyer ce message dans la conversation.")
        if code != 0 or "verify_dataset_integrity Done" not in sortie:
            arret(3, f"\nARRÊT : nnU-Net s'est arrêté en erreur (code {code}) ou la vérification n'a pas abouti.")
        print("\nVérification nnU-Net (forme, espacement, origine, orientation, étiquettes) : aucun écart signalé.")
        # Marqueur écrit seulement ici : SimpleITK conforme, aucune alerte, code 0, « verify_dataset_integrity Done ».
        with open(os.path.join(ds_pre, NOM_MARQUEUR), "w", encoding="utf-8") as f:
            json.dump({"ok": True, "date": datetime.datetime.now().isoformat(timespec="seconds"),
                       "cas": sorted(cas), "controle_simpleitk": "conforme", "alertes_nnunet": 0,
                       "code_retour_nnunet": code, "commande": " ".join(commande)}, f, indent=2, ensure_ascii=False)

    # --- 5. Contrôle des cas prétraités, puis écriture des plis ---------------------------------
    chemin_plans = os.path.join(ds_pre, "nnUNetPlans.json")
    if not os.path.exists(chemin_plans):
        arret(1, f"ARRÊT : {chemin_plans} absent : le prétraitement n'a pas été fait (enlever --sauter-pretraitement).")
    with open(chemin_plans, encoding="utf-8") as f:
        conf = json.load(f)["configurations"].get(CONFIGURATION, {})
    dossier_donnees = os.path.join(ds_pre, conf.get("data_identifier", f"nnUNetPlans_{CONFIGURATION}"))
    # nnunetv2 2.8.1 écrit un fichier .b2nd par cas (et un _seg.b2nd pour le masque).
    pretraites = {os.path.basename(x)[:-5] for x in glob.glob(os.path.join(dossier_donnees, "*.b2nd"))
                  if not x.endswith("_seg.b2nd")}
    if pretraites != set(cas):
        arret(1, f"ARRÊT : cas prétraités ({dossier_donnees}) ≠ cas attendus. "
                 f"Manquants : {sorted(set(cas) - pretraites)} ; en trop : {sorted(pretraites - set(cas))}.")
    print(f"Cas prétraités en {CONFIGURATION} : {len(pretraites)}, identiques aux cas attendus.")
    print(f"Réglages choisis par nnU-Net pour {CONFIGURATION} : espacement {conf.get('spacing')} mm, "
          f"patch {conf.get('patch_size')}, lot (batch) {conf.get('batch_size')}")

    tmp = chemin_splits + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(plis, f, indent=2)
    os.replace(tmp, chemin_splits)                   # remplacement d'un seul coup : jamais de fichier à moitié écrit
    trace = {"script": "scripts/05a_preparer_nnunet.py", "commit": commit_git(),
             "date": datetime.datetime.now().isoformat(timespec="seconds"),
             "graine": graine, "nb_plis": NB_PLIS, "groupes_patients": groupes,
             "decoupage_fige": os.path.relpath(a.split, DEPOT) if dans(a.split, DEPOT) else a.split,
             "sha256_decoupage_fige": sha256_fichier(a.split),
             "sha256_splits_final": sha256_fichier(chemin_splits)}
    with open(os.path.join(ds_pre, NOM_TRACE), "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    print(f"\nÉcrit : {chemin_splits}\nÉcrit : {os.path.join(ds_pre, NOM_TRACE)}")
    print("Résultat : PRÊT pour l'essai (script 05b). Le lot de test n'a été ni lu ni utilisé.")


if __name__ == "__main__":
    main()
