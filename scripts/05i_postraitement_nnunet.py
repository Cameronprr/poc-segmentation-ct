"""
Étape 5i — Déterminer le post-traitement officiel de nnU-Net v2 (« ne garder que le plus grand morceau »)
sur les prédictions hors pli des 5 plis, et montrer ce qu'il change : Dice et nombre de morceaux, avant / après.

ATTENTION : CONTRAIREMENT À 05d-05h (lecture seule), CE SCRIPT FAIT ÉCRIRE nnU-Net DANS nnUNet_results.
La commande officielle lancée est :
    nnUNetv2_find_best_configuration 101 -c 3d_fullres -tr nnUNetTrainer_250epochs_reprise10 -p nnUNetPlans
                                     -f 0 1 2 3 4 --disable_ensembling -np <processus>
Ce qu'elle écrit (code de nnunetv2 2.8.1, vérifié) :
  .../Dataset101_MMWHS/nnUNetTrainer_250epochs_reprise10__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/
      COPIES des 15 prédictions hors pli (<cas>.nii.gz), dataset.json, plans.json, summary.json (Dice AVANT) ;
      postprocessing.pkl et postprocessing.json (post-traitement retenu, Dice avant / après) ;
      postprocessed/ : les 15 prédictions post-traitées + summary.json (Dice APRÈS).
      Si ce dossier existe déjà, nnU-Net l'EFFACE puis le refait (option par défaut « overwrite ») ;
  .../Dataset101_MMWHS/inference_information.json et inference_instructions.txt (remplacés s'ils existent).
Elle ne modifie PAS fold_X/validation (elle y lit les prédictions et les copie). Le script le VÉRIFIE :
empreinte SHA-256 de chaque fichier de fold_X/validation avant et après, et état (taille, date) de tous les
autres fichiers de nnUNet_results/Dataset101_MMWHS ; toute écriture inattendue = code 6.

Ce que fait le script :
  0. garde-fous : --sortie hors du dépôt et des dossiers nnU-Net ; splits_final.json vérifié comme dans
     05b/05c/05f ; nnunetv2 en version 2.8.1 (la règle ci-dessous a été lue dans ce code-là) ; les 5 plis
     TERMINÉS (checkpoint_final.pth, validation/summary.json, aucun verrou d'entraînement récent, prédictions
     de validation = EXACTEMENT les 3 cas de validation du pli) ; références : gt_segmentations (dans
     nnUNet_preprocessed) et labelsTr contiennent les 15 cas d'entraînement, aucun cas de test, et sont
     identiques octet pour octet. Jamais de lecture de imagesTs ni de labelsTs, ni d'un cas de test
     (1004, 1010, 1011, 1013, 1014) ;
  1. lance la commande (sauf --analyser-seulement), en recopiant sa sortie dans --sortie ;
  2. relit, EN LECTURE SEULE, ce que nnU-Net a écrit : post-traitement retenu (postprocessing.json), Dice
     avant (crossval_results_folds_0_1_2_3_4/summary.json) et après (postprocessed/summary.json), par
     structure (étiquettes 1 à 7) et par cas ; contrôle que le Dice « avant » est celui des 5 summary.json
     des plis ;
  3. compte les MORCEAUX de chaque structure, avant et après, pour les 15 cas (même méthode que mesurer()
     de 05d : scipy.ndimage.label, connectivité 26) ; vérifie que le post-traitement n'a fait QUE retirer
     des voxels (remis à 0, le fond) ; volume retiré par structure ;
  4. moyennes « comme nnU-Net » (sur les cas) et par patient (paires 1001/1020 et 1005/1012 moyennées) ;
  5. trace l'essai dans MLflow (base SQLite dans Drive, comme 05b/05c), sauf --sans-mlflow ou
     --analyser-seulement.
  Le script NE DÉCIDE PAS d'appliquer le post-traitement au test, et NE CALIBRE AUCUN seuil du contrôle de
  plausibilité (D32) : il détermine et montre ; la décision reste humaine.

Règle de décision de nnU-Net v2 2.8.1 (nnunetv2/postprocessing/remove_connected_components.py,
fonction determine_postprocessing ; appelée par nnunetv2/evaluation/find_best_configuration.py) :
  a. « tout le premier plan » : les étiquettes 1 à 7 sont réunies en un seul masque (le cœur entier) ; on
     ne garde que son plus grand morceau, le reste devient du fond. Retenu SI le Dice moyen (moyenne des
     7 structures) AUGMENTE strictement ET qu'AUCUNE structure ne voit son Dice moyen baisser ;
  b. puis structure par structure, dans l'ordre 1 à 7, en partant du résultat précédent (a si retenu) : on
     ne garde que le plus grand morceau de CETTE étiquette. Retenu SI le Dice moyen de cette structure
     augmente strictement (les autres structures ne bougent pas : seuls des voxels de celle-ci sont retirés).
  Dice moyen d'une structure = moyenne sur les 15 cas (les cas où il n'est pas défini sont ignorés).
  Connectivité : skimage.measure.label sans argument (acvl_utils, remove_all_but_largest_component), soit
  la connectivité complète en 3D = 26 voisins. Les voxels retirés deviennent du fond (0), jamais une autre
  étiquette. Les 15 cas comptent chacun une fois (pas de moyenne par patient dans la règle de nnU-Net).

Entrées :
  --raw           nnUNet_raw (défaut /content/drive/MyDrive/poc_ct/nnunet_raw) : labelsTr (lu par nnU-Net
                  et par ce script pour l'empreinte), dataset.json, en-têtes de imagesTr (contrôle des plis)
  --preprocessed  nnUNet_preprocessed (défaut .../poc_ct/nnunet_preprocessed) : splits_final.json et fiches
                  de 05a, nnUNetPlans.json, gt_segmentations/ (lus)
  --results       nnUNet_results (défaut .../poc_ct/nnunet_results) : fold_X/ lus ; nnU-Net y écrit (voir haut)
  --sortie        dossier des résultats du script (défaut .../poc_ct/analyses/05i_postraitement)
  --mlflow        dossier MLflow (défaut .../poc_ct/mlflow)
  --split         découpage figé (défaut ../splits/mmwhs_split.json)
  --processus     processus de nnU-Net (-np, défaut 2 : Colab a 2 processeurs)
  --analyser-seulement  ne lance pas nnU-Net : relit seulement ce qu'il a déjà écrit (pas de MLflow)
  --sans-mlflow   ne trace rien dans MLflow
Sorties (dans --sortie) : journal_nnunet_05i.txt (sortie de nnU-Net), postraitement_05i.csv (une ligne par
  cas : Dice avant / après et morceaux avant / après par structure), synthese_05i.json.
Codes de sortie : 0 = terminé ; 1 = garde-fou (nnU-Net n'est pas lancé, rien n'est écrit) ; 3 = nnU-Net
  s'est arrêté en erreur ; 6 = incohérence (prédiction d'origine modifiée, écriture inattendue, Dice
  « avant » différent de celui des plis, post-traitement qui ajoute des voxels...) : ne pas utiliser.

Vocabulaire :
  - composante connexe (« morceau ») : groupe de voxels d'une même étiquette qui se touchent ; deux morceaux
    séparés par au moins un voxel d'une autre étiquette ou du fond sont deux composantes ;
  - connectivité 26 : deux voxels se touchent s'ils partagent une face (6 voisins), une arête (12) ou un
    coin (8), soit 26 voisins possibles ;
  - post-traitement : correction automatique appliquée APRÈS la prédiction du réseau ; ici, effacer les
    petits morceaux isolés ;
  - prédiction hors pli : prédiction d'un cas par le modèle qui ne l'a pas vu à l'entraînement ;
  - premier plan (foreground) : tout ce qui n'est pas du fond (étiquettes 1 à 7 réunies).
"""
import argparse, csv, datetime, hashlib, importlib.util, json, math, os, re, shutil, statistics, sys, time

ICI = os.path.dirname(os.path.abspath(__file__))


def charger_module(nom, fichier):
    spec = importlib.util.spec_from_file_location(nom, os.path.join(ICI, fichier))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


essai = charger_module("essai05b", "05b_essai_nnunet.py")    # contrôle de splits_final.json, MLflow
ana = charger_module("analyse05d", "05d_analyser_1019.py")    # noms (trainer, jeu), étiquettes, dossiers interdits
prep = essai.prep                                             # module du script 05a

# --- Paramètres fixes -------------------------------------------------------------------------
NOM_DS, TRAINER, CONFIGURATION, PLANS = ana.NOM_DS, ana.TRAINER, ana.CONFIGURATION, "nnUNetPlans"
ID_DATASET = 101
PLIS = (0, 1, 2, 3, 4)
VERSION_NNUNET = "2.8.1"                     # version dont le code a été lu pour la règle de décision
STRUCTURES = {1: "myo_VG", 2: "OG", 3: "VG", 4: "OD", 5: "VD", 6: "aorte", 7: "AP"}   # comme 05f
CONNEXITE = "26 (scipy.ndimage.label, structure 3×3×3, comme mesurer() de 05d)"
MORCEAU_ML = 0.5                             # définition de la MESURE de R8 (spécification § 5), pas un seuil
TOLERANCE_DICE = 1e-6                        # Dice « avant » de nnU-Net (crossval) contre celui des plis
VERROU_MINUTES = 15                          # comme 05c : verrou plus récent = entraînement peut-être en cours
NOM_CV = "crossval_results_folds_" + "_".join(map(str, PLIS))
FICHIERS_DATASET_AUTORISES = ("inference_information.json", "inference_instructions.txt")
NOM_JOURNAL, NOM_CSV, NOM_SYNTHESE = "journal_nnunet_05i.txt", "postraitement_05i.csv", "synthese_05i.json"


def arret(code, message):
    print(f"ARRÊT : {message}")
    sys.exit(code)


def lire_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=f"{prep.DRIVE}/nnunet_raw")
    p.add_argument("--preprocessed", default=f"{prep.DRIVE}/nnunet_preprocessed")
    p.add_argument("--results", default=f"{prep.DRIVE}/nnunet_results")
    p.add_argument("--sortie", default=f"{prep.DRIVE}/analyses/05i_postraitement")
    p.add_argument("--mlflow", default=f"{prep.DRIVE}/mlflow")
    p.add_argument("--split", default=os.path.join(prep.DEPOT, "splits", "mmwhs_split.json"))
    p.add_argument("--processus", type=int, default=2, choices=[1, 2, 3, 4])
    p.add_argument("--analyser-seulement", action="store_true")
    p.add_argument("--sans-mlflow", action="store_true")
    return p.parse_args()


def sha256(chemin):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def test_dans_chemin(chemin):
    return any(d in ("imagesTs", "labelsTs") for d in re.split(r"[\\/]", chemin))


# --- 0. Garde-fous ----------------------------------------------------------------------------
def verifier_dossiers(a, raw, pre, res):
    """--sortie hors du dépôt et des dossiers nnU-Net (règle de 05d/05f) ; --mlflow hors du dépôt."""
    sortie = os.path.abspath(a.sortie)
    interdits = [d for d in sortie.split(os.sep) if ana.DOSSIERS_INTERDITS.match(d)]
    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if os.environ.get(var) and prep.dans(sortie, os.path.abspath(os.environ[var])):
            interdits.append(f"${var}")
    if prep.dans(sortie, prep.DEPOT) or any(prep.dans(sortie, d) for d in (raw, pre, res)) or interdits:
        arret(1, f"--sortie ({sortie}) est dans le dépôt ou dans un dossier nnU-Net ({interdits}). "
                 "Choisir un dossier de /content/drive/MyDrive/poc_ct/analyses/.")
    if prep.dans(os.path.abspath(a.mlflow), prep.DEPOT):
        arret(1, "--mlflow est dans le dépôt : les résultats restent hors du dépôt.")
    return sortie


def dossier_modele(res):
    return os.path.join(res, NOM_DS, f"{TRAINER}__{PLANS}__{CONFIGURATION}")


def verifier_plis_termines(res, plis, cas_test):
    """Les 5 plis doivent être terminés : sinon nnU-Net déciderait sur une partie des cas seulement."""
    pb = []
    for k in PLIS:
        d = os.path.join(dossier_modele(res), f"fold_{k}")
        val = set(plis[k]["val"])
        if val & cas_test:
            arret(1, f"pli {k} : un cas de TEST est en validation ({sorted(val & cas_test)}).")
        if not os.path.isfile(os.path.join(d, "checkpoint_final.pth")):
            pb.append(f"pli {k} : checkpoint_final.pth absent (entraînement non terminé)")
        if not os.path.isfile(os.path.join(d, "validation", "summary.json")):
            pb.append(f"pli {k} : validation/summary.json absent (validation finale non terminée)")
        verrou = os.path.join(d, "entrainement_en_cours.verrou")
        if os.path.isfile(verrou) and (time.time() - os.path.getmtime(verrou)) / 60 < VERROU_MINUTES:
            pb.append(f"pli {k} : verrou d'entraînement récent ({verrou}) : une session 05c tourne peut-être")
        vus = {n[:-len(".nii.gz")] for n in (os.listdir(os.path.join(d, "validation"))
                                              if os.path.isdir(os.path.join(d, "validation")) else [])
               if n.endswith(".nii.gz")}
        if vus != val:
            pb.append(f"pli {k} : prédictions de validation {sorted(vus)} ≠ cas de validation du pli {sorted(val)}")
    if pb:
        arret(1, "les 5 plis ne sont pas tous terminés et conformes :\n  - " + "\n  - ".join(pb))
    print("Les 5 plis sont terminés ; chaque dossier fold_X/validation contient exactement les 3 cas de son pli.")


def verifier_references(raw, pre, res, cas, cas_test):
    """gt_segmentations (référence du Dice « après ») et labelsTr (référence du Dice « avant », choisie par
    accumulate_cv_results quand nnUNet_raw existe) : les 15 cas d'entraînement, aucun cas de test, et
    identiques octet pour octet. Vérifie aussi dataset.json, plans.json et nnUNetPlans.json."""
    pb = []
    gt, ltr = os.path.join(pre, NOM_DS, "gt_segmentations"), os.path.join(raw, NOM_DS, "labelsTr")
    for nom, d in (("gt_segmentations", gt), ("labelsTr", ltr)):
        if not os.path.isdir(d):
            arret(1, f"{d} introuvable.")
        presents = {n[:-len(".nii.gz")] for n in os.listdir(d) if n.endswith(".nii.gz")}
        if presents & cas_test:
            pb.append(f"{nom} contient des cas de TEST : {sorted(presents & cas_test)}")
        if presents != set(cas):
            pb.append(f"{nom} : cas {sorted(presents)} ≠ les 15 cas d'entraînement")
    for c in sorted(cas):
        a, b = os.path.join(gt, f"{c}.nii.gz"), os.path.join(ltr, f"{c}.nii.gz")
        if os.path.isfile(a) and os.path.isfile(b) and sha256(a) != sha256(b):
            pb.append(f"{c} : gt_segmentations et labelsTr diffèrent (Dice avant et après ne seraient pas "
                      "mesurés sur la même vérité)")
    for chemin in (os.path.join(pre, NOM_DS, f"{PLANS}.json"), os.path.join(dossier_modele(res), "plans.json"),
                   os.path.join(dossier_modele(res), "dataset.json")):
        if not os.path.isfile(chemin):
            pb.append(f"{chemin} absent")
    if not pb:
        with open(os.path.join(dossier_modele(res), "dataset.json"), encoding="utf-8") as f:
            ds = json.load(f)
        if ds.get("labels") != ana.ETIQUETTES or ds.get("file_ending") != ".nii.gz":
            pb.append(f"dataset.json du modèle : étiquettes {ds.get('labels')} ou extension {ds.get('file_ending')} "
                      "inattendues")
    if pb:
        arret(1, "références ou fichiers de nnU-Net non conformes :\n  - " + "\n  - ".join(pb))
    print("Références conformes : gt_segmentations = labelsTr (15 cas d'entraînement, aucun cas de test).")


def photographier(res):
    """État de nnUNet_results/Dataset101_MMWHS, sans le dossier que nnU-Net va refaire :
    SHA-256 des fichiers de fold_X/validation (les prédictions d'origine) ; taille et date des autres."""
    racine, cv = os.path.join(res, NOM_DS), os.path.join(dossier_modele(res), NOM_CV)
    etat = {}
    for r, dossiers, fichiers in os.walk(racine):
        if prep.dans(r, cv):
            dossiers[:] = []
            continue
        for n in fichiers:
            p = os.path.join(r, n)
            dans_validation = os.path.basename(r) == "validation" and os.path.basename(os.path.dirname(r)).startswith("fold_")
            st = os.stat(p)
            # Date à la seconde : Drive peut arrondir les fractions ; l'empreinte reste le contrôle strict.
            etat[os.path.relpath(p, racine)] = ("sha256", sha256(p)) if dans_validation else (st.st_size, int(st.st_mtime))
    return etat


def comparer_photos(avant, apres):
    """Renvoie (fichiers écrits attendus, écritures INATTENDUES, fichiers dont SEULE la date a changé).
    Hors prédictions d'origine, « même taille, date seule changée » n'est qu'un AVERTISSEMENT : Google Drive
    peut resynchroniser une date sans que le fichier change. Les prédictions d'origine restent contrôlées
    strictement par leur empreinte SHA-256 (toute différence = écriture inattendue)."""
    attendus, inattendus, dates_seules = [], [], []
    for p in sorted(set(avant) | set(apres)):
        if avant.get(p) == apres.get(p):
            continue
        quoi = "créé" if p not in avant else ("supprimé" if p not in apres else "modifié")
        if p in FICHIERS_DATASET_AUTORISES and quoi != "supprimé":
            attendus.append(f"{p} ({quoi})")
        elif quoi == "modifié" and avant[p][0] != "sha256" and avant[p][0] == apres[p][0]:
            dates_seules.append(p)                                 # même taille, seule la date a bougé
        else:
            inattendus.append(f"{p} ({quoi})")
    return attendus, inattendus, dates_seules


# --- 1. Lancement de la commande officielle ---------------------------------------------------
def lancer_nnunet(a, raw, pre, res, sortie):
    exe = shutil.which("nnUNetv2_find_best_configuration")
    if exe is None:
        arret(1, "nnUNetv2_find_best_configuration introuvable : installer nnU-Net (notebook, section 12).")
    commande = [exe, str(ID_DATASET), "-c", CONFIGURATION, "-tr", TRAINER, "-p", PLANS,
                "-f", *map(str, PLIS), "--disable_ensembling", "-np", str(a.processus)]
    env = prep.environnement_nnunet(raw, pre, res, a.processus)
    cv = os.path.join(dossier_modele(res), NOM_CV)
    print("\nnnU-Net va ÉCRIRE dans nnUNet_results (et seulement là) :")
    print(f"  {cv}/  {'(EXISTE DÉJÀ : nnU-Net va l’effacer puis le refaire)' if os.path.isdir(cv) else '(nouveau)'}")
    for n in FICHIERS_DATASET_AUTORISES:
        print(f"  {os.path.join(res, NOM_DS, n)}  ({'remplacé' if os.path.isfile(os.path.join(res, NOM_DS, n)) else 'nouveau'})")
    print("Les prédictions d'origine (fold_X/validation) sont seulement LUES et copiées.\n", flush=True)
    t0 = time.time()
    code, _ = prep.lancer(commande, env, os.path.join(sortie, NOM_JOURNAL))
    duree = time.time() - t0
    print(f"\nnnU-Net terminé (code {code}) en {duree / 60:.1f} min.")
    return code, " ".join(commande[1:]), duree


# --- 2. Lecture de ce qu'a écrit nnU-Net -------------------------------------------------------
def nombre(x):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def lire_summary(chemin, cas, cas_test):
    """{cas: {étiquette: {dice, n_ref, n_pred}}} + Dice moyen écrit par nnU-Net. Arrêt 6 si non conforme."""
    if not os.path.isfile(chemin):
        arret(6, f"{chemin} absent : nnU-Net n'a pas terminé son travail.")
    with open(chemin, encoding="utf-8") as f:
        s = json.load(f)
    lus = {}
    for e in s.get("metric_per_case", []):
        nom = os.path.basename(e.get("prediction_file", ""))[:-len(".nii.gz")]
        if nom in cas_test or test_dans_chemin(e.get("reference_file", "")) or test_dans_chemin(e.get("prediction_file", "")):
            arret(6, f"{chemin} : cas de TEST ou dossier de test ({nom}, {e.get('reference_file')}).")
        lus[nom] = {k: {"dice": nombre((e["metrics"].get(str(k)) or {}).get("Dice")),
                        "n_ref": int((e["metrics"].get(str(k)) or {}).get("n_ref", 0)),
                        "n_pred": int((e["metrics"].get(str(k)) or {}).get("n_pred", 0))} for k in STRUCTURES}
    if set(lus) != set(cas):
        arret(6, f"{chemin} : cas {sorted(lus)} ≠ les 15 cas d'entraînement.")
    return lus, nombre((s.get("foreground_mean") or {}).get("Dice"))


def lire_summary_pli(chemin, val, cas_test):
    """Dice par cas et par étiquette d'un summary.json de pli (référence du contrôle « avant »)."""
    lus, _ = lire_summary(chemin, set(val), cas_test)
    return {c: {k: m[k]["dice"] for k in STRUCTURES} for c, m in lus.items()}


def lire_postraitement(cv):
    """Post-traitement retenu, d'après postprocessing.json (le .pkl contient la même chose, sous forme de
    fonctions Python : on ne le désérialise pas)."""
    chemin = os.path.join(cv, "postprocessing.json")
    if not os.path.isfile(chemin):
        arret(6, f"{chemin} absent.")
    with open(chemin, encoding="utf-8") as f:
        pp = json.load(f)
    etapes = []
    for fn, kw in zip(pp.get("postprocessing_fns", []), pp.get("postprocessing_kwargs", [])):
        cible = kw.get("labels_or_regions")
        if fn != "remove_all_but_largest_component_from_segmentation":
            arret(6, f"post-traitement inattendu dans postprocessing.json : {fn}")
        if isinstance(cible, list):
            etapes.append({"etape": "a", "cible": "premier plan (étiquettes 1 à 7 réunies)", "etiquettes": cible})
        else:
            etapes.append({"etape": "b", "cible": f"{cible} {STRUCTURES.get(int(cible), '?')}", "etiquettes": [int(cible)]})
    return pp, etapes


def lignes_de_decision(sortie):
    """Les phrases de décision affichées par nnU-Net, relues dans le journal (les dossiers intermédiaires
    sont effacés par nnU-Net : c'est la seule trace des étapes non retenues)."""
    chemin = os.path.join(sortie, NOM_JOURNAL)
    if not os.path.isfile(chemin):
        return []
    with open(chemin, encoding="utf-8", errors="replace") as f:
        texte = f.read().split("\n===== ")[-1]           # dernier lancement seulement
    return [l.strip() for l in texte.splitlines()
            if l.startswith(("Results were improved", "Removing all but"))]


# --- 3. Morceaux -------------------------------------------------------------------------------
def morceaux(m, vox_ml):
    """Nombre de morceaux (connectivité 26, comme mesurer() de 05d), nombre de morceaux > 0,5 mL, part du
    plus gros dans le volume de la structure, volume total (mL)."""
    import numpy as np
    from scipy import ndimage
    n = int(m.sum())
    if n == 0:
        return {"n": 0, "n_gros": 0, "part_plus_gros": None, "ml": 0.0}
    bornes = [np.where(m.any(axis=tuple(x for x in range(3) if x != ax)))[0] for ax in range(3)]
    boite = m[tuple(slice(int(b.min()), int(b.max()) + 1) for b in bornes)]      # recadré : moins de mémoire
    etiq, nb = ndimage.label(boite, structure=np.ones((3, 3, 3)))
    tailles = np.bincount(etiq.ravel())[1:]
    return {"n": int(nb), "n_gros": int((tailles * vox_ml > MORCEAU_ML).sum()),
            "part_plus_gros": float(tailles.max() / n), "ml": n * vox_ml}


def analyser_cas(c, pli, cv, res, resume_apres):
    """Lit la prédiction d'origine (copie de nnU-Net, comparée à l'originale) et la prédiction post-traitée."""
    import nibabel as nib
    import numpy as np
    orig = os.path.join(dossier_modele(res), f"fold_{pli}", "validation", f"{c}.nii.gz")
    copie, post = os.path.join(cv, f"{c}.nii.gz"), os.path.join(cv, "postprocessed", f"{c}.nii.gz")
    pb = []
    for p in (copie, post):
        if not os.path.isfile(p):
            arret(6, f"{p} absent.")
    if sha256(copie) != sha256(orig):
        pb.append(f"{c} : la copie de nnU-Net diffère de la prédiction d'origine du pli {pli}")
    ia, ib = nib.load(copie), nib.load(post)
    if ia.shape != ib.shape or not np.allclose(ia.affine, ib.affine, atol=1e-3):
        arret(6, f"{c} : forme ou affine de la prédiction post-traitée ≠ prédiction d'origine.")
    avant, apres = np.asanyarray(ia.dataobj).astype(np.uint8), np.asanyarray(ib.dataobj).astype(np.uint8)
    change = avant != apres
    if (apres[change] != 0).any():
        pb.append(f"{c} : le post-traitement a CHANGÉ l'étiquette de voxels au lieu de les remettre au fond")
    vox_ml = float(np.prod(ia.header.get_zooms()[:3])) / 1000
    r = {"pli": pli, "vox_ml": vox_ml, "voxels_retires": int(change.sum()),
         "premier_plan": {"avant": morceaux(avant > 0, vox_ml), "apres": morceaux(apres > 0, vox_ml)}, "structures": {}}
    for k in STRUCTURES:
        ma, mb = avant == k, apres == k
        if int(mb.sum()) != resume_apres[c][k]["n_pred"]:
            pb.append(f"{c} étiquette {k} : {int(mb.sum())} voxels lus ≠ {resume_apres[c][k]['n_pred']} dans summary.json")
        r["structures"][k] = {"avant": morceaux(ma, vox_ml), "apres": morceaux(mb, vox_ml),
                              "ml_retire": float((ma & ~mb).sum()) * vox_ml}
    return r, pb


# --- 4. Moyennes -------------------------------------------------------------------------------
def moyenne(v):
    v = [x for x in v if x is not None]
    return statistics.fmean(v) if v else None


def moyennes(dice, groupes):
    """Par structure : moyenne sur les cas (comme nnU-Net : cas non définis ignorés) et par patient ;
    puis Dice moyen = moyenne des 7 structures (foreground_mean de nnU-Net)."""
    patient_de = {c: "+".join(x[-4:] for x in g) for g in groupes for c in g}
    par_patient = {}
    for c in dice:
        par_patient.setdefault(patient_de.get(c, c[-4:]), []).append(c)
    m = {"sur_les_cas": {}, "par_patient": {}}
    for k in STRUCTURES:
        m["sur_les_cas"][k] = moyenne([dice[c][k]["dice"] for c in dice])
        m["par_patient"][k] = moyenne([moyenne([dice[c][k]["dice"] for c in cs]) for cs in par_patient.values()])
    for cle in ("sur_les_cas", "par_patient"):
        m[cle]["moyenne"] = moyenne([m[cle][k] for k in STRUCTURES])
    return m, patient_de


# --- Affichage ---------------------------------------------------------------------------------
def f3(x, signe=False):
    return "   —  " if x is None else (f"{x:+.3f}" if signe else f"{x:.3f}")


def afficher_dice(titre, valeurs, moy, signe=False):
    print(f"\n{titre}")
    print(f"  {'cas':<6}{'pli':>4}  " + "".join(f"{n:>8}" for n in STRUCTURES.values()) + f"{'moy. cas':>10}")
    for c, (pli, d) in valeurs.items():
        print(f"  {c[-4:]:<6}{pli:>4}  " + "".join(f"{f3(d[k], signe):>8}" for k in STRUCTURES) + f"{f3(d['moyenne'], signe):>10}")
    for libelle, cle in (("moyenne sur les cas (comme nnU-Net)", "sur_les_cas"), ("moyenne PAR PATIENT", "par_patient")):
        print(f"  {libelle}")
        print(f"  {'':<10}  " + "".join(f"{f3(moy[cle][k], signe):>8}" for k in STRUCTURES) + f"{f3(moy[cle]['moyenne'], signe):>10}")


def afficher_morceaux(titre, analyses, quand):
    print(f"\n{titre}  (cellule = morceaux au total / morceaux de plus de {MORCEAU_ML} mL)")
    print(f"  {'cas':<6}{'pli':>4}  " + "".join(f"{n:>8}" for n in STRUCTURES.values()) + f"{'1 à 7':>9}")
    for c, r in analyses.items():
        cell = lambda m: f"{m['n']}/{m['n_gros']}" + ("*" if m["n"] > 1 else " ")
        print(f"  {c[-4:]:<6}{r['pli']:>4}  " + "".join(f"{cell(r['structures'][k][quand]):>8}" for k in STRUCTURES)
              + f"{cell(r['premier_plan'][quand]):>9}")
    plusieurs = {k: sum(r["structures"][k][quand]["n"] > 1 for r in analyses.values()) for k in STRUCTURES}
    print(f"  {'cas en plusieurs morceaux':<12}" + "".join(f"{plusieurs[k]:>8}" for k in STRUCTURES)
          + f"{sum(r['premier_plan'][quand]['n'] > 1 for r in analyses.values()):>9}")
    print("  * = plus d'un morceau. « 1 à 7 » = le cœur entier (toutes les étiquettes réunies), ce que regarde l'étape a.")


# --- Programme principal -----------------------------------------------------------------------
def main():
    a = lire_arguments()
    raw, pre, res = (os.path.abspath(x) for x in (a.raw, a.preprocessed, a.results))
    sortie = verifier_dossiers(a, raw, pre, res)
    try:
        cas, cas_test, groupes, _ = prep.lire_decoupage(a.split)        # lecture seule du découpage figé
    except (OSError, ValueError, KeyError) as err:
        arret(1, f"découpage figé illisible ou incohérent ({a.split}) : {err}")
    cas, cas_test = set(cas), set(cas_test)
    _, _, _, trace, plis = essai.verifier_prerequis_05a(raw, pre, NOM_DS, a.split, 0)
    print("(La ligne « Pli 0 » ci-dessus vient du contrôle commun avec 05c ; ce script porte sur les 5 plis.)")
    pli_de = {c: k for k in PLIS for c in plis[k]["val"]}
    if set(pli_de) != cas:
        arret(1, f"les cas de validation des 5 plis {sorted(pli_de)} ≠ les 15 cas d'entraînement.")
    version = essai.version_paquet("nnunetv2")
    print(f"nnunetv2 {version} ; acvl-utils {essai.version_paquet('acvl-utils')} ; scikit-image "
          f"{essai.version_paquet('scikit-image')}")
    if version != VERSION_NNUNET and not a.analyser_seulement:
        arret(1, f"nnunetv2 {version} installé, mais la règle de décision a été vérifiée dans la version "
                 f"{VERSION_NNUNET} : réinstaller la version épinglée (section 12).")
    verifier_plis_termines(res, plis, cas_test)
    verifier_references(raw, pre, res, cas, cas_test)
    cv = os.path.join(dossier_modele(res), NOM_CV)

    # --- 1. Commande officielle, avec photographie de nnUNet_results avant / après ---------------------
    commande, duree, ecrits = None, None, []
    if not a.analyser_seulement:
        print("Empreintes des prédictions d'origine et état de nnUNet_results (avant)...", flush=True)
        photo_avant = photographier(res)
        os.makedirs(sortie, exist_ok=True)
        code, commande, duree = lancer_nnunet(a, raw, pre, res, sortie)
        photo_apres = photographier(res)
        ecrits, inattendus, dates_seules = comparer_photos(photo_avant, photo_apres)
        for p in dates_seules:
            print(f"AVERTISSEMENT : {p} : même taille, seule la date a changé (resynchronisation de Drive ?). "
                  "Les prédictions d'origine, elles, sont contrôlées par empreinte.")
        if inattendus:
            arret(6, "écritures INATTENDUES dans nnUNet_results (dont, peut-être, des prédictions d'origine) :\n  - "
                     + "\n  - ".join(inattendus))
        print(f"Contrôle : les {sum(1 for p in photo_avant if photo_avant[p][0] == 'sha256')} fichiers de "
              "fold_X/validation sont INCHANGÉS (empreintes identiques) ; hors du dossier "
              f"{NOM_CV}/, seuls ont été écrits : {', '.join(ecrits) or 'aucun'}.")
        if code:
            arret(3, f"nnU-Net s'est arrêté en erreur (code {code}) : voir sa sortie ci-dessus et {NOM_JOURNAL}.")
    elif not os.path.isdir(cv):
        arret(1, f"{cv} absent : lancer d'abord le script sans --analyser-seulement.")
    else:
        print(f"--analyser-seulement : nnU-Net n'est pas lancé ; relecture de {cv}.")

    # --- 2. Dice avant / après et post-traitement retenu ------------------------------------------------
    avant, fg_avant = lire_summary(os.path.join(cv, "summary.json"), cas, cas_test)
    apres, fg_apres = lire_summary(os.path.join(cv, "postprocessed", "summary.json"), cas, cas_test)
    pp, etapes = lire_postraitement(cv)
    problemes = []
    for k in PLIS:                     # Dice « avant » = celui des 5 validations finales ?
        plis_lus = lire_summary_pli(os.path.join(dossier_modele(res), f"fold_{k}", "validation", "summary.json"),
                                       plis[k]["val"], cas_test)
        for c, m in plis_lus.items():
            for e in STRUCTURES:
                x, y = m[e], avant[c][e]["dice"]
                if (x is None) != (y is None) or (x is not None and abs(x - y) > TOLERANCE_DICE):
                    problemes.append(f"{c} étiquette {e} : Dice {y} (crossval) ≠ {x} (pli {k})")
    dice = {"avant": avant, "apres": apres}
    moy = {q: moyennes(dice[q], groupes)[0] for q in dice}
    patient_de = moyennes(avant, groupes)[1]
    for q, fg in (("avant", fg_avant), ("apres", fg_apres)):
        if fg is None or abs(fg - moy[q]["sur_les_cas"]["moyenne"]) > 1e-6:
            problemes.append(f"Dice moyen {q} recalculé {moy[q]['sur_les_cas']['moyenne']} ≠ écrit par nnU-Net {fg}")

    # --- 3. Morceaux, cas par cas (un cas en mémoire à la fois) ------------------------------------------
    print(f"\nComptage des morceaux (connectivité {CONNEXITE}) sur les 15 cas, avant et après...", flush=True)
    analyses = {}
    for c in sorted(cas):
        analyses[c], pb = analyser_cas(c, pli_de[c], cv, res, apres)
        problemes += pb
    if problemes:
        arret(6, "incohérences, résultats à NE PAS utiliser :\n  - " + "\n  - ".join(problemes))

    # --- Affichage ---------------------------------------------------------------------------------------
    print("\n" + "=" * 100 + "\nPOST-TRAITEMENT DE nnU-Net SUR LES 15 PRÉDICTIONS HORS PLI (validation, PAS le test)\n" + "=" * 100)
    print("Décisions affichées par nnU-Net (étape a = cœur entier, puis étape b = structure par structure) :")
    for l in lignes_de_decision(sortie) or ["(journal absent : voir postprocessing.json ci-dessous)"]:
        print(f"  {l}")
    print("\nPOST-TRAITEMENT RETENU par nnU-Net (postprocessing.json) :")
    for e in etapes or [{"etape": "-", "cible": "AUCUN : aucune étape n'a amélioré le Dice moyen"}]:
        print(f"  étape {e['etape']} : ne garder que le plus grand morceau de : {e['cible']}")
    valeurs = lambda q: {c: (pli_de[c], {**{k: dice[q][c][k]["dice"] for k in STRUCTURES},
                                         "moyenne": moyenne([dice[q][c][k]["dice"] for k in STRUCTURES])})
                         for c in sorted(cas)}
    va, vb = valeurs("avant"), valeurs("apres")
    ecart = {c: (va[c][0], {k: (None if va[c][1][k] is None or vb[c][1][k] is None else vb[c][1][k] - va[c][1][k])
                            for k in [*STRUCTURES, "moyenne"]}) for c in va}
    moy_ecart = {cle: {k: (None if moy["avant"][cle][k] is None or moy["apres"][cle][k] is None
                           else moy["apres"][cle][k] - moy["avant"][cle][k]) for k in [*STRUCTURES, "moyenne"]}
                 for cle in ("sur_les_cas", "par_patient")}
    afficher_dice("DICE AVANT post-traitement :", va, moy["avant"])
    afficher_dice("DICE APRÈS post-traitement :", vb, moy["apres"])
    afficher_dice("ÉCART (après − avant) ; +0.000 = inchangé :", ecart, moy_ecart, signe=True)
    afficher_morceaux("MORCEAUX AVANT post-traitement :", analyses, "avant")
    afficher_morceaux("MORCEAUX APRÈS post-traitement :", analyses, "apres")
    print(f"\nVolume retiré par le post-traitement (mL), par cas et par structure :")
    print(f"  {'cas':<6}" + "".join(f"{n:>8}" for n in STRUCTURES.values()))
    for c, r in analyses.items():
        print(f"  {c[-4:]:<6}" + "".join(f"{r['structures'][k]['ml_retire']:>8.2f}" for k in STRUCTURES))
    print("\nÀ RETENIR :")
    print("  - nnU-Net juge sur le Dice MOYEN des 15 cas (chaque cas compte une fois, paires comprises). Un petit")
    print("    fragment pèse peu dans le Dice : il peut rester si l'effacer n'améliore pas la moyenne ;")
    print("  - AUCUN seuil du contrôle de plausibilité n'est calibré ici (D32) ; le post-traitement n'est PAS appliqué")
    print("    au test. Décider (humain) d'abord, puis calibrer R8 sur les prédictions retenues.")
    print("  - inference_instructions.txt (écrit par nnU-Net) propose nnUNetv2_apply_postprocessing : ne PAS le lancer.")

    # --- Écriture ----------------------------------------------------------------------------------------
    os.makedirs(sortie, exist_ok=True)
    chemin_csv = os.path.join(sortie, NOM_CSV)
    with open(chemin_csv + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cas", "pli", "patient", "epaisseur_mm",
                    *[f"dice_{q}_{k}_{n}" for q in ("avant", "apres") for k, n in STRUCTURES.items()],
                    "dice_avant_moyenne_cas", "dice_apres_moyenne_cas",
                    *[f"morceaux_{q}_{k}_{n}" for q in ("avant", "apres") for k, n in STRUCTURES.items()],
                    *[f"morceaux_gros_{q}_{k}_{n}" for q in ("avant", "apres") for k, n in STRUCTURES.items()],
                    *[f"ml_retire_{k}_{n}" for k, n in STRUCTURES.items()]])
        arr = lambda x, n=4: "" if x is None else round(x, n)
        for c, r in analyses.items():
            w.writerow([c, r["pli"], patient_de.get(c, c[-4:]), trace["strates_mm"].get(c),
                        *[arr(dice[q][c][k]["dice"]) for q in ("avant", "apres") for k in STRUCTURES],
                        arr(va[c][1]["moyenne"]), arr(vb[c][1]["moyenne"]),
                        *[r["structures"][k][q]["n"] for q in ("avant", "apres") for k in STRUCTURES],
                        *[r["structures"][k][q]["n_gros"] for q in ("avant", "apres") for k in STRUCTURES],
                        *[arr(r["structures"][k]["ml_retire"], 3) for k in STRUCTURES]])
    os.replace(chemin_csv + ".tmp", chemin_csv)
    synthese = {"objet": "post-traitement nnU-Net déterminé sur les prédictions hors pli (validation, PAS le test)",
                "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "commande": commande, "duree_s": duree, "nnunetv2": version,
                "trainer": TRAINER, "configuration": CONFIGURATION, "plans": PLANS, "plis": list(PLIS),
                "split_sha256": prep.sha256_fichier(a.split), "code_commit": prep.commit_git(),
                "ecrit_par_nnunet": {"dossier": cv, "hors_dossier": ecrits},
                "regle_nnunet": "a) cœur entier : retenu si Dice moyen ↑ strictement et aucune structure ↓ ; "
                                "b) puis étiquettes 1 à 7 dans l'ordre : retenu si le Dice moyen de l'étiquette ↑ "
                                "strictement. Connectivité 26 (skimage.measure.label par défaut). Code : nnunetv2 "
                                "2.8.1, postprocessing/remove_connected_components.py, determine_postprocessing",
                "postraitement_retenu": etapes, "postprocessing_json": pp,
                "decisions_affichees": lignes_de_decision(sortie),
                "moyennes": {q: {cle: {str(k): v for k, v in moy[q][cle].items()} for cle in moy[q]} for q in moy},
                "connexite_morceaux": CONNEXITE, "definition_mesure_R8_ml": MORCEAU_ML,          # définition de la mesure (spéc.), PAS un seuil calibré
                "cas": {c: {"pli": r["pli"], "premier_plan": r["premier_plan"], "voxels_retires": r["voxels_retires"],
                            "structures": {str(k): {**r["structures"][k], "dice_avant": dice["avant"][c][k]["dice"],
                                                    "dice_apres": dice["apres"][c][k]["dice"]} for k in STRUCTURES}}
                        for c, r in analyses.items()},
                "seuils": None, "decision_application": "humaine, non prise par ce script"}
    chemin_synthese = os.path.join(sortie, NOM_SYNTHESE)
    with open(chemin_synthese + ".tmp", "w", encoding="utf-8") as f:
        json.dump(synthese, f, indent=2, ensure_ascii=False)
    os.replace(chemin_synthese + ".tmp", chemin_synthese)
    print(f"\nÉcrit dans {sortie} : {NOM_CSV}, {NOM_SYNTHESE}" + ("" if a.analyser_seulement else f", {NOM_JOURNAL}"))

    # --- 5. MLflow -----------------------------------------------------------------------------------------
    if a.analyser_seulement or a.sans_mlflow:
        print("MLflow : rien n'est tracé (--analyser-seulement ou --sans-mlflow).")
        return
    params = {"trainer": TRAINER, "configuration": CONFIGURATION, "plans": PLANS, "plis": "0,1,2,3,4",
              "commande": commande, "nnunetv2": version, "connexite_morceaux": "26",
              "postraitement_retenu": "; ".join(e["cible"] for e in etapes) or "aucun",
              "sha256_splits_final": trace["sha256_splits_final"]}
    metriques = {"dice_moyen_avant": moy["avant"]["sur_les_cas"]["moyenne"],
                 "dice_moyen_apres": moy["apres"]["sur_les_cas"]["moyenne"],
                 "dice_moyen_par_patient_avant": moy["avant"]["par_patient"]["moyenne"],
                 "dice_moyen_par_patient_apres": moy["apres"]["par_patient"]["moyenne"], "duree_s": duree}
    for k, n in STRUCTURES.items():
        metriques[f"dice_avant_{n}"], metriques[f"dice_apres_{n}"] = moy["avant"]["sur_les_cas"][k], moy["apres"]["sur_les_cas"][k]
        for q in ("avant", "apres"):
            metriques[f"cas_plusieurs_morceaux_{q}_{n}"] = sum(r["structures"][k][q]["n"] > 1 for r in analyses.values())
    tags = {"etape": "5", "type": "postraitement", "mlflow.source.git.commit": prep.commit_git()}
    artefacts = [os.path.join(cv, "postprocessing.json"), os.path.join(cv, "summary.json"), chemin_csv,
                 chemin_synthese, os.path.join(sortie, NOM_JOURNAL)]
    try:
        id_run = essai.tracer_mlflow(os.path.abspath(a.mlflow), "postraitement_nnunet_5plis", params, tags, {},
                                     metriques, artefacts)
        print(f"MLflow : run {id_run} (expérience {essai.EXPERIENCE_MLFLOW}).")
    except Exception as err:                       # le suivi ne doit pas faire perdre les résultats
        print(f"ATTENTION : traçage MLflow impossible ({err}) ; les résultats sont dans {sortie}.")


if __name__ == "__main__":
    main()
