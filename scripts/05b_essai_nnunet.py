"""
Étape 5b — Essai à blanc de nnU-Net v2 : 5 époques sur le pli 0, pour mesurer la durée d'une époque sur le T4.

Ce que fait le script :
  1. garde-fous : dossiers hors du dépôt, GPU présent, variante d'entraînement (trainer) disponible ;
     marqueur verification_05a.json présent (05a a vérifié image contre masque, avec SimpleITK et
     nnU-Net, sans aucun écart) ; splits_final.json existe ET est celui écrit par 05a : même empreinte
     que sa fiche de traçabilité, et mêmes plis une fois recalculés depuis le découpage figé, la graine et
     les épaisseurs de coupe relues dans les en-têtes de imagesTr (plis stratifiés, comme en 05a). Sinon
     nnU-Net tirerait ses plis au hasard, cas par cas ;
  2. (par défaut) copie les données prétraitées de Drive vers le disque local de Colab, beaucoup plus
     rapide à lire pendant l'entraînement. Les résultats et points de reprise restent écrits dans Drive ;
  3. lance `nnUNetv2_train 101 3d_fullres 0 -tr nnUNetTrainer_5epochs` (réglages par défaut de nnU-Net
     pour tout le reste). À la fin, nnU-Net fait aussi sa « validation finale » sur les cas du pli ;
  4. lit le journal d'entraînement de nnU-Net : confirme qu'il a utilisé NOTRE splits_final.json, relève
     pour chaque époque durée, pertes et Dice pseudo ; affiche la durée moyenne (sans la 1re époque, plus
     lente), le temps passé hors « Epoch time » (écriture des points de reprise dans Drive) et une
     extrapolation INDICATIVE (100, 250, 1000 époques). Rien n'est décidé ici ;
  5. trace l'essai dans MLflow (base SQLite et fichiers dans Drive, aucun serveur distant).

Entrées :
  --raw / --preprocessed / --results   dossiers nnU-Net dans Drive (mêmes défauts que 05a)
  --mlflow        dossier MLflow dans Drive (défaut /content/drive/MyDrive/poc_ct/mlflow)
  --split         découpage figé (défaut ../splits/mmwhs_split.json)
  --pli           pli à entraîner (défaut 0)
  --trainer       variante d'entraînement (défaut nnUNetTrainer_5epochs : 5 époques au lieu de 1000)
  --configuration défaut 3d_fullres
  --processus     nnUNet_def_n_proc : processus d'export pendant la validation finale (défaut 2, RAM)
  --processus-augmentation  nnUNet_n_proc_DA : processus d'augmentation de données. Défaut : celui de
                  nnU-Net, qui prend min(12, nombre de processeurs), soit 2 sur Colab gratuit
  --sans-compile  désactive torch.compile (nnUNet_compile=f). Par défaut nnU-Net compile le réseau
  --lire-depuis-drive  n'utilise pas de copie locale (déconseillé : lectures lentes depuis Drive)
  --local         dossier de la copie locale (défaut /content/nnunet_preprocessed_local)
  --continuer     reprend un entraînement interrompu (option --c de nnU-Net)
  --ecraser       supprime un essai déjà présent pour ce pli et ce trainer, puis recommence
  --tracer-seulement  ne lance rien : relit les journaux existants et trace dans MLflow
Sorties (dans Drive) :
  nnUNet_results/Dataset101_MMWHS/<trainer>__nnUNetPlans__3d_fullres/fold_0/ : journal training_log_*.txt,
      checkpoint_best.pth, checkpoint_final.pth, progress.png, validation/ (par nnU-Net) ;
  --mlflow/mlflow.db (base MLflow) et --mlflow/artefacts/ (journal, courbes, plans, plis).
Codes de sortie :
  0 = essai terminé et tracé ; 1 = garde-fou (rien n'est lancé ; ex. marqueur de 05a absent) ; 2 = un essai existe déjà
  (--continuer ou --ecraser) ; 3 = nnU-Net s'est arrêté en erreur (ce qui a été journalisé est tracé) ;
  5 = pas de GPU ; 6 = le journal ne confirme pas l'usage de NOTRE splits_final.json (ne pas utiliser
  les résultats).

Vocabulaire :
  - époque (epoch) : pour nnU-Net, 250 pas d'entraînement (des patchs tirés au hasard), suivis de 50 pas
    de validation rapide. Ce n'est PAS « un passage sur toutes les images » ;
  - trainer : la classe Python qui définit l'entraînement. nnUNetTrainer_5epochs est identique au
    trainer par défaut, sauf le nombre d'époques (5 au lieu de 1000) : idéal pour mesurer sans attendre ;
  - perte (loss) : l'erreur que le réseau cherche à réduire (Dice + entropie croisée chez nnU-Net ;
    négative, plus elle est basse, mieux c'est). train_loss sur les patchs d'entraînement, val_loss sur
    ceux de validation ;
  - Dice pseudo : recouvrement prédiction / vérité (0 à 1) calculé à chaque époque sur quelques patchs
    des cas de validation du pli. Indicateur de tendance seulement : le vrai Dice de validation est
    calculé à la fin, sur les volumes entiers (« Mean Validation Dice ») ;
  - point de reprise (checkpoint) : fichier .pth qui contient l'état du réseau et de l'optimiseur.
    nnU-Net écrit checkpoint_best.pth (meilleur Dice pseudo lissé), checkpoint_latest.pth toutes les
    50 époques, et checkpoint_final.pth à la fin. Reprise : relancer avec --continuer (option --c), qui
    repart de checkpoint_final, sinon checkpoint_latest, sinon checkpoint_best ;
  - torch.compile : PyTorch traduit le réseau en code GPU optimisé. Plus rapide ensuite, mais la
    compilation prend plusieurs minutes pendant la 1re époque ;
  - MLflow : outil qui range chaque essai (paramètres, mesures, fichiers) pour les comparer plus tard.
"""
import argparse, datetime, glob, importlib.util, json, os, re, shutil, subprocess, sys, tempfile, time
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

DEPOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DRIVE = "/content/drive/MyDrive/poc_ct"
EXPERIENCE_MLFLOW = "etape5_nnunet"
EPOQUES_EXTRAPOLEES = (100, 250, 1000)

# Fonctions communes avec 05a (construction et contrôle des plis). Le nom du fichier commence par un
# chiffre : on le charge par son chemin. 05a n'exécute rien à l'import (main() protégé).
_spec = importlib.util.spec_from_file_location("prep05a", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                         "05a_preparer_nnunet.py"))
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


# --- Lecture du journal de nnU-Net (testable sans GPU) ---------------------------------------
# Chaque ligne du journal commence par la date : « 2026-11-09 10:00:00.123456: message ».
RE_LIGNE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?): ?(.*)$")


def _nombres(liste):
    """« [np.float32(0.1), np.float32(nan)] » (numpy 2) ou « [0.1, nan] » (numpy 1) → [0.1, nan]."""
    texte = re.sub(r"np\.float\d*\(([^)]*)\)", r"\1", liste).strip().strip("[]")
    return [float(x) for x in texte.split(",") if x.strip()]


# Valeurs notées à chaque époque (nom de la mesure MLflow → motif de la ligne du journal).
CHAMPS_EPOQUE = {"duree_epoque_s": r"Epoch time: (\S+) s", "train_loss": r"train_loss (\S+)",
                 "val_loss": r"val_loss (\S+)", "taux_apprentissage": r"Current learning rate: (\S+)"}


def lire_journal(texte):
    """Extrait d'un journal nnU-Net : les époques (CHAMPS_EPOQUE + Dice pseudo), les messages sur les plis
    et quelques repères de temps. Renvoie un dictionnaire.
    Le journal peut être lu PENDANT que nnU-Net l'écrit : une dernière ligne sans fin de ligne est ignorée
    (elle est peut-être incomplète), et une ligne illisible est sautée au lieu d'arrêter le script.
    Repères de sauvegarde, par époque (05c) : nnU-Net écrit « Epoch time », PUIS checkpoint_latest.pth,
    PUIS « Yayy! New best », PUIS checkpoint_best.pth. Une ligne qui suit prouve que l'écriture précédente
    est finie : « apres_temps » (une ligne suit « Epoch time »), « meilleur » (ligne « Yayy » présente),
    « apres_meilleur » (une ligne suit « Yayy »)."""
    r = {"epoques": {}, "splits_utilise": None, "split_cree": False, "split_hasard_80_20": False,
         "pli_demande": None, "nb_train": None, "nb_val": None, "debut": None, "premiere_epoque": None,
         "fin_entrainement": None, "fin": None, "dice_validation_finale": None}
    ep, attente = None, None                      # attente : (époque, repère) à poser à la ligne suivante
    lignes = texte.splitlines()
    if lignes and not texte.endswith(("\n", "\r")):
        lignes = lignes[:-1]                          # dernière ligne en cours d'écriture
    for ligne in lignes:
        m = RE_LIGNE.match(ligne)
        if not m:
            continue
        try:
            date, msg = datetime.datetime.fromisoformat(m.group(1)), m.group(2).strip()
            ep, attente = _lire_ligne(r, date, msg, ep, attente)
        except (ValueError, TypeError, IndexError):  # ligne abîmée : ignorée
            continue
    return r


def _lire_ligne(r, date, msg, ep, attente):
    """Traite une ligne datée du journal (voir lire_journal). Renvoie (époque en cours, repère en attente)."""
    if attente is not None:
        attente[0][attente[1]] = True
        attente = None
    r["debut"], r["fin"] = r["debut"] or date, date
    if (e := re.fullmatch(r"Epoch (\d+)", msg)):
        ep = r["epoques"][int(e.group(1))] = {"debut": date}   # une reprise (--c) réécrit l'époque : on garde la dernière
        r["premiere_epoque"] = r["premiere_epoque"] or date
    elif msg.startswith("Using splits from existing split file:"):
        r["splits_utilise"] = msg.split(":", 1)[1].strip()
    elif msg.startswith("Creating new 5-fold cross-validation split"):
        r["split_cree"] = True
    elif "random (but seeded) 80:20 split" in msg:
        r["split_hasard_80_20"] = True
    elif (e := re.match(r"Desired fold for training: (\d+)", msg)):
        r["pli_demande"] = int(e.group(1))
    elif (e := re.match(r"This split has (\d+) training and (\d+) validation cases", msg)):
        r["nb_train"], r["nb_val"] = int(e.group(1)), int(e.group(2))
    elif msg.startswith("Training done"):
        r["fin_entrainement"] = date
    elif (e := re.match(r"Mean Validation Dice:\s*(\S+)", msg)):
        r["dice_validation_finale"] = float(e.group(1))
    elif ep is not None and msg.startswith("Pseudo dice"):
        ep["pseudo_dice"] = _nombres(msg[len("Pseudo dice"):])
    elif ep is not None and msg.startswith("Yayy! New best"):
        ep["meilleur"] = True
        attente = (ep, "apres_meilleur")
    elif ep is not None:
        for nom, motif in CHAMPS_EPOQUE.items():
            if (e := re.match(motif, msg)):
                ep[nom] = float(e.group(1))
                if nom == "duree_epoque_s":
                    attente = (ep, "apres_temps")
    return ep, attente


def journaux_chronologiques(dossier_pli):
    """Journaux training_log_*.txt du pli, du plus ancien au plus récent. nnU-Net écrit le mois et le jour
    sans zéro (training_log_2026_11_9_...) : un tri alphabétique placerait le 10 avant le 9. On trie donc
    sur les nombres du nom."""
    cle = lambda c: [int(x) for x in re.findall(r"\d+", os.path.basename(c))]
    return sorted(glob.glob(os.path.join(dossier_pli, "training_log_*.txt")), key=cle)


def series_par_epoque(epoques):
    """Ajoute à chaque époque son Dice pseudo moyen (sans les classes absentes : NaN) et renvoie les séries
    MLflow : {nom: [(numéro d'époque, valeur), ...]} (CHAMPS_EPOQUE, Dice pseudo moyen et par étiquette 1..7)."""
    for e in epoques.values():
        valides = [d for d in e.get("pseudo_dice", []) if d == d]
        e["pseudo_dice_moyen"] = sum(valides) / len(valides) if valides else float("nan")
    series = {nom: [(k, e[nom]) for k, e in epoques.items() if nom in e]
              for nom in list(CHAMPS_EPOQUE) + ["pseudo_dice_moyen"]}
    for k, e in epoques.items():
        for i, d in enumerate(e.get("pseudo_dice", []), start=1):
            series.setdefault(f"pseudo_dice_etiquette_{i}", []).append((k, d))
    return series


def afficher_epoques(epoques, dernieres=None):
    """Tableau des époques (les « dernieres » seulement, si précisé). À appeler après series_par_epoque."""
    print("\nÉpoque | durée (s) | train_loss | val_loss | Dice pseudo moyen")
    for k in sorted(epoques)[-dernieres if dernieres else 0:]:
        e = {c: epoques[k].get(c, float("nan")) for c in ("duree_epoque_s", "train_loss", "val_loss", "pseudo_dice_moyen")}
        print(f"{k:>6} | {e['duree_epoque_s']:>9.1f} | {e['train_loss']:>10.4f} | {e['val_loss']:>8.4f} | "
              f"{e['pseudo_dice_moyen']:.4f}")


def bilan_durees(epoques):
    """Durées des époques terminées, moyenne sans la 1re (compilation, mise en route), extrapolation."""
    durees = [epoques[k]["duree_epoque_s"] for k in sorted(epoques) if "duree_epoque_s" in epoques[k]]
    base = durees[1:] if len(durees) >= 2 else durees
    moyenne = sum(base) / len(base) if base else None
    extrapolation = {n: n * moyenne / 3600 for n in EPOQUES_EXTRAPOLEES} if moyenne else {}
    return durees, moyenne, extrapolation


def verifier_usage_splits(journaux, chemins_acceptes, pli, plis):
    """Problèmes constatés dans les journaux de CETTE exécution (vide = nnU-Net a bien utilisé nos plis).
    chemins_acceptes : emplacements possibles de NOTRE splits_final.json (Drive, ou copie locale vérifiée)."""
    pb, acceptes = [], {os.path.normpath(c) for c in chemins_acceptes}
    for nom, j in journaux.items():
        if j["split_cree"]:
            pb.append(f"{nom} : nnU-Net a CRÉÉ un nouveau découpage au hasard")
        if j["split_hasard_80_20"]:
            pb.append(f"{nom} : nnU-Net a tiré un découpage 80:20 au hasard (pli absent du fichier)")
        if j["splits_utilise"] is None:
            pb.append(f"{nom} : aucune ligne « Using splits from existing split file »")
        elif os.path.normpath(j["splits_utilise"]) not in acceptes:
            pb.append(f"{nom} : fichier de plis utilisé {j['splits_utilise']} ≠ attendu {sorted(acceptes)}")
        if j["pli_demande"] != pli:
            pb.append(f"{nom} : pli {j['pli_demande']} au lieu de {pli}")
        if (j["nb_train"], j["nb_val"]) != (len(plis[pli]["train"]), len(plis[pli]["val"])):
            pb.append(f"{nom} : {j['nb_train']} cas d'entraînement / {j['nb_val']} de validation, attendu "
                      f"{len(plis[pli]['train'])} / {len(plis[pli]['val'])}")
    if not journaux:
        pb.append("aucun journal d'entraînement produit par cette exécution")
    return pb


# --- MLflow : base SQLite dans Drive, écrite d'un seul coup ----------------------------------
def tracer_mlflow(dossier, nom_run, params, tags, series, metriques, artefacts, id_run=None, remplacer_depuis=None):
    """Enregistre un essai. Une base SQLite ne supporte pas bien les écritures directes dans Drive
    (verrous de fichiers) : on travaille sur une copie locale, puis on la recopie dans Drive en une fois.
    Les artefacts (fichiers) sont écrits directement dans --mlflow/artefacts. Renvoie l'identifiant du run.
    id_run (05c) : complète ce run s'il existe dans la base, au lieu d'en créer un nouveau. Les paramètres
    déjà notés ne sont pas réécrits (MLflow les interdit de modification), et une mesure par époque n'est
    ajoutée que pour les époques pas encore notées : relancer ne crée pas de doublons.
    remplacer_depuis (05c, cas de secours) : à partir de cette époque, une valeur différente de celles déjà
    notées est ajoutée quand même (MLflow ne sait pas effacer une valeur : la plus récente fait foi)."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")      # message d'aide de MLflow, inutile ici
    import mlflow
    os.makedirs(dossier, exist_ok=True)
    base_drive = os.path.join(dossier, "mlflow.db")
    base_locale = os.path.join(tempfile.mkdtemp(), "mlflow.db")
    if os.path.exists(base_drive):
        shutil.copy2(base_drive, base_locale)
    mlflow.set_tracking_uri("sqlite:///" + base_locale.replace("\\", "/"))
    client = mlflow.tracking.MlflowClient()
    exp = mlflow.get_experiment_by_name(EXPERIENCE_MLFLOW)
    id_exp = exp.experiment_id if exp else mlflow.create_experiment(
        EXPERIENCE_MLFLOW, artifact_location=Path(dossier, "artefacts").resolve().as_uri())
    if id_run is not None:
        try:
            client.get_run(id_run)
        except Exception:                                # run absent de la base (base perdue ou remplacée)
            print(f"ATTENTION : run MLflow {id_run} introuvable dans {base_drive} : un nouveau run est créé.")
            id_run = None
    options = {"run_id": id_run} if id_run else {"experiment_id": id_exp, "run_name": nom_run}
    with mlflow.start_run(**options) as run:
        deja = client.get_run(run.info.run_id).data.params
        mlflow.log_params({k: str(v)[:500] for k, v in params.items() if k not in deja})
        mlflow.set_tags(tags)
        for nom, valeurs in series.items():              # une valeur par époque (step = numéro d'époque)
            historique = client.get_metric_history(run.info.run_id, nom) if id_run else []
            notees = {m.step for m in historique}
            couples = {(m.step, m.value) for m in historique}
            for etape, v in valeurs:
                if v != v:                               # ignore NaN (classe absente des patchs de validation)
                    continue
                if etape not in notees or (remplacer_depuis is not None and etape >= remplacer_depuis
                                           and (etape, v) not in couples):
                    mlflow.log_metric(nom, v, step=etape)
        mlflow.log_metrics({k: v for k, v in metriques.items() if v is not None and v == v})
        for f in artefacts:
            if os.path.isfile(f):
                mlflow.log_artifact(f)
    tmp = base_drive + ".tmp"
    shutil.copy2(base_locale, tmp)
    os.replace(tmp, base_drive)
    return run.info.run_id


# --- Utilitaires ------------------------------------------------------------------------------
def version_paquet(nom):
    try:
        return version(nom)
    except PackageNotFoundError:
        return "inconnu"


def copier_si_different(source, cible):
    """Copie un dossier fichier par fichier, en sautant ceux déjà identiques (même taille, même date)."""
    copies, octets = 0, 0
    for racine, _, fichiers in os.walk(source):
        for nom in fichiers:
            s = os.path.join(racine, nom)
            c = os.path.join(cible, os.path.relpath(s, source))
            st = os.stat(s)
            if os.path.exists(c) and os.stat(c).st_size == st.st_size and int(os.stat(c).st_mtime) == int(st.st_mtime):
                continue
            os.makedirs(os.path.dirname(c), exist_ok=True)
            shutil.copy2(s, c)                           # copy2 garde la date : la comparaison suivante marche
            copies, octets = copies + 1, octets + st.st_size
    return copies, octets


def taille_dossier(d):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)


# --- Prérequis communs avec 05c (arrêt en code 1 si un contrôle échoue) -----------------------
def verifier_prerequis_05a(raw, pre, nom_ds, split, pli):
    """Marqueur de vérification de 05a présent ; splits_final.json = celui de 05a (empreinte de la fiche,
    et plis recalculés depuis le découpage figé, la graine et les strates relues dans imagesTr).
    Renvoie (ds_pre, chemin_splits, chemin_trace, trace, plis)."""
    ds_pre = os.path.join(pre, nom_ds)
    cas, cas_test, groupes, _ = prep.lire_decoupage(split)
    message = prep.lire_marqueur(ds_pre, cas)
    if message:
        sys.exit(f"ARRÊT : {message} Lancer d'abord le script 05a jusqu'à « PRÊT ».")
    chemin_splits, chemin_trace = os.path.join(ds_pre, prep.NOM_SPLITS), os.path.join(ds_pre, prep.NOM_TRACE)
    if not (os.path.exists(chemin_splits) and os.path.exists(chemin_trace)):
        sys.exit(f"ARRÊT : {chemin_splits} ou sa fiche {prep.NOM_TRACE} manque. Lancer d'abord le script 05a "
                 "(sinon nnU-Net tirerait ses plis au hasard, cas par cas).")
    with open(chemin_trace, encoding="utf-8") as f:
        trace = json.load(f)
    with open(chemin_splits, encoding="utf-8") as f:
        plis = json.load(f)
    # Strates relues dans les en-têtes de imagesTr (pas recopiées de la fiche), puis comparées à la fiche.
    strates, problemes = prep.lire_strates(os.path.join(raw, nom_ds), cas)
    if strates != trace.get("strates_mm"):
        problemes.append(f"strates relues {strates} ≠ strates de la fiche 05a {trace.get('strates_mm')}")
    if problemes:
        sys.exit("ARRÊT : épaisseurs de coupe incohérentes :\n  - " + "\n  - ".join(problemes))
    attendus = prep.construire_plis(cas, groupes, strates, prep.NB_PLIS, trace["graine"])
    problemes = prep.controler_plis(plis, cas, cas_test, groupes, strates)
    if prep.sha256_fichier(chemin_splits) != trace["sha256_splits_final"]:
        problemes.append("empreinte de splits_final.json ≠ celle notée par 05a (fichier modifié depuis)")
    if [{"train": sorted(p["train"]), "val": sorted(p["val"])} for p in plis] != attendus:
        problemes.append("splits_final.json ≠ plis recalculés depuis le découpage figé, les strates et la graine de 05a")
    if problemes:
        sys.exit("ARRÊT : splits_final.json n'est pas celui de 05a :\n  - " + "\n  - ".join(problemes))
    print(f"splits_final.json vérifié (écrit par 05a, graine {trace['graine']}). "
          f"Pli {pli} : validation = {plis[pli]['val']}")
    return ds_pre, chemin_splits, chemin_trace, trace, plis


def copie_locale(ds_pre, local, nom_ds, trace):
    """Copie les données prétraitées de Drive vers le disque local (seulement les fichiers changés) et
    vérifie la copie de splits_final.json. Renvoie le chemin de cette copie (celui que lira nnU-Net)."""
    locale = os.path.abspath(local)
    besoin, libre = taille_dossier(ds_pre), shutil.disk_usage(os.path.dirname(locale) or "/").free
    if besoin * 1.1 > libre:
        sys.exit(f"ARRÊT : pas assez de place sur le disque local ({besoin / 1e9:.1f} Go à copier, "
                 f"{libre / 1e9:.1f} Go libres). Relancer avec --lire-depuis-drive.")
    t0 = time.time()
    n, octets = copier_si_different(ds_pre, os.path.join(locale, nom_ds))
    print(f"Copie locale : {n} fichiers ({octets / 1e9:.2f} Go) copiés en {time.time() - t0:.0f} s "
          f"vers {locale} (total {besoin / 1e9:.2f} Go)")
    chemin = os.path.join(locale, nom_ds, prep.NOM_SPLITS)
    if prep.sha256_fichier(chemin) != trace["sha256_splits_final"]:
        sys.exit("ARRÊT : la copie locale de splits_final.json diffère de celle de Drive.")
    return chemin


def lire_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=f"{DRIVE}/nnunet_raw")
    p.add_argument("--preprocessed", default=f"{DRIVE}/nnunet_preprocessed")
    p.add_argument("--results", default=f"{DRIVE}/nnunet_results")
    p.add_argument("--mlflow", default=f"{DRIVE}/mlflow")
    p.add_argument("--split", default=os.path.join(DEPOT, "splits", "mmwhs_split.json"))
    p.add_argument("--id-dataset", type=int, default=101)
    p.add_argument("--pli", type=int, default=0, choices=range(prep.NB_PLIS))
    p.add_argument("--trainer", default="nnUNetTrainer_5epochs")
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--processus", type=int, default=2, choices=[1, 2, 3, 4])
    p.add_argument("--processus-augmentation", type=int, default=None)
    p.add_argument("--sans-compile", action="store_true")
    p.add_argument("--lire-depuis-drive", action="store_true")
    p.add_argument("--local", default="/content/nnunet_preprocessed_local")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--continuer", action="store_true")
    g.add_argument("--ecraser", action="store_true")
    g.add_argument("--tracer-seulement", action="store_true")
    return p.parse_args()


# --- Programme principal ----------------------------------------------------------------------
def main():
    a = lire_arguments()
    raw, pre, res, dossier_mlflow = (os.path.abspath(x) for x in (a.raw, a.preprocessed, a.results, a.mlflow))
    nom_ds = f"Dataset{a.id_dataset:03d}_MMWHS"
    for nom, d in (("--raw", raw), ("--preprocessed", pre), ("--results", res), ("--mlflow", dossier_mlflow),
                   ("--local", a.local)):
        if prep.dans(d, DEPOT):
            sys.exit(f"ARRÊT : {nom} est dans le dépôt. Les données et résultats restent hors du dépôt.")

    # --- 1. Vérification du jeu faite par 05a, puis splits_final.json écrit par 05a -----------------
    ds_pre, chemin_splits, chemin_trace, trace, plis = verifier_prerequis_05a(raw, pre, nom_ds, a.split, a.pli)

    # --- 2. Dossier de l'essai : déjà présent ? ---------------------------------------------------
    dossier_pli = os.path.join(res, nom_ds, f"{a.trainer}__nnUNetPlans__{a.configuration}", f"fold_{a.pli}")
    avant = set(glob.glob(os.path.join(dossier_pli, "training_log_*.txt")))
    if a.tracer_seulement:
        if not avant:
            sys.exit(f"ARRÊT : aucun journal dans {dossier_pli}.")
    elif avant and not (a.continuer or a.ecraser):
        print(f"ARRÊT : un essai existe déjà dans {dossier_pli}. Rien n'a été modifié.")
        print("  --continuer : reprendre depuis le dernier point de reprise ; --ecraser : tout recommencer.")
        sys.exit(2)

    env = prep.environnement_nnunet(raw, pre, res, a.processus)
    if a.sans_compile:
        env["nnUNet_compile"] = "f"
    if a.processus_augmentation:
        env["nnUNet_n_proc_DA"] = str(a.processus_augmentation)
    gpu, code, duree_totale, chemin_splits_vu = "non utilisé", 0, None, chemin_splits

    if not a.tracer_seulement:
        # --- 3. GPU et trainer ----------------------------------------------------------------------
        import torch
        if not torch.cuda.is_available():
            print("ARRÊT : pas de GPU visible (Exécution → Modifier le type d'exécution → GPU T4).")
            sys.exit(5)
        gpu = torch.cuda.get_device_name(0)
        import nnunetv2
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        classe = recursive_find_python_class(os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                             a.trainer, "nnunetv2.training.nnUNetTrainer")
        if classe is None:
            sys.exit(f"ARRÊT : le trainer {a.trainer} n'existe pas dans nnunetv2 {version_paquet('nnunetv2')}.")
        print(f"GPU : {gpu} ; nnunetv2 {version_paquet('nnunetv2')} ; trainer {a.trainer} trouvé ; "
              f"torch.compile : {'non' if a.sans_compile else 'oui (défaut nnU-Net)'}")
        exe = shutil.which("nnUNetv2_train")
        if exe is None:
            sys.exit("ARRÊT : nnUNetv2_train introuvable. Installer nnU-Net (notebook, section 12).")

        # --- 4. Copie locale des données prétraitées ------------------------------------------------
        if not a.lire_depuis_drive:
            chemin_splits_vu = copie_locale(ds_pre, a.local, nom_ds, trace)
            env["nnUNet_preprocessed"] = os.path.abspath(a.local)

        # --- 5. Entraînement --------------------------------------------------------------------------
        if a.ecraser and os.path.isdir(dossier_pli):
            shutil.rmtree(dossier_pli)                   # --ecraser demandé explicitement
            avant = set()
        commande = [exe, str(a.id_dataset), a.configuration, str(a.pli), "-tr", a.trainer]
        if a.continuer:
            commande.append("--c")
        print("Commande :", " ".join(commande), flush=True)
        t0 = time.time()
        code = subprocess.run(commande, env=env).returncode   # sortie de nnU-Net affichée telle quelle
        duree_totale = time.time() - t0
        print(f"\nnnU-Net terminé (code {code}) en {duree_totale / 60:.1f} min")

    # --- 6. Lecture des journaux --------------------------------------------------------------------
    tous = journaux_chronologiques(dossier_pli)
    nouveaux = tous if a.tracer_seulement else [j for j in tous if j not in avant]
    lus = {}
    for chemin in tous:
        with open(chemin, encoding="utf-8", errors="replace") as f:
            lus[chemin] = lire_journal(f.read())
    epoques = {}
    for chemin in tous:                                  # ordre chronologique : une reprise remplace l'époque
        epoques.update(lus[chemin]["epoques"])
    # Pendant l'essai, nnU-Net a lu soit Drive, soit la copie locale (vérifiée identique à l'étape 4).
    # En retraçage (--tracer-seulement), on ne sait plus laquelle : les deux emplacements sont acceptés.
    acceptes = {chemin_splits_vu} if not a.tracer_seulement else \
        {chemin_splits, os.path.join(os.path.abspath(a.local), nom_ds, prep.NOM_SPLITS)}
    usage = verifier_usage_splits({os.path.basename(j): lus[j] for j in nouveaux}, acceptes, a.pli, plis)
    val_vus = {os.path.basename(x)[:-len(".nii.gz")] for x in glob.glob(os.path.join(dossier_pli, "validation", "*.nii.gz"))}
    if val_vus and val_vus != set(plis[a.pli]["val"]):
        usage.append(f"cas prédits en validation finale {sorted(val_vus)} ≠ validation du pli {plis[a.pli]['val']}")

    # --- 7. Affichage des durées --------------------------------------------------------------------
    durees, moyenne, extrapolation = bilan_durees(epoques)
    series = series_par_epoque(epoques)                # calcule aussi le Dice pseudo moyen de chaque époque
    afficher_epoques(epoques)
    j = lus[nouveaux[-1]] if nouveaux else {}
    demarrage = (j["premiere_epoque"] - j["debut"]).total_seconds() if j.get("premiere_epoque") else None
    validation_finale = ((j["fin"] - j["fin_entrainement"]).total_seconds()
                         if j.get("fin_entrainement") and j.get("dice_validation_finale") is not None else None)
    # « Epoch time » de nnU-Net s'arrête AVANT l'écriture de checkpoint_best.pth (dans Drive) et de la courbe :
    # on mesure ce temps « hors époque » = (époque 0 → « Training done ») − somme des « Epoch time » du journal.
    hors_epoque = None
    durees_j = [e["duree_epoque_s"] for e in j.get("epoques", {}).values() if "duree_epoque_s" in e]
    if j.get("premiere_epoque") and j.get("fin_entrainement") and durees_j:
        hors_epoque = (j["fin_entrainement"] - j["premiere_epoque"]).total_seconds() - sum(durees_j)
    if moyenne:
        print(f"\nDurée moyenne d'une époque (sans la 1re) : {moyenne:.1f} s ; 1re époque : {durees[0]:.1f} s")
        if hors_epoque is not None:
            print(f"Temps hors « Epoch time » (écriture des points de reprise dans Drive, courbes) : {hors_epoque:.0f} s "
                  f"sur {len(durees_j)} époques, soit ≈ {hors_epoque / len(durees_j):.0f} s par époque")
        print("Extrapolation INDICATIVE (somme des « Epoch time » seulement : sans le démarrage, la validation finale,")
        print("ni le temps hors époque ci-dessus, qui diminue quand checkpoint_best.pth cesse de progresser) :")
        for n, h in extrapolation.items():
            print(f"  {n:>5} époques : ≈ {h:.1f} h par pli, ≈ {5 * h:.1f} h pour les 5 plis")
        print("  Rappel : checkpoint_latest.pth n'est écrit que toutes les 50 époques (une coupure de session "
              "peut faire perdre jusqu'à 49 époques).")
    if duree_totale is not None and durees_j:
        print(f"Durée totale de la commande nnU-Net : {duree_totale:.0f} s, dont {sum(durees_j):.0f} s de « Epoch time » "
              f"(écart {duree_totale - sum(durees_j):.0f} s : démarrage, temps hors époque, validation finale)")
    if demarrage is not None:
        print(f"Démarrage (chargement, avant l'époque 0) : {demarrage:.0f} s")
    if validation_finale is not None:
        print(f"Validation finale ({len(plis[a.pli]['val'])} volumes entiers) : {validation_finale:.0f} s ; "
              f"Dice moyen de validation (pli {a.pli}, PAS le test) : {j['dice_validation_finale']:.4f}")

    # --- 8. MLflow -----------------------------------------------------------------------------------
    plans = os.path.join(ds_pre, "nnUNetPlans.json")
    artefacts = tous + [os.path.join(dossier_pli, f) for f in ("progress.png", "debug.json")] + \
        [os.path.join(dossier_pli, "validation", "summary.json"), chemin_splits, chemin_trace, plans,
         os.path.join(ds_pre, prep.NOM_MARQUEUR)]
    params = {"pli": a.pli, "trainer": a.trainer, "configuration": a.configuration,
              "nnunetv2": version_paquet("nnunetv2"), "torch": version_paquet("torch"), "gpu": gpu,
              "torch_compile": not a.sans_compile, "nnUNet_def_n_proc": a.processus,
              "nnUNet_n_proc_DA": a.processus_augmentation or "défaut nnU-Net",
              "lecture": "Drive" if a.lire_depuis_drive else "copie locale",
              "reprise": a.continuer, "graine_plis": trace["graine"], "val_pli": ",".join(plis[a.pli]["val"]),
              "sha256_splits_final": trace["sha256_splits_final"], "commande": " ".join(sys.argv)}
    metriques = {"duree_epoque_moyenne_sans_1re_s": moyenne, "duree_1re_epoque_s": durees[0] if durees else None,
                 "nb_epoques_terminees": len(durees), "duree_totale_s": duree_totale,
                 "demarrage_s": demarrage, "validation_finale_s": validation_finale, "hors_epoque_s": hors_epoque,
                 "dice_validation_finale": j.get("dice_validation_finale") if j else None}
    metriques.update({f"extrapolation_{n}_epoques_h_par_pli": h for n, h in extrapolation.items()})
    statut = "echec_nnunet" if code else ("plis_non_confirmes" if usage else "ok")
    tags = {"etape": "5", "type": "essai_a_blanc", "statut": statut, "mlflow.source.git.commit": prep.commit_git()}
    try:
        id_run = tracer_mlflow(dossier_mlflow, f"essai_{a.trainer}_{a.configuration}_pli{a.pli}",
                               params, tags, series, metriques, artefacts)
        print(f"\nEssai tracé dans MLflow : {dossier_mlflow} (run {id_run}, statut {statut})")
    except Exception as err:
        print(f"\nATTENTION : traçage MLflow impossible ({err}). Relancer avec --tracer-seulement.")

    # --- 9. Verdict ------------------------------------------------------------------------------------
    if code:
        print(f"Résultat : nnU-Net s'est arrêté en erreur (code {code}). Envoyer la fin de la sortie dans la conversation.")
        sys.exit(3)
    if usage:
        print("Résultat : NON CONFIRMÉ, nnU-Net n'a peut-être pas utilisé nos plis. Ne pas utiliser ces résultats :")
        for u in usage:
            print("  -", u)
        sys.exit(6)
    print(f"Contrôle des plis dans le journal : nnU-Net a utilisé {j['splits_utilise']} "
          f"(pli {a.pli} : {len(plis[a.pli]['train'])} en entraînement, {len(plis[a.pli]['val'])} en validation).")
    print("Résultat : ESSAI TERMINÉ. Le lot de test n'a été ni lu ni utilisé.")


if __name__ == "__main__":
    main()
