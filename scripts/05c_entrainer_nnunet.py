"""
Étape 5c — Entraînement nnU-Net v2 d'un pli : 3d_fullres, 250 époques, reprise automatique (décision D25).

Ce que fait le script :
  1. mêmes garde-fous que 05b (fonctions réutilisées) : dossiers hors du dépôt ; marqueur
     verification_05a.json ; splits_final.json identique aux plis recalculés depuis le découpage figé,
     la graine et les strates relues dans imagesTr. Aucune lecture de imagesTs ni de labelsTs ;
  2. regarde où en est le pli dans nnUNet_results (Drive) :
       - aucun point de reprise : nouvel entraînement ;
       - checkpoint_latest.pth ou checkpoint_best.pth : REPRISE automatique (option --c de nnU-Net),
         à l'époque notée dans le point de reprise ;
       - checkpoint_final.pth : entraînement TERMINÉ. Rien n'est relancé (sauf --terminer-validation
         si la validation finale a été interrompue) ; le suivi MLflow est seulement mis à jour ;
  3. vérifie le GPU (avertit si ce n'est pas un L4, sans bloquer) et la variante d'entraînement
     nnUNetTrainer_250epochs_reprise10 (scripts/trainers_nnunet/, trouvée par nnUNet_extTrainer) ;
  4. copie les données prétraitées sur le disque local (lectures rapides) ; résultats et points de
     reprise restent écrits dans Drive ;
  5. pose un verrou (fold_X/entrainement_en_cours.verrou, rafraîchi toutes les 5 min, retiré à la fin) pour
     qu'une 2e session ne puisse pas entraîner le même pli en même temps ; lance nnUNetv2_train, et toutes
     les --tracer-toutes-les minutes, relit le journal et met à jour MLflow dans Drive. Seules les époques
     « consolidées » (couvertes par un point de reprise, donc jamais refaites) sont notées dans MLflow.
     Une erreur du suivi n'arrête jamais l'entraînement ; si le script s'arrête, il arrête aussi nnU-Net ;
  6. à la fin : contrôle que nnU-Net a utilisé NOS plis, durées, estimation du temps restant et des
     unités de calcul, Dice de validation finale (3 cas du pli, PAS le test), traçage MLflow.
  Le script ne lance AUCUNE prédiction sur le lot de test (étape 6).

Entrées :
  --pli           pli à entraîner, OBLIGATOIRE (0 à 4)
  --raw / --preprocessed / --results / --mlflow / --split / --local   comme 05b
  --processus, --processus-augmentation, --sans-compile, --lire-depuis-drive   comme 05b
  --unites-par-heure  coût du GPU en unités de calcul Colab par heure (défaut 1,94 : L4, mesuré en 05b)
  --tracer-toutes-les minutes entre deux mises à jour de MLflow pendant l'entraînement (défaut 30)
  --terminer-validation  relance nnU-Net (--c) alors que checkpoint_final.pth existe : sert seulement à
                  refaire une validation finale interrompue
  --tracer-seulement     ne lance rien : relit les journaux et met à jour MLflow
  --ignorer-verrou       lance même si le verrou d'une autre session a moins de 15 min (seulement si l'on est
                  SÛR qu'aucune autre session n'entraîne ce pli)
Sorties (dans Drive) :
  nnUNet_results/Dataset101_MMWHS/nnUNetTrainer_250epochs_reprise10__nnUNetPlans__3d_fullres/fold_X/ :
      training_log_*.txt (un par session), checkpoint_latest.pth (toutes les 10 époques),
      checkpoint_best.pth, checkpoint_final.pth, progress.png, validation/ (par nnU-Net) ;
      mlflow_run_id.txt (identifiant du run MLflow de ce pli, retrouvé à chaque reprise) ;
      entrainement_en_cours.verrou (pendant l'entraînement seulement) ;
  --mlflow/mlflow.db et --mlflow/artefacts/ : UN run par pli, complété à chaque session.
Codes de sortie :
  0 = session terminée normalement (entraînement fini, ou déjà fini) ; 1 = garde-fou (dont : verrou d'une
      autre session) ;
  2 = checkpoint_final.pth présent mais validation finale incomplète (relancer avec --terminer-validation) ;
  3 = nnU-Net s'est arrêté en erreur ou a été interrompu (relancer la même cellule pour reprendre) ;
  5 = pas de GPU ; 6 = le journal ne confirme pas l'usage de NOS plis (ne pas utiliser les résultats).

Vocabulaire :
  - point de reprise (checkpoint) : fichier .pth qui contient l'état complet de l'entraînement (poids
    du réseau, optimiseur, numéro d'époque, historique). nnU-Net repart de là au lieu de recommencer ;
  - reprise (--c) : nnU-Net repart du plus récent de checkpoint_final, checkpoint_latest, checkpoint_best
    (dans cet ordre). checkpoint_best (meilleur Dice pseudo) peut être plus récent que checkpoint_latest :
    nnU-Net repart quand même de latest, c'est le comportement normal ;
  - session : une connexion à une machine Colab. Elle peut être coupée (durée maximale, inactivité,
    unités épuisées). Relancer la même cellule dans une nouvelle session reprend l'entraînement ;
  - unité de calcul : la « monnaie » de Colab ; chaque heure de GPU en consomme un nombre fixe.
"""
import argparse, datetime, glob, importlib.util, json, os, shutil, subprocess, sys, time

ICI = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("essai05b", os.path.join(ICI, "05b_essai_nnunet.py"))
essai = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(essai)                   # 05b n'exécute rien à l'import ; il apporte aussi 05a
prep = essai.prep

TRAINER = "nnUNetTrainer_250epochs_reprise10"
DOSSIER_TRAINERS = os.path.join(ICI, "trainers_nnunet")
CONFIGURATION = "3d_fullres"
NB_EPOQUES = 250                                  # doit rester égal à num_epochs du trainer
SAUVEGARDE_TOUTES_LES = 10                        # doit rester égal à save_every du trainer
FICHIER_RUN = "mlflow_run_id.txt"
NOM_VERROU = "entrainement_en_cours.verrou"       # empêche deux sessions d'entraîner le même pli
VERROU_MINUTES = 15                               # un verrou plus vieux est le reste d'une session coupée
PAS_VERROU_S = 5 * 60                             # le verrou est rafraîchi toutes les 5 min
SESSION = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")   # fixé au lancement : un tag par session
ORDRE_REPRISE = ("checkpoint_final.pth", "checkpoint_latest.pth", "checkpoint_best.pth")   # ordre de nnU-Net --c


# --- Fonctions testables sans GPU ------------------------------------------------------------
def point_de_reprise(dossier_pli):
    """Le fichier depuis lequel nnU-Net --c repartirait, ou None (nouvel entraînement)."""
    return next((os.path.join(dossier_pli, n) for n in ORDRE_REPRISE if os.path.isfile(os.path.join(dossier_pli, n))), None)


def etat_du_pli(dossier_pli):
    """« neuf », « reprise », « termine » ou « validation_incomplete »."""
    ckpt = point_de_reprise(dossier_pli)
    if ckpt is None:
        return "neuf"
    if os.path.basename(ckpt) != "checkpoint_final.pth":
        return "reprise"
    return "termine" if os.path.isfile(os.path.join(dossier_pli, "validation", "summary.json")) else "validation_incomplete"


def epoque_de_reprise(chemin):
    """Numéro de la prochaine époque, lu dans le point de reprise (clé current_epoch de nnU-Net), ou None.
    Le fichier (quelques centaines de Mo) est lu sur le processeur, puis libéré."""
    try:
        import torch
        return int(torch.load(chemin, map_location="cpu", weights_only=False)["current_epoch"])
    except Exception as err:
        print(f"  (époque non lisible dans {os.path.basename(chemin)} : {err})")
        return None


def lire_journaux(dossier_pli):
    """Lit tous les journaux du pli (un par session). Renvoie ({chemin: journal lu}, époques fusionnées).
    Fusion : une reprise repart du point de reprise, pas de la dernière époque faite. Exemple : session 1
    coupée après l'époque 16, reprise depuis checkpoint_latest (époque 10) : les époques 10 à 16 de la
    session 1 sont ABANDONNÉES. Pour chaque journal, on retire donc des journaux précédents toutes les
    époques ≥ à sa première époque (ici : 0-9 de la session 1, puis 10... de la session 2)."""
    lus, epoques = {}, {}
    for chemin in essai.journaux_chronologiques(dossier_pli):   # du plus ancien au plus récent
        with open(chemin, encoding="utf-8", errors="replace") as f:
            j = lus[chemin] = essai.lire_journal(f.read())
        if j["epoques"]:
            debut = min(j["epoques"])
            epoques = {k: e for k, e in epoques.items() if k < debut}
        epoques.update(j["epoques"])
    return lus, epoques


def borne_consolidee(dossier_pli, epoques):
    """Nombre d'époques consolidées : les époques 0..borne-1 ne peuvent plus être refaites, car nnU-Net --c
    repartirait au plus tôt de la borne. Seules ces époques sont envoyées à MLflow (MLflow ne sait pas
    effacer une valeur : une époque notée puis refaite garderait ses anciennes valeurs).
    nnU-Net reprend depuis checkpoint_final, sinon checkpoint_latest, sinon checkpoint_best :
      - final présent : tout est consolidé ;
      - latest présent : nnU-Net repart de lui, même si best est plus récent. latest est écrit à la fin des
        époques 9, 19, 29... (current_epoch = 10, 20, 30...), APRÈS la ligne « Epoch time » : l'écriture
        n'est sûre que si une ligne suit (repère apres_temps). Borne = 10 × n pour la dernière époque
        10 × n - 1 confirmée ;
      - seulement best (avant l'époque 10) : borne = dernière époque « Yayy! New best » confirmée + 1.
    Les époques sont celles de lire_journaux (branche en cours), donc jamais une époque abandonnée."""
    present = lambda n: os.path.isfile(os.path.join(dossier_pli, n))
    if present("checkpoint_final.pth"):
        return max(epoques) + 1 if epoques else 0
    if present("checkpoint_latest.pth"):
        return max((k + 1 for k, e in epoques.items()
                    if (k + 1) % SAUVEGARDE_TOUTES_LES == 0 and e.get("apres_temps")), default=0)
    if present("checkpoint_best.pth"):
        return max((k + 1 for k, e in epoques.items() if e.get("meilleur") and e.get("apres_meilleur")), default=0)
    return 0


def verifier_verrou(chemin, ignorer):
    """Refuse de lancer si une autre session entraîne ce pli (verrou rafraîchi il y a moins de
    VERROU_MINUTES). Un verrou plus ancien est le reste d'une session coupée : il est remplacé."""
    if not os.path.isfile(chemin):
        return
    try:
        with open(chemin, encoding="utf-8") as f:
            infos = json.load(f)
        age = (time.time() - float(infos["horodatage"])) / 60
    except Exception:                             # verrou illisible : on se fie à sa date de modification
        infos, age = {}, (time.time() - os.path.getmtime(chemin)) / 60
    if age < VERROU_MINUTES and not ignorer:
        print(f"ARRÊT : une autre session semble entraîner ce pli (verrou {os.path.basename(chemin)} rafraîchi il y a "
              f"{age:.0f} min, session {infos.get('session', '?')}).")
        print(f"  Si c'est le reste d'une session coupée, attendre {VERROU_MINUTES} min ; si tu es SÛR qu'aucune "
              "autre session ne tourne, relancer avec --ignorer-verrou.")
        sys.exit(1)
    if age < VERROU_MINUTES:
        print(f"ATTENTION : verrou récent ({age:.0f} min) ignoré à ta demande (--ignorer-verrou) ; il est remplacé.")
    else:
        print(f"Verrou ancien ({age:.0f} min) : reste d'une session coupée, il est remplacé.")


def poser_verrou(chemin):
    os.makedirs(os.path.dirname(chemin), exist_ok=True)     # pli neuf : nnU-Net n'a pas encore créé le dossier
    with open(chemin + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"session": SESSION, "horodatage": time.time(),
                   "date": datetime.datetime.now().isoformat(timespec="seconds")}, f)
    os.replace(chemin + ".tmp", chemin)


def retirer_verrou(chemin):
    """Supprime le verrou s'il est bien celui de cette session."""
    try:
        with open(chemin, encoding="utf-8") as f:
            session = json.load(f).get("session")
        if session == SESSION:                    # fichier refermé avant suppression
            os.remove(chemin)
    except Exception as err:
        print(f"(verrou non retiré : {err} ; il sera ignoré dans {VERROU_MINUTES} min)")


def durees_par_session(lus):
    """Durées des époques, sans la 1re de chaque session (compilation, mise en route). Renvoie
    (« Epoch time » de nnU-Net, durées réelles). Durée réelle = écart entre les débuts de deux époques
    successives d'un même journal : elle inclut l'écriture des points de reprise dans Drive."""
    epoch_time, reelles = [], []
    for j in lus.values():
        cles = sorted(j["epoques"])
        epoch_time += [j["epoques"][k]["duree_epoque_s"] for k in cles[1:] if "duree_epoque_s" in j["epoques"][k]]
        debuts = [j["epoques"][k]["debut"] for k in cles]
        reelles += [(b - a).total_seconds() for a, b in zip(debuts, debuts[1:])][1:]
    moyenne = lambda x: sum(x) / len(x) if x else None
    return moyenne(epoch_time), moyenne(reelles)


def estimation(epoques_faites, duree_epoque_s, unites_par_heure):
    """(époques restantes, heures restantes, unités restantes) ; heures et unités None si durée inconnue."""
    restantes = max(NB_EPOQUES - epoques_faites, 0)
    if not duree_epoque_s:
        return restantes, None, None
    heures = restantes * duree_epoque_s / 3600
    return restantes, heures, heures * unites_par_heure


def lire_id_run(dossier_pli):
    chemin = os.path.join(dossier_pli, FICHIER_RUN)
    return open(chemin, encoding="utf-8").read().strip() if os.path.isfile(chemin) else None


def ecrire_id_run(dossier_pli, id_run):
    os.makedirs(dossier_pli, exist_ok=True)
    with open(os.path.join(dossier_pli, FICHIER_RUN), "w", encoding="utf-8") as f:
        f.write(id_run + "\n")


def tracer(a, dossier_pli, ds_pre, chemin_splits, chemin_trace, trace, plis, gpu, statut, afficher=True):
    """Relit TOUS les journaux du pli et complète le run MLflow du pli (créé au premier appel) avec les
    époques consolidées. Renvoie (lus, époques), ou None si les journaux sont illisibles pour l'instant
    (Drive indisponible, par exemple). Une erreur de MLflow n'empêche pas de renvoyer les journaux."""
    try:
        lus, epoques = lire_journaux(dossier_pli)
        borne = borne_consolidee(dossier_pli, epoques)
    except Exception as err:
        print(f"ATTENTION : journaux illisibles pour l'instant ({type(err).__name__} : {err}). Suivi MLflow non mis à jour.",
              flush=True)
        return None
    tous = list(lus)
    series = {nom: [(k, v) for k, v in valeurs if k < borne]          # époques qui ne seront plus refaites
              for nom, valeurs in essai.series_par_epoque(epoques).items()}
    durees = essai.bilan_durees(epoques)[0]
    moyenne, reelle = durees_par_session(lus)
    dice_final = next((j["dice_validation_finale"] for j in reversed(list(lus.values()))
                       if j["dice_validation_finale"] is not None), None)
    params = {"pli": a.pli, "trainer": TRAINER, "configuration": CONFIGURATION, "nb_epoques": NB_EPOQUES,
              "point_de_reprise_toutes_les": 10, "nnunetv2": essai.version_paquet("nnunetv2"),
              "graine_plis": trace["graine"], "val_pli": ",".join(plis[a.pli]["val"]),
              "sha256_splits_final": trace["sha256_splits_final"], "torch_compile": not a.sans_compile}
    tags = {"etape": "5", "type": "entrainement", "pli": str(a.pli), "statut": statut,
            "mlflow.source.git.commit": prep.commit_git()}
    if gpu is not None:                           # une session d'entraînement (pas un simple retraçage)
        tags.update({f"session_{SESSION}.gpu": gpu, f"session_{SESSION}.commande": " ".join(sys.argv)[:500]})
    metriques = {"epoques_terminees": len(durees), "epoques_consolidees": borne, "duree_epoque_moyenne_s": moyenne,
                 "duree_epoque_reelle_s": reelle, "dice_validation_finale": dice_final}
    artefacts = tous + [os.path.join(dossier_pli, f) for f in ("progress.png", "debug.json")] + \
        [os.path.join(dossier_pli, "validation", "summary.json"), chemin_splits, chemin_trace,
         os.path.join(ds_pre, "nnUNetPlans.json"), os.path.join(ds_pre, prep.NOM_MARQUEUR)]
    try:
        id_run = essai.tracer_mlflow(a.mlflow, f"entrainement_{TRAINER}_pli{a.pli}", params, tags, series,
                                     metriques, artefacts, id_run=lire_id_run(dossier_pli),
                                     remplacer_depuis=getattr(a, "remplacer_depuis", None))
        ecrire_id_run(dossier_pli, id_run)
        if afficher:
            print(f"MLflow mis à jour : run {id_run} (pli {a.pli}, statut {statut}, {len(durees)} époques faites, "
                  f"{borne} notées : celles couvertes par un point de reprise)", flush=True)
    except Exception as err:
        print(f"ATTENTION : traçage MLflow impossible ({err}). Relancer plus tard avec --tracer-seulement.", flush=True)
    return lus, epoques


def tracer_ou_arreter(*args, essais=3, pause_s=20):
    """tracer(), réessayé quelques fois si les journaux sont illisibles ; sinon arrêt (code 3)."""
    for i in range(essais):
        resultat = tracer(*args)
        if resultat is not None:
            return resultat
        if i < essais - 1:
            time.sleep(pause_s)
    print("ARRÊT : journaux toujours illisibles. Relancer plus tard avec --tracer-seulement.")
    sys.exit(3)


def lire_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--pli", type=int, required=True, choices=range(prep.NB_PLIS))
    p.add_argument("--raw", default=f"{essai.DRIVE}/nnunet_raw")
    p.add_argument("--preprocessed", default=f"{essai.DRIVE}/nnunet_preprocessed")
    p.add_argument("--results", default=f"{essai.DRIVE}/nnunet_results")
    p.add_argument("--mlflow", default=f"{essai.DRIVE}/mlflow")
    p.add_argument("--split", default=os.path.join(essai.DEPOT, "splits", "mmwhs_split.json"))
    p.add_argument("--id-dataset", type=int, default=101)
    p.add_argument("--local", default="/content/nnunet_preprocessed_local")
    p.add_argument("--lire-depuis-drive", action="store_true")
    p.add_argument("--processus", type=int, default=2, choices=[1, 2, 3, 4])
    p.add_argument("--processus-augmentation", type=int, default=None)
    p.add_argument("--sans-compile", action="store_true")
    p.add_argument("--unites-par-heure", type=float, default=1.94)
    p.add_argument("--tracer-toutes-les", type=float, default=30.0)
    p.add_argument("--ignorer-verrou", action="store_true")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--terminer-validation", action="store_true")
    g.add_argument("--tracer-seulement", action="store_true")
    return p.parse_args()


# --- Programme principal ----------------------------------------------------------------------
def main():
    a = lire_arguments()
    a.remplacer_depuis = None                     # fixé seulement en cas d'incohérence à la reprise (voir 5.)
    raw, pre, res = (os.path.abspath(x) for x in (a.raw, a.preprocessed, a.results))
    a.mlflow = os.path.abspath(a.mlflow)
    nom_ds = f"Dataset{a.id_dataset:03d}_MMWHS"
    for nom, d in (("--raw", raw), ("--preprocessed", pre), ("--results", res), ("--mlflow", a.mlflow),
                   ("--local", a.local)):
        if prep.dans(d, essai.DEPOT):
            sys.exit(f"ARRÊT : {nom} est dans le dépôt. Les données et résultats restent hors du dépôt.")

    # --- 1. Prérequis de 05a (marqueur, plis) ----------------------------------------------------
    ds_pre, chemin_splits, chemin_trace, trace, plis = essai.verifier_prerequis_05a(raw, pre, nom_ds, a.split, a.pli)
    dossier_pli = os.path.join(res, nom_ds, f"{TRAINER}__nnUNetPlans__{CONFIGURATION}", f"fold_{a.pli}")
    contexte = (a, dossier_pli, ds_pre, chemin_splits, chemin_trace, trace, plis)

    # --- 2. Où en est ce pli ? --------------------------------------------------------------------
    etat = etat_du_pli(dossier_pli)
    print(f"Dossier du pli : {dossier_pli}\nÉtat : {etat}")
    if a.tracer_seulement or (etat == "termine" and not a.terminer_validation):
        if not glob.glob(os.path.join(dossier_pli, "training_log_*.txt")):
            sys.exit(f"ARRÊT : aucun journal dans {dossier_pli}.")
        statut = "termine" if etat == "termine" else "en_cours"
        lus, epoques = tracer_ou_arreter(*contexte, None, statut)
        if etat == "termine":
            print(f"Le pli {a.pli} est déjà TERMINÉ (checkpoint_final.pth et validation finale présents). "
                  "Rien n'a été relancé.")
        return bilan(a, contexte, lus, epoques, list(lus), {chemin_splits, os.path.join(os.path.abspath(a.local),
                                                                                       nom_ds, prep.NOM_SPLITS)}, 0)
    if etat == "validation_incomplete" and not a.terminer_validation:
        print("ARRÊT : checkpoint_final.pth existe mais la validation finale n'a pas abouti (summary.json absent).")
        print("  Relancer avec --terminer-validation pour la refaire (aucune époque ne sera réentraînée).")
        sys.exit(2)
    verrou = os.path.join(dossier_pli, NOM_VERROU)
    verifier_verrou(verrou, a.ignorer_verrou)     # une autre session entraîne-t-elle déjà ce pli ?

    # --- 3. GPU et variante d'entraînement --------------------------------------------------------
    import torch
    if not torch.cuda.is_available():
        print("ARRÊT : pas de GPU visible (Exécution → Modifier le type d'exécution → GPU L4).")
        sys.exit(5)
    gpu = torch.cuda.get_device_name(0)
    if "L4" not in gpu:
        print(f"ATTENTION : GPU {gpu}, pas un L4. L'entraînement marche, mais les durées et le coût estimés (D25) "
              "ne valent que pour le L4 (sur T4 : environ 3,7 fois plus lent).")
    os.environ["nnUNet_extTrainer"] = DOSSIER_TRAINERS          # pour la vérification ci-dessous
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
    classe = recursive_find_trainer_class_by_name(TRAINER)
    # nnU-Net retire le module de sys.modules après la recherche : inspect.getfile(classe) échoue alors.
    # Le code de __init__ garde, lui, le chemin du fichier où la classe est définie.
    fichier = classe.__init__.__code__.co_filename
    if os.path.normcase(os.path.dirname(os.path.abspath(fichier))) != os.path.normcase(DOSSIER_TRAINERS):
        sys.exit(f"ARRÊT : la classe {TRAINER} trouvée ne vient pas de {DOSSIER_TRAINERS}.")
    exe = shutil.which("nnUNetv2_train")
    if exe is None:
        sys.exit("ARRÊT : nnUNetv2_train introuvable. Installer nnU-Net (notebook, section 12).")
    print(f"GPU : {gpu} ; nnunetv2 {essai.version_paquet('nnunetv2')} ; trainer {TRAINER} ({DOSSIER_TRAINERS})")

    # --- 4. Environnement et copie locale -----------------------------------------------------------
    env = prep.environnement_nnunet(raw, pre, res, a.processus)   # contient PYTHONUNBUFFERED=1
    env["nnUNet_extTrainer"] = DOSSIER_TRAINERS
    if a.sans_compile:
        env["nnUNet_compile"] = "f"
    if a.processus_augmentation:
        env["nnUNet_n_proc_DA"] = str(a.processus_augmentation)
    chemin_splits_vu = chemin_splits
    if not a.lire_depuis_drive:
        chemin_splits_vu = essai.copie_locale(ds_pre, a.local, nom_ds, trace)
        env["nnUNet_preprocessed"] = os.path.abspath(a.local)

    # --- 5. Lancement, avec reprise automatique ----------------------------------------------------
    commande = [exe, str(a.id_dataset), CONFIGURATION, str(a.pli), "-tr", TRAINER]
    ckpt = point_de_reprise(dossier_pli)
    if ckpt:
        commande.append("--c")
        epoque = epoque_de_reprise(ckpt)
        print(f"REPRISE depuis {os.path.basename(ckpt)} : époque {epoque if epoque is not None else 'inconnue'} "
              f"sur {NB_EPOQUES}")
        lus_avant, epoques_avant = lire_journaux(dossier_pli)
        # Contrôle de cohérence : l'époque réelle de reprise doit être la borne calculée depuis les journaux.
        borne = borne_consolidee(dossier_pli, epoques_avant)
        if epoque is not None and epoque < borne:
            # Cas de secours (ex. : point de reprise écrit mais pas encore recopié par Drive avant la coupure) :
            # les époques epoque..borne-1 déjà notées dans MLflow vont être refaites. Leurs nouvelles valeurs
            # seront ajoutées ; dans MLflow, la valeur la plus récente d'une époque fait foi.
            a.remplacer_depuis = epoque
            print(f"ATTENTION : reprise à l'époque {epoque}, mais les journaux indiquaient {borne}. Les époques "
                  f"{epoque} à {borne - 1} vont être refaites ; MLflow recevra leurs nouvelles valeurs.")
        elif epoque is not None and epoque > borne:
            print(f"  (journaux en retard sur le point de reprise : {borne} ; le suivi MLflow se complétera)")
        if etat == "reprise" and epoque is not None:
            restantes, heures, unites = estimation(epoque, durees_par_session(lus_avant)[1], a.unites_par_heure)
            if heures:
                print(f"  Reste {restantes} époques : ≈ {heures:.1f} h, ≈ {unites:.1f} unités (indicatif)")
    else:
        print(f"NOUVEL ENTRAÎNEMENT du pli {a.pli} ({NB_EPOQUES} époques, point de reprise toutes les 10)")
    avant = set(glob.glob(os.path.join(dossier_pli, "training_log_*.txt")))
    print("Commande :", " ".join(commande), flush=True)
    t0, interrompu, proc = time.time(), False, None
    poser_verrou(verrou)
    try:
        proc = subprocess.Popen(commande, env=env)    # sortie de nnU-Net affichée telle quelle
        dernier_trace = time.time()
        while True:
            try:
                proc.wait(timeout=min(PAS_VERROU_S, a.tracer_toutes_les * 60))
                break
            except subprocess.TimeoutExpired:
                try:
                    poser_verrou(verrou)              # « cette session est toujours là »
                    if time.time() - dernier_trace >= a.tracer_toutes_les * 60:
                        dernier_trace = time.time()
                        tracer(*contexte, gpu, "en_cours")   # suivi MLflow recopié dans Drive
                except Exception as err:          # jamais d'arrêt du suivi pendant que nnU-Net tourne
                    print(f"ATTENTION : suivi non mis à jour ({type(err).__name__} : {err}) ; l'entraînement continue.",
                          flush=True)
    except KeyboardInterrupt:                     # cellule arrêtée à la main
        interrompu = True
    finally:
        # Quoi qu'il arrive, ne jamais laisser nnU-Net tourner seul : une relance de la cellule lancerait
        # sinon un 2e entraînement dans le même pli.
        if proc is not None and proc.poll() is None:
            print("Arrêt de nnU-Net...", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        retirer_verrou(verrou)
    code = proc.returncode
    print(f"\nnnU-Net {'interrompu' if interrompu else 'terminé'} (code {code}) après "
          f"{(time.time() - t0) / 3600:.2f} h", flush=True)

    # --- 6. Bilan --------------------------------------------------------------------------------------
    statut = "interrompu" if interrompu else ("echec_nnunet" if code else
                                              ("termine" if etat_du_pli(dossier_pli) == "termine" else "en_cours"))
    lus, epoques = tracer_ou_arreter(*contexte, gpu, statut)
    nouveaux = [j for j in lus if j not in avant]
    return bilan(a, contexte, lus, epoques, nouveaux, {chemin_splits_vu}, 3 if (code or interrompu) else 0)


def bilan(a, contexte, lus, epoques, nouveaux, acceptes, code):
    """Contrôle des plis dans les journaux, durées, estimation, verdict. Termine le programme."""
    _, dossier_pli, _, _, _, _, plis = contexte
    if code and not nouveaux:                     # nnU-Net s'est arrêté avant d'écrire son journal : erreur (3), pas 6
        print("nnU-Net s'est arrêté avant d'écrire son journal : voir les messages d'erreur ci-dessus.")
        usage = []
    else:
        usage = essai.verifier_usage_splits({os.path.basename(j): lus[j] for j in nouveaux}, acceptes, a.pli, plis)
    val_vus = {os.path.basename(x)[:-len(".nii.gz")] for x in glob.glob(os.path.join(dossier_pli, "validation", "*.nii.gz"))}
    if val_vus and val_vus != set(plis[a.pli]["val"]):
        usage.append(f"cas prédits en validation finale {sorted(val_vus)} ≠ validation du pli {plis[a.pli]['val']}")
    essai.afficher_epoques(epoques, dernieres=10)
    durees = essai.bilan_durees(epoques)[0]
    moyenne, reelle = durees_par_session(lus)
    reelle = reelle or moyenne
    if moyenne:
        print(f"\nÉpoques terminées : {len(durees)} sur {NB_EPOQUES}. Durée moyenne d'une époque (sans la 1re de "
              f"chaque session) : {moyenne:.1f} s « Epoch time », {reelle:.1f} s en réalité (points de reprise compris)")
        restantes, heures, unites = estimation(len(durees), reelle, a.unites_par_heure)
        if restantes:
            print(f"Reste {restantes} époques : ≈ {heures:.1f} h et ≈ {unites:.1f} unités de calcul "
                  f"({a.unites_par_heure} unités/h), plus la validation finale. Estimation indicative.")
    dice = next((j["dice_validation_finale"] for j in reversed(list(lus.values()))
                 if j["dice_validation_finale"] is not None), None)
    if dice is not None:
        print(f"Dice moyen de validation finale (pli {a.pli}, 3 cas d'entraînement, PAS le test) : {dice:.4f}")
    if usage:
        print("Résultat : NON CONFIRMÉ, nnU-Net n'a peut-être pas utilisé nos plis. Ne pas utiliser ces résultats :")
        for u in usage:
            print("  -", u)
        sys.exit(6)
    if code:
        print("Résultat : session INTERROMPUE ou en erreur. Pour reprendre, relancer la même cellule "
              "(au besoin dans une nouvelle session, après les sections 1, 2 et 12).")
        sys.exit(code)
    termine = etat_du_pli(dossier_pli) == "termine"
    print(f"Résultat : pli {a.pli} {'TERMINÉ' if termine else 'en cours'}. Le lot de test n'a été ni lu ni utilisé.")


if __name__ == "__main__":
    main()
