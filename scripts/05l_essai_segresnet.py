"""
Étape 5l — Essai court d'Auto3DSeg (MONAI 1.6.1), SegResNet seul, pli 0 seulement (D37) : durée d'une époque,
mémoire GPU, estimation pour les 5 plis. Rien n'est décidé ici : le nombre d'époques sera choisi ensuite.

Ce que fait le script :
  1. garde-fous : dossiers hors du dépôt et sans espace (MONAI découpe ses commandes aux espaces) ; MONAI
     installé en version 1.6.1 exactement ; GPU présent (avertit si ce n'est pas un L4) ; liste de données de
     05k inchangée (empreinte de sa fiche) et TOUJOURS conforme à splits_final.json (contrôles de 05k refaits) ;
  2. copie sur le disque local de Colab les SEULS 30 fichiers de la liste (15 images, 15 masques de imagesTr
     et labelsTr). Le dossier lu par Auto3DSeg ne contient donc physiquement rien d'autre : ni imagesTs, ni
     labelsTs, ni un cas hors liste ;
  3. modèles d'algorithmes d'Auto3DSeg (« templates ») épinglés : archive 21ed8e5.tar.gz (version par défaut
     de MONAI 1.6.1), téléchargée une fois dans Drive, empreinte SHA-256 vérifiée à chaque lancement ;
  4. AutoRunner de MONAI, limité à : analyse des données (DataAnalyzer) + génération de l'algorithme
     segresnet pour le pli 0 seulement (set_num_fold(1)), SANS entraînement ni ensemble ni inférence
     (train=False, ensemble=False). Le nombre d'époques est fixé dans la configuration (num_epochs), sinon le
     modèle d'algorithme le recalculerait (1 250 époques pour 15 cas). Contrôles : un seul dossier
     d'algorithme (segresnet_0), fold = 0, num_epochs demandé, analyse faite sur 15 cas exactement ;
  5. entraînement du pli 0 avec la commande que construit MONAI (BundleAlgo._create_cmd), mais lancée par ce
     script pour afficher la progression en direct (MONAI la cache jusqu'à la fin). Le journal training.log
     est suivi et recopié à l'écran ; la mémoire GPU est relevée toutes les 2 s (nvidia-smi) ;
  6. bilan : contrôle que le pli 0 a bien validé sur les 3 cas de splits_final.json, durée d'une époque
     (entraînement, validation, temps réel), pic de mémoire GPU, estimation pour les 5 plis selon
     --epoques-cible (heures et unités de calcul L4), suivi MLflow (même base que nnU-Net, expérience
     « etape5_auto3dseg_segresnet »).
  Tout l'affichage est recopié dans <travail>/sortie_05l_<SESSION>.txt (Drive), heures en UTC et Paris (05c).
  Le script ne lance AUCUNE prédiction sur le lot de test, et la liste de données n'a pas de clé « testing ».

Reprise après une coupure de session : relancer la même cellule. Ce qui est repris :
  - l'analyse des données et la génération de segresnet_0 (cache.yaml d'AutoRunner, dans Drive) ;
  - l'entraînement, depuis model_final.pt (écrit à chaque époque, num_epochs_per_saving = 1) : poids du
    réseau, numéro d'époque, et une valeur de « meilleur Dice » (option continue du modèle segresnet).
  Ce qui est refait ou perdu : l'époque en cours au moment de la coupure ; l'état de l'optimiseur AdamW
  (ses moyennes internes repartent de zéro) et du GradScaler ; la mise en cache des images en mémoire. Le
  « meilleur Dice » repris est celui de la dernière époque sauvegardée, pas forcément le meilleur : model.pt
  peut ensuite être remplacé par un modèle un peu moins bon. Sans effet sur la MESURE de durée de l'essai.
  Deux cas particuliers, vérifiés dans le code du modèle segresnet (segmenter.py de l'archive 21ed8e5) :
    a) si la coupure suit une époque SANS validation, le « meilleur Dice » repris vaut −1 (valeur écrite
       dans model_final.pt dans ce cas, l. 1385 et 1510) : la première validation après la reprise
       remplacera donc toujours model.pt ;
    b) avec num_crops_per_image > 1, une « boucle » couvre plusieurs époques et le numéro sauvegardé est
       celui du début de la boucle (l. 1029 et 1300) : la reprise refait la dernière boucle entière.
  Verrou (une seule session à la fois, comme 05c) : <travail>/verrou_essai_segresnet_pli0, posé avant toute
  écriture (y compris l'effacement par --ecraser) et retiré à la fin du script.

Note pour l'entraînement des 5 plis (étape suivante, PAS ce script) : après une reprise, le modèle réécrit
  sa configuration avec « continue: true » (config_save_updated). Un nouveau pli ou un nouvel entraînement
  lancé à partir de ce dossier reprendrait donc l'ancien point au lieu de partir de zéro : il faudra
  générer chaque pli dans un dossier neuf ou remettre « continue » à false et le contrôler.

Entrées :
  --raw / --preprocessed / --split   comme 05k (contrôle de la liste contre splits_final.json)
  --travail       dossier Auto3DSeg dans Drive (défaut /content/drive/MyDrive/poc_ct/auto3dseg) : liste de 05k,
                  archive des modèles, et dossier de l'essai essai_segresnet_pli0/
  --mlflow        dossier MLflow dans Drive (défaut /content/drive/MyDrive/poc_ct/mlflow, comme 05b)
  --local         copie locale des 30 fichiers (défaut /content/auto3dseg_donnees_local)
  --epoques       époques de l'essai (défaut 5, comme 05b)
  --epoques-cible nombres d'époques pour l'estimation des 5 plis (défaut 300 1250 : 300 = valeur de base du
                  modèle segresnet ; 1250 = valeur qu'il calculerait lui-même pour 15 cas)
  --unites-par-heure  défaut 1,94 (L4, D25)
  --ecraser       efface l'essai existant (dossier essai_segresnet_pli0/) et recommence
  --tracer-seulement  ne lance rien : relit le journal et met à jour MLflow
  --ignorer-verrou    comme 05c
Codes de sortie :
  0 = essai terminé et tracé ; 1 = garde-fou (rien n'est lancé) ; 2 = un essai existe avec d'autres réglages
  (--ecraser) ; 3 = entraînement interrompu ou en erreur (relancer la même cellule pour reprendre) ;
  5 = pas de GPU ; 6 = le pli 0 n'a pas été validé sur les cas attendus (ne pas utiliser les résultats).

Vocabulaire :
  - SegResNet : réseau de segmentation 3D (encodeur-décodeur à blocs résiduels) ; la version d'Auto3DSeg
    (SegResNetDS) apprend aussi à plusieurs échelles (« supervision profonde ») ;
  - époque (chez SegResNet) : un passage sur les 12 images d'entraînement du pli (parfois répété :
    num_steps_per_image), PAS 250 pas comme chez nnU-Net. Les durées ne se comparent donc pas époque par
    époque, seulement en heures totales ;
  - modèle d'algorithme (template) : dossier de scripts et de réglages que MONAI copie et remplit avec les
    statistiques des données ; il est téléchargé à part de MONAI, d'où son épinglage par empreinte ;
  - pic de mémoire GPU : la plus haute occupation de la mémoire de la carte pendant l'entraînement, lue par
    nvidia-smi (carte entière, y compris ~0,5 Go pris par ce script lui-même) ;
  - Dice de validation : recouvrement prédiction / vérité sur les 3 cas de validation du pli (PAS le test).
"""
import argparse, atexit, datetime, glob, importlib.util, json, os, re, shutil, subprocess, sys, tarfile, threading, time
import urllib.request

ICI = os.path.dirname(os.path.abspath(__file__))


def _charger(nom, fichier):
    spec = importlib.util.spec_from_file_location(nom, os.path.join(ICI, fichier))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)              # aucun de ces scripts n'exécute quoi que ce soit à l'import
    return module


prep05k = _charger("prep05k", "05k_preparer_auto3dseg.py")   # apporte aussi 05b (essai) et 05a (prep)
suivi05c = _charger("suivi05c", "05c_entrainer_nnunet.py")   # copie de la sortie, heures, verrou, id du run
essai, prep = prep05k.essai, prep05k.prep

VERSION_MONAI = "1.6.1"
HASH_MODELES = "21ed8e5"                          # MONAIEnvVars.algo_hash() par défaut dans MONAI 1.6.1
URL_MODELES = ("https://github.com/Project-MONAI/research-contributions/releases/download/algo_templates/"
               f"{HASH_MODELES}.tar.gz")
SHA256_MODELES = "b4f24826bb974d4627d28c7dfe3b64bb8bcae455ec7a7edbeeaadd3168eebfc2"
ALGO, PLI = "segresnet", 0
NOM_ALGO = f"{ALGO}_{PLI}"                        # nom donné par BundleGen.generate : <algo>_<pli>
NOM_ESSAI = "essai_segresnet_pli0"
NOM_FICHE = "essai_05l.json"
EXPERIENCE_MLFLOW = "etape5_auto3dseg_segresnet"
NB_PLIS = 5


# --- Lecture du journal de SegResNet (training.log, testable sans GPU) -------------------------
# Lignes écrites par segmenter.py du modèle 21ed8e5 (format « %(message)s », sans date).
MOTIFS = {
    "num_epochs": r"^\s*Using num_epochs => (\d+)", "start_epoch": r"^\s*Using start_epoch => (\d+)",
    "batch_size": r"^\s*batch_size => (\d+)", "num_crops_per_image": r"^\s*num_crops_per_image => (\d+)",
    "num_steps_per_image": r"^\s*num_steps_per_image => (\d+)",
    "fichiers": r"^train_files files (\d+), validation files (\d+)",
    "cache": r"^Calculating cache required (\d+)GB, available RAM (\d+)GB",
    "roi": r"ROI size \[[^\]]*\] => \[([^\]]*)\]", "parametres": r"^Total parameters count: (\d+)",
}
AVANT_SESSION = ("roi", "parametres")             # écrits dans Segmenter.__init__, avant « train_files files »
RE_ENTR = re.compile(r"^Final training\s+(\d+)/(\d+) loss: (\S+) acc_avg: (\S+) .*time ([\d.]+)s\s+lr: (\S+)")
RE_VAL = re.compile(r"^Final validation (\d+)/(\d+) loss: (\S+) acc_avg: (\S+) acc: \[([^\]]*)\] time: ([\d.]+)s")
RE_ORIG = re.compile(r"^Original resolution validation: loss: (\S+) acc_avg: (\S+) acc \[([^\]]*)\] time ([\d.]+)s")
RE_FIN = re.compile(r"^=== DONE: best_metric: (\S+) at epoch: (\d+)")


def lire_journal(texte):
    """Découpe training.log en sessions (une par lancement) et relève, pour chacune, les réglages effectifs et
    chaque époque. Une époque refaite après une reprise garde sa dernière valeur dans « epoques ».
    Début de session = ligne « train_files files ... » : segmenter.py l'écrit une fois par lancement (train(),
    l.1176), avant le cache et « Using num_epochs ». « Segmenter started » (l.594) n'est PAS dans le fichier :
    il est émis avant logger_configure (l.609), qui ajoute le fichier. « ROI size » et « Total parameters
    count » sont écrits avant (dans __init__) : gardés en attente et rattachés à la session qui suit."""
    sessions, epoques, attente = [], {}, {}
    for ligne in texte.splitlines():
        if re.match(MOTIFS["fichiers"], ligne):
            sessions.append({"reglages": attente, "epoques": [], "fin": None, "orig": None})
            attente = {}
        for nom, motif in MOTIFS.items():
            if (m := re.search(motif, ligne)):
                cible = attente if (nom in AVANT_SESSION or not sessions) else sessions[-1]["reglages"]
                cible[nom] = m.groups() if len(m.groups()) > 1 else m.group(1)
        if not sessions:
            continue
        s = sessions[-1]
        if (m := RE_ENTR.match(ligne)):
            e = int(m.group(1))
            epoques[e] = {"train_loss": float(m.group(3)), "train_time_s": float(m.group(5))}
            s["epoques"].append(e)
        elif (m := RE_VAL.match(ligne)) and int(m.group(1)) in epoques:
            epoques[int(m.group(1))].update({"val_dice_moyen": float(m.group(4)), "val_time_s": float(m.group(6)),
                                             "val_dice": [float(x) for x in m.group(5).split()]})
        elif (m := RE_ORIG.match(ligne)):
            s["orig"] = {"dice_moyen": float(m.group(2)), "dice": [float(x) for x in m.group(3).split()],
                         "duree_s": float(m.group(4))}
        elif (m := RE_FIN.match(ligne)):
            s["fin"] = {"best_metric": float(m.group(1)), "epoque": int(m.group(2))}
    return {"sessions": sessions, "epoques": epoques}


def calendrier_validation(nb, par_validation=None, fraction=0.16):
    """Copie de schedule_validation_epochs (segmenter.py, modèle 21ed8e5) : époques où la validation a lieu."""
    import numpy as np
    if par_validation is None:
        x = (np.sin(np.linspace(0, np.pi / 2, max(10, int(fraction * nb)))) * nb).astype(int)
        x = np.cumsum(np.sort(np.diff(np.unique(x)))[::-1])
        x[-1] = nb
        x = x.tolist()
    else:
        x = [par_validation] if par_validation >= nb else list(range(par_validation, nb, par_validation))
    return x or [0]


def estimer(nb_epoques, crops, t_entr, t_val, t_autre, t_orig, unites_par_heure, t_demarrage=0):
    """Estimation INDICATIVE d'un pli puis des 5 plis, pour nb_epoques demandées dans la configuration.
    Comme segmenter.py : boucles = nb_epoques // min(3, crops) ; validations selon calendrier_validation.
    t_demarrage : temps du lancement à la 1re époque (chargement, mise en cache), payé une fois par pli."""
    boucles = max(1, nb_epoques // min(3, crops))
    nb_val = len(calendrier_validation(boucles, None, min(0.3, 0.16 * crops)))
    pli_s = (t_demarrage or 0) + boucles * (t_entr + t_autre) + nb_val * t_val + (t_orig or 0)
    return {"boucles": boucles, "validations": nb_val, "h_par_pli": pli_s / 3600,
            "h_5_plis": NB_PLIS * pli_s / 3600, "unites_5_plis": NB_PLIS * pli_s / 3600 * unites_par_heure}


def moyenne(x):
    return sum(x) / len(x) if x else None


# --- Fils de surveillance pendant l'entraînement ------------------------------------------------
class SuiviJournal(threading.Thread):
    """Lit training.log au fur et à mesure (Drive), recopie les nouvelles lignes à l'écran et note l'heure
    d'arrivée de chaque « Final training » (le journal de SegResNet n'est pas daté) : l'écart entre deux
    arrivées donne la durée RÉELLE d'une époque (entraînement + validation + sauvegarde dans Drive)."""
    def __init__(self, chemin):
        super().__init__(daemon=True)
        self.chemin, self.arret, self.arrivees, self.debut = chemin, threading.Event(), [], None
        self.position = os.path.getsize(chemin) if os.path.exists(chemin) else 0

    def run(self):
        reste = ""
        while not self.arret.wait(2):
            try:
                if not os.path.exists(self.chemin):
                    continue
                with open(self.chemin, encoding="utf-8", errors="replace") as f:
                    f.seek(self.position)
                    morceau = f.read()
                    self.position = f.tell()
                lignes = (reste + morceau).split("\n")
                reste = lignes.pop()                 # dernière ligne peut-être incomplète
                for l in lignes:
                    print(f"  | {l}", flush=True)
                    if RE_ENTR.match(l):
                        self.arrivees.append(time.time())
            except Exception:                        # Drive momentanément indisponible : on réessaie
                pass


class SuiviGPU(threading.Thread):
    """Relève toutes les 2 s la mémoire occupée sur la carte (nvidia-smi, en Mo) et garde le maximum."""
    def __init__(self):
        super().__init__(daemon=True)
        self.arret, self.pic, self.total, self.base = threading.Event(), None, None, self.lire()[0]

    @staticmethod
    def lire():
        try:
            sortie = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
            occupe, total = (int(x) for x in sortie.split(","))
            return occupe, total
        except Exception:
            return None, None

    def run(self):
        while not self.arret.wait(2):
            occupe, total = self.lire()
            if occupe is not None:
                self.pic, self.total = max(self.pic or 0, occupe), total


# --- Préparation : liste, copie locale, modèles d'algorithmes ----------------------------------
def verifier_liste(a, nom_ds, pli=PLI):
    """Liste de 05k : même empreinte que sa fiche, et contrôles de 05k refaits contre splits_final.json.
    pli : pli dont la validation est affichée (0 pour cet essai ; 05m passe le pli qu'il entraîne)."""
    chemin_liste = os.path.join(a.travail, prep05k.NOM_LISTE)
    chemin_fiche = os.path.join(a.travail, prep05k.NOM_FICHE)
    if not (os.path.isfile(chemin_liste) and os.path.isfile(chemin_fiche)):
        sys.exit(f"ARRÊT : {chemin_liste} ou sa fiche manque. Lancer d'abord le script 05k.")
    with open(chemin_fiche, encoding="utf-8") as f:
        fiche = json.load(f)
    if prep.sha256_fichier(chemin_liste) != fiche.get("sha256_liste"):
        sys.exit("ARRÊT : la liste de données a changé depuis 05k (empreinte différente). Relancer 05k.")
    chemin_splits, _, plis, cas, cas_test, groupes = prep05k.verifier_plis_05a(a.raw, a.preprocessed, nom_ds, a.split)
    if prep.sha256_fichier(chemin_splits) != fiche.get("sha256_splits_final"):
        sys.exit("ARRÊT : splits_final.json a changé depuis 05k. Relancer 05k.")
    with open(chemin_liste, encoding="utf-8") as f:
        liste = json.load(f)
    problemes = prep05k.controler_liste(liste, plis, cas, cas_test, groupes)
    if problemes:
        sys.exit("ARRÊT : liste non conforme :\n  - " + "\n  - ".join(problemes))
    print(f"Liste de données vérifiée ({len(liste['training'])} cas, plis = splits_final.json, aucun cas de test). "
          f"Pli {pli} : validation = {plis[pli]['val']}")
    return chemin_liste, fiche, liste, plis


def copier_donnees(ds_raw, liste, local):
    """Copie dans `local` les seuls fichiers de la liste (mêmes chemins relatifs). Un fichier déjà copié
    (même taille, même date) n'est pas recopié. Le dossier local ne doit rien contenir d'autre."""
    attendus = {p for e in liste["training"] for p in (e["image"], e["label"])}
    presents = {os.path.relpath(os.path.join(r, f), local).replace("\\", "/")
                for r, _, fs in os.walk(local) for f in fs} if os.path.isdir(local) else set()
    if presents - attendus:                       # restes d'autre chose : on repart d'un dossier vide
        shutil.rmtree(local)
    t0, n, octets = time.time(), 0, 0
    for rel in sorted(attendus):
        s, c = os.path.join(ds_raw, rel), os.path.join(local, rel)
        st = os.stat(s)
        if os.path.exists(c) and os.stat(c).st_size == st.st_size and int(os.stat(c).st_mtime) == int(st.st_mtime):
            continue
        os.makedirs(os.path.dirname(c), exist_ok=True)
        shutil.copy2(s, c)
        n, octets = n + 1, octets + st.st_size
    presents = {os.path.relpath(os.path.join(r, f), local).replace("\\", "/") for r, _, fs in os.walk(local) for f in fs}
    if presents != attendus:
        sys.exit(f"ARRÊT : la copie locale ne contient pas exactement les {len(attendus)} fichiers de la liste.")
    print(f"Copie locale : {n} fichiers ({octets / 1e9:.2f} Go) copiés en {time.time() - t0:.0f} s ; "
          f"{local} contient exactement les {len(attendus)} fichiers de la liste (rien d'autre)")


def preparer_modeles(travail, local):
    """Archive des modèles d'algorithmes épinglée (hash 21ed8e5, SHA-256 vérifiée), gardée dans Drive,
    extraite dans un dossier local neuf. Renvoie le dossier « algorithm_templates » extrait."""
    archive = os.path.join(travail, f"algorithm_templates_{HASH_MODELES}.tar.gz")
    if not os.path.isfile(archive):
        print(f"Téléchargement des modèles d'algorithmes : {URL_MODELES}")
        urllib.request.urlretrieve(URL_MODELES, archive + ".tmp")
        os.replace(archive + ".tmp", archive)
    empreinte = prep.sha256_fichier(archive)
    if empreinte != SHA256_MODELES:
        sys.exit(f"ARRÊT : empreinte de {archive} = {empreinte}, attendu {SHA256_MODELES}. Fichier effacé à la main ?"
                 " Supprimer l'archive et relancer.")
    if os.path.isdir(local):
        shutil.rmtree(local)
    os.makedirs(local)
    with tarfile.open(archive) as t:
        try:
            t.extractall(local, filter="data")    # refuse les chemins dangereux (Python ≥ 3.12)
        except TypeError:
            t.extractall(local)
    dossier = os.path.join(local, "algorithm_templates")
    if not os.path.isfile(os.path.join(dossier, ALGO, "scripts", "algo.py")):
        sys.exit(f"ARRÊT : modèle {ALGO} introuvable dans l'archive {archive}.")
    print(f"Modèles d'algorithmes {HASH_MODELES} : empreinte SHA-256 conforme ({archive})")
    return dossier


def generer(travail_essai, chemin_liste, local, modeles, nb_epoques, uri_mlflow_interne, nb_plis=1, options=None):
    """AutoRunner limité à l'analyse des données et à la génération de segresnet_0 (aucun entraînement,
    aucun ensemble). Les étapes déjà faites (cache.yaml) ne sont pas refaites.
    nb_plis : AutoRunner génère les plis 0 à nb_plis − 1 (1 pour cet essai). options : réglages ajoutés à
    l'entrée, recopiés comme num_epochs dans hyper_parameters.yaml (05m : fréquence de validation...)."""
    from monai.apps.auto3dseg import AutoRunner
    entree = {"modality": "ct", "datalist": chemin_liste, "dataroot": local,
              "num_epochs": nb_epoques}          # recopié dans hyper_parameters.yaml (algo.py : config.update)
    entree.update(options or {})
    runner = AutoRunner(work_dir=travail_essai, input=entree, algos=ALGO, train=False, ensemble=False,
                        templates_path_or_url=modeles, mlflow_tracking_uri=uri_mlflow_interne,
                        mlflow_experiment_name="auto3dseg_segresnet_interne")
    runner.set_num_fold(nb_plis)                  # 1 : ne génère que le pli 0 (segresnet_0) ; les champs fold restent
    runner.run()


def _lire_yaml(chemin):
    """Lit un fichier YAML et referme le fichier aussitôt (bloc with)."""
    import yaml
    with open(chemin, encoding="utf-8") as f:
        return yaml.safe_load(f)


def controler_generation(travail_essai, local, chemin_liste, nb_epoques, cas, pli=PLI, attendus_en_plus=None):
    """Contrôles après AutoRunner. Renvoie (algo segresnet_<pli>, hyper-paramètres, problèmes).
    pli : 0 pour cet essai ; attendus_en_plus : autres valeurs exigées dans hyper_parameters.yaml (05m)."""
    from monai.apps.auto3dseg.utils import import_bundle_algo_history
    from monai.utils.enums import AlgoKeys
    pb = []
    nom_algo = f"{ALGO}_{pli}"                    # = NOM_ALGO pour le pli 0
    historique = import_bundle_algo_history(travail_essai, only_trained=False)
    noms = sorted(h[AlgoKeys.ID] for h in historique)
    if noms != [nom_algo]:
        pb.append(f"dossiers d'algorithme générés {noms} ≠ [{nom_algo}]")
    algo = next((h[AlgoKeys.ALGO] for h in historique if h[AlgoKeys.ID] == nom_algo), None)
    hp_chemin = os.path.join(travail_essai, nom_algo, "configs", "hyper_parameters.yaml")
    hp = _lire_yaml(hp_chemin) if os.path.isfile(hp_chemin) else {}
    copie_liste = os.path.join(travail_essai, os.path.basename(chemin_liste))
    attendu = {"fold": pli, "num_epochs": nb_epoques, "data_file_base_dir": os.path.abspath(local),
               "data_list_file_path": os.path.abspath(copie_liste), **(attendus_en_plus or {})}
    for cle, valeur in attendu.items():
        if hp.get(cle) != valeur:
            pb.append(f"hyper_parameters.yaml : {cle} = {hp.get(cle)!r}, attendu {valeur!r}")
    for cle in ("validation_key",):
        if hp.get(cle) is not None:
            pb.append(f"hyper_parameters.yaml : {cle} présent ({hp.get(cle)!r}) : la validation ne suivrait plus le pli")
    if (hp.get("infer") or {}).get("enabled") or (hp.get("validate") or {}).get("enabled"):
        pb.append("hyper_parameters.yaml : inférence ou validation seule activée")
    if not os.path.isfile(copie_liste) or prep.sha256_fichier(copie_liste) != prep.sha256_fichier(chemin_liste):
        pb.append(f"copie de la liste dans le dossier de l'essai ≠ liste de 05k ({copie_liste})")
    # L'analyse (DataAnalyzer) n'a vu que les 15 cas de la liste.
    stats = os.path.join(travail_essai, "datastats.yaml")
    par_cas = os.path.join(travail_essai, "datastats_by_case.yaml")
    try:
        n = _lire_yaml(stats)["stats_summary"]["n_cases"]
        vus = {prep05k._cas_du_chemin(c["image_filepath"])           # DataStatsKeys.BY_CASE_IMAGE_PATH
               for c in _lire_yaml(par_cas)["stats_by_cases"]}
        if n != len(cas) or vus != set(cas):
            pb.append(f"analyse des données sur {n} cas ({sorted(vus ^ set(cas))} en écart) au lieu des {len(cas)} de la liste")
    except Exception as err:
        pb.append(f"analyse des données illisible ({type(err).__name__} : {err})")
    return algo, hp, pb


def controler_pli(chemin_liste_copie, local, plis, pli=PLI):
    """Cas d'entraînement / validation du pli (0 pour cet essai) tels que les lira segmenter.py (même
    fonction datafold_read)."""
    from monai.auto3dseg.utils import datafold_read
    tr, val = datafold_read(datalist=chemin_liste_copie, basedir=local, fold=pli)
    tr = {prep05k._cas_du_chemin(d["image"]) for d in tr}
    val = {prep05k._cas_du_chemin(d["image"]) for d in val}
    if tr != set(plis[pli]["train"]) or val != set(plis[pli]["val"]):
        return [f"datafold_read(pli {pli}) : validation {sorted(val)} ≠ splits_final.json {plis[pli]['val']}"]
    return []


def etat_essai(dossier_modele, journal):
    """« neuf », « reprise » (model_final.pt présent) ou « termine » : « === DONE » écrit APRÈS le dernier début
    de session (ligne « train_files files »), c'est-à-dire dans la dernière session de lire_journal."""
    if journal["sessions"] and journal["sessions"][-1]["fin"] is not None:
        return "termine"
    return "reprise" if os.path.isfile(os.path.join(dossier_modele, "model_final.pt")) else "neuf"


def epoque_du_point(chemin):
    try:
        import torch
        return int(torch.load(chemin, map_location="cpu", weights_only=False).get("epoch"))
    except Exception:
        return None


class Verrou(threading.Thread):
    """Verrou de 05c (une seule session à la fois), posé AVANT toute écriture (--ecraser, analyse, génération)
    et rafraîchi toutes les 5 min par ce fil jusqu'à la fin du script, où il est retiré (atexit)."""
    def __init__(self, chemin, ignorer):
        super().__init__(daemon=True)
        self.chemin = chemin
        suivi05c.verifier_verrou(chemin, ignorer)   # arrêt (code 1) si une autre session tient l'essai
        suivi05c.poser_verrou(chemin)
        atexit.register(suivi05c.retirer_verrou, chemin)
        self.start()

    def run(self):
        while True:
            time.sleep(suivi05c.PAS_VERROU_S)
            try:
                suivi05c.poser_verrou(self.chemin)
            except Exception as err:
                print(f"ATTENTION : verrou non rafraîchi ({err}) ; le travail continue.", flush=True)


def lancer_entrainement(algo, params, dossier_modele, dossier_travail):
    """Lance la commande de MONAI (BundleAlgo._create_cmd, découpée aux espaces comme BundleAlgo._run_cmd),
    depuis le dossier de l'essai (fichiers que le modèle écrit dans le dossier courant : runner.log...),
    affiche la sortie et le journal en direct, relève la mémoire GPU. Renvoie (code, interrompu, suivis, durée).
    Le verrou est tenu par main() (classe Verrou)."""
    commande, _ = algo._create_cmd(params)
    morceaux = commande.split()
    if morceaux[0] == "python":                   # même interpréteur que ce script (celui de Colab)
        morceaux[0] = sys.executable
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(algo.device_setting["CUDA_VISIBLE_DEVICES"]), "PYTHONUNBUFFERED": "1"})
    print("Commande :", " ".join(morceaux))
    print(f"Lancement de l'entraînement : {suivi05c.heure()}", flush=True)
    journal = SuiviJournal(os.path.join(dossier_modele, "training.log"))
    gpu = SuiviGPU()
    t0, interrompu, proc, fil = time.time(), False, None, None
    try:
        os.makedirs(dossier_modele, exist_ok=True)
        journal.debut = time.time()               # pour le temps avant la 1re époque (chargement, cache)
        proc = subprocess.Popen(morceaux, env=env, cwd=dossier_travail, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        fil = threading.Thread(target=suivi05c.recopier_sortie, args=(proc.stdout,), daemon=True)
        for f in (fil, journal, gpu):
            f.start()
        proc.wait()
    except KeyboardInterrupt:
        interrompu = True
    finally:
        if proc is not None and proc.poll() is None:  # ne jamais laisser l'entraînement tourner seul
            print("Arrêt de l'entraînement...", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if fil is not None:
            fil.join(timeout=30)
        time.sleep(3)                             # dernières lignes du journal
        journal.arret.set()
        gpu.arret.set()
    return (proc.returncode if proc else 1), interrompu, journal, gpu, time.time() - t0


def lire_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=f"{prep.DRIVE}/nnunet_raw")
    p.add_argument("--preprocessed", default=f"{prep.DRIVE}/nnunet_preprocessed")
    p.add_argument("--split", default=os.path.join(prep.DEPOT, "splits", "mmwhs_split.json"))
    p.add_argument("--travail", default=f"{prep.DRIVE}/auto3dseg")
    p.add_argument("--mlflow", default=f"{prep.DRIVE}/mlflow")
    p.add_argument("--local", default="/content/auto3dseg_donnees_local")
    p.add_argument("--id-dataset", type=int, default=101)
    p.add_argument("--epoques", type=int, default=5)
    p.add_argument("--epoques-cible", type=int, nargs="+", default=[300, 1250])
    p.add_argument("--unites-par-heure", type=float, default=1.94)
    p.add_argument("--ignorer-verrou", action="store_true")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--ecraser", action="store_true")
    g.add_argument("--tracer-seulement", action="store_true")
    return p.parse_args()


# --- Programme principal ----------------------------------------------------------------------
def main():
    journal_sortie = suivi05c.Journal()
    sys.stdout, sys.stderr = suivi05c.Copie(sys.stdout, journal_sortie), suivi05c.Copie(sys.stderr, journal_sortie)
    print(f"Session {suivi05c.SESSION} (heure de la machine Colab, UTC) ; début : {suivi05c.heure()}")
    a = lire_arguments()
    for nom in ("raw", "preprocessed", "split", "travail", "mlflow", "local"):
        setattr(a, nom, os.path.abspath(getattr(a, nom)))
    nom_ds = f"Dataset{a.id_dataset:03d}_MMWHS"
    travail_essai = os.path.join(a.travail, NOM_ESSAI)
    dossier_modele = os.path.join(travail_essai, NOM_ALGO, "model")
    local_modeles, local_mlflow = a.local + "_modeles", a.local + "_mlflow_interne"

    # --- 1. Garde-fous ------------------------------------------------------------------------------
    for nom in ("raw", "preprocessed", "travail", "mlflow", "local"):
        d = getattr(a, nom)
        if prep.dans(d, prep.DEPOT):
            sys.exit(f"ARRÊT : --{nom} est dans le dépôt. Les données et résultats restent hors du dépôt.")
        if " " in d:
            sys.exit(f"ARRÊT : --{nom} contient un espace ({d}) : MONAI découpe ses commandes aux espaces.")
    if a.epoques < 1:
        sys.exit("ARRÊT : --epoques doit valoir au moins 1.")
    # Verrou posé AVANT toute écriture (--ecraser, analyse, génération, entraînement), dans le dossier
    # parent puisque --ecraser efface le dossier de l'essai. Pas de verrou en --tracer-seulement (lecture).
    if not a.tracer_seulement:
        os.makedirs(a.travail, exist_ok=True)
        Verrou(os.path.join(a.travail, f"verrou_{NOM_ESSAI}"), a.ignorer_verrou)
    if a.ecraser and os.path.isdir(travail_essai):
        shutil.rmtree(travail_essai)              # --ecraser demandé explicitement
        print(f"Essai précédent effacé : {travail_essai}")
    os.makedirs(travail_essai, exist_ok=True)
    journal_sortie.ouvrir(os.path.join(travail_essai, f"sortie_05l_{suivi05c.SESSION}.txt"))
    print(f"Copie de cet affichage : {journal_sortie.chemin}")
    import monai
    if monai.__version__ != VERSION_MONAI:
        sys.exit(f"ARRÊT : MONAI {monai.__version__} installé, {VERSION_MONAI} attendu (notebook, section 24, installation).")
    chemin_liste, fiche05k, liste, plis = verifier_liste(a, nom_ds)
    cas = [prep05k._cas_du_chemin(e["image"]) for e in liste["training"]]

    # Réglages de l'essai : un essai existant doit avoir été lancé avec les mêmes.
    reglages = {"monai": VERSION_MONAI, "modeles": HASH_MODELES, "sha256_modeles": SHA256_MODELES, "pli": PLI,
                "num_epochs": a.epoques, "sha256_liste": fiche05k["sha256_liste"],
                "sha256_splits_final": fiche05k["sha256_splits_final"], "dataroot": a.local}
    chemin_fiche = os.path.join(travail_essai, NOM_FICHE)
    if os.path.isfile(chemin_fiche):
        with open(chemin_fiche, encoding="utf-8") as f:
            anciens = json.load(f)["reglages"]
        if anciens != reglages:
            ecarts = {k: (anciens.get(k), v) for k, v in reglages.items() if anciens.get(k) != v}
            print(f"ARRÊT : un essai existe dans {travail_essai} avec d'autres réglages (ancien, demandé) : {ecarts}.")
            print("  Relancer avec --ecraser pour l'effacer et recommencer.")
            sys.exit(2)
    else:
        with open(chemin_fiche, "w", encoding="utf-8") as f:
            json.dump({"reglages": reglages, "cree_le": datetime.datetime.now().isoformat(timespec="seconds"),
                       "commit": prep.commit_git(), "decision": "D37"}, f, indent=2, ensure_ascii=False)

    import torch
    gpu, code, interrompu, suivi_j, suivi_g, duree = "non utilisé", 0, False, None, None, None
    chemin_log = os.path.join(dossier_modele, "training.log")

    def lire():
        if not os.path.isfile(chemin_log):
            return {"sessions": [], "epoques": {}}
        with open(chemin_log, encoding="utf-8", errors="replace") as f:   # fichier refermé aussitôt
            return lire_journal(f.read())

    if not a.tracer_seulement:
        if not torch.cuda.is_available():
            print("ARRÊT : pas de GPU visible (Exécution → Modifier le type d'exécution → GPU L4).")
            sys.exit(5)
        gpu = torch.cuda.get_device_name(0)
        if "L4" not in gpu:
            print(f"ATTENTION : GPU {gpu}, pas un L4 : durées et coût estimés ne vaudront que pour ce GPU.")
        print(f"GPU : {gpu} ; MONAI {monai.__version__} ; torch {torch.__version__}")
        # --- 2 et 3. Copie locale des 30 fichiers, modèles d'algorithmes ---------------------------
        copier_donnees(os.path.join(a.raw, nom_ds), liste, a.local)
        modeles = preparer_modeles(a.travail, local_modeles)
        # --- 4. Analyse + génération de segresnet_0 (reprises depuis cache.yaml si déjà faites) ------
        os.makedirs(local_mlflow, exist_ok=True)
        uri_interne = "sqlite:///" + os.path.join(local_mlflow, "mlflow.db").replace("\\", "/")
        t0 = time.time()
        os.chdir(travail_essai)     # MONAI écrit runner.log... dans le dossier courant : on le garde dans l'essai
        generer(travail_essai, chemin_liste, a.local, modeles, a.epoques, uri_interne)
        print(f"Analyse des données et génération : {time.time() - t0:.0f} s (0 s si déjà faites)")
    algo, hp, problemes = controler_generation(travail_essai, a.local, chemin_liste, a.epoques, cas)
    problemes += controler_pli(os.path.join(travail_essai, os.path.basename(chemin_liste)), a.local, plis)
    if algo is not None and (int(algo.device_setting.get("n_devices", 1)) > 1 or int(algo.device_setting.get("NUM_NODES", 1)) > 1):
        problemes.append(f"plusieurs GPU ou nœuds dans device_setting {algo.device_setting} : non prévu")
    if problemes:
        sys.exit("ARRÊT : génération non conforme :\n  - " + "\n  - ".join(problemes))
    print(f"Génération conforme : un seul algorithme ({NOM_ALGO}), fold = {PLI}, num_epochs = {a.epoques}, "
          f"analyse des données sur {len(cas)} cas, validation du pli {PLI} = {plis[PLI]['val']}")

    # --- 5. Entraînement (nouveau ou reprise) ----------------------------------------------------------
    etat = etat_essai(dossier_modele, lire())
    print(f"État de l'essai : {etat}")
    if not a.tracer_seulement and etat != "termine":
        params = {}
        if etat == "reprise":
            point = os.path.join(dossier_modele, "model_final.pt")
            params = {"continue": True, "pretrained_ckpt_name": point}
            print(f"REPRISE depuis model_final.pt (époque {epoque_du_point(point)} terminée) : poids et numéro d'époque "
                  "repris ; optimiseur et cache des images repartent de zéro (voir l'en-tête du script).")
        torch.cuda.empty_cache()
        code, interrompu, suivi_j, suivi_g, duree = lancer_entrainement(algo, params, dossier_modele, travail_essai)
        print(f"\nEntraînement {'interrompu' if interrompu else 'terminé'} (code {code}) le {suivi05c.heure()}, "
              f"après {duree / 60:.1f} min")

    # --- 6. Bilan --------------------------------------------------------------------------------------
    j = lire()
    epoques = j["epoques"]
    derniere = j["sessions"][-1] if j["sessions"] else {"reglages": {}, "epoques": [], "fin": None, "orig": None}
    r = derniere["reglages"]
    usage = []
    if r.get("fichiers") and tuple(int(x) for x in r["fichiers"]) != (len(plis[PLI]["train"]), len(plis[PLI]["val"])):
        usage.append(f"segmenter : {r['fichiers'][0]} cas d'entraînement / {r['fichiers'][1]} de validation, attendu "
                     f"{len(plis[PLI]['train'])} / {len(plis[PLI]['val'])}")
    crops = int(r.get("num_crops_per_image", 1))
    print("\nÉpoque | entraînement (s) | train_loss | validation (s) | Dice de validation moyen (3 cas du pli 0)")
    for e in sorted(epoques):
        x = epoques[e]
        print(f"{e:>6} | {x['train_time_s']:>16.1f} | {x['train_loss']:>10.4f} | {x.get('val_time_s', float('nan')):>14.1f} | "
              f"{x.get('val_dice_moyen', float('nan')):.4f}")
    # Durées sans la 1re époque de chaque session (mise en cache des images, compilation cuDNN).
    premieres = {s["epoques"][0] for s in j["sessions"] if s["epoques"]}
    t_entr = moyenne([x["train_time_s"] for e, x in epoques.items() if e not in premieres]) or \
        moyenne([x["train_time_s"] for x in epoques.values()])
    t_val = moyenne([x["val_time_s"] for e, x in epoques.items() if "val_time_s" in x and e not in premieres]) or \
        moyenne([x["val_time_s"] for x in epoques.values() if "val_time_s" in x]) or 0
    reelles = [b - a_ for a_, b in zip(suivi_j.arrivees, suivi_j.arrivees[1:])] if suivi_j else []
    t_autre = max(0.0, moyenne(reelles) - t_entr - t_val) if reelles and t_entr else 0.0
    t_orig = derniere["orig"]["duree_s"] if derniere["orig"] else None
    # Temps de démarrage (une fois par pli) : du lancement à la fin de la 1re époque, moins son entraînement
    # (import, analyse du pli, mise en cache). Mesuré seulement si cette session a lancé l'entraînement.
    t_demarrage = 0.0
    if suivi_j and suivi_j.arrivees and suivi_j.debut and derniere["epoques"]:
        t1 = epoques.get(derniere["epoques"][0], {}).get("train_time_s", 0.0)
        t_demarrage = max(0.0, suivi_j.arrivees[0] - suivi_j.debut - t1)
    metriques, estimations = {}, {}
    if t_entr:
        print(f"\nRéglages effectifs (choisis par le modèle segresnet) : époques demandées {a.epoques} → boucles "
              f"{r.get('num_epochs')} ; batch {r.get('batch_size')} ; num_crops_per_image {crops} ; "
              f"num_steps_per_image {r.get('num_steps_per_image')} ; ROI {r.get('roi')} ; cache RAM {r.get('cache')} (Go requis, Go libres)")
        print(f"Durée moyenne (sans la 1re époque de chaque session) : entraînement {t_entr:.1f} s, validation {t_val:.1f} s"
              + (f", temps réel par époque {moyenne(reelles):.1f} s (sauvegardes dans Drive comprises)" if reelles else ""))
        if t_demarrage:
            print(f"Démarrage (lancement → 1re époque, hors entraînement de celle-ci) : {t_demarrage:.0f} s, compté une fois par pli")
        if len(epoques) < 3:
            print(f"ATTENTION : seulement {len(epoques)} boucle(s) mesurée(s) : estimation très fragile.")
        if t_orig:
            print(f"Validation finale à la résolution d'origine (3 volumes entiers) : {t_orig:.0f} s")
        if suivi_g and suivi_g.pic:
            print(f"Pic de mémoire GPU (nvidia-smi, carte entière) : {suivi_g.pic / 1024:.1f} Go sur {suivi_g.total / 1024:.1f} Go "
                  f"(avant le lancement : {(suivi_g.base or 0) / 1024:.1f} Go occupés par ce script)")
        print(f"\nEstimation INDICATIVE pour les {NB_PLIS} plis (L4, {a.unites_par_heure} unités/h ; validation finale comprise) :")
        for n in a.epoques_cible:
            est = estimations[n] = estimer(n, crops, t_entr, t_val, t_autre, t_orig, a.unites_par_heure, t_demarrage)
            print(f"  num_epochs = {n:>5} : {est['boucles']} boucles, {est['validations']} validations → ≈ {est['h_par_pli']:.1f} h "
                  f"par pli, ≈ {est['h_5_plis']:.1f} h et ≈ {est['unites_5_plis']:.0f} unités pour les 5 plis")
        print("  Repère nnU-Net (D25) : ≈ 8 h et ≈ 16 unités par pli, ≈ 80 unités pour les 5 plis.")
        metriques.update({"duree_entrainement_epoque_s": t_entr, "duree_validation_s": t_val,
                          "duree_reelle_epoque_s": moyenne(reelles), "validation_origine_s": t_orig,
                          "duree_demarrage_s": t_demarrage or None,
                          "pic_memoire_gpu_mo": suivi_g.pic if suivi_g else None,
                          "memoire_gpu_totale_mo": suivi_g.total if suivi_g else None,
                          "nb_epoques_terminees": len(epoques), "duree_session_s": duree,
                          "dice_validation_meilleur": derniere["fin"]["best_metric"] if derniere["fin"] else None,
                          "dice_validation_origine": derniere["orig"]["dice_moyen"] if derniere["orig"] else None})
        for n, est in estimations.items():
            metriques.update({f"estimation_{n}_epoques_h_par_pli": est["h_par_pli"],
                              f"estimation_{n}_epoques_unites_5_plis": est["unites_5_plis"]})

    # --- MLflow : même base SQLite dans Drive que nnU-Net, expérience à part ----------------------------
    series = {"train_loss": [], "duree_entrainement_s": [], "val_dice_moyen": [], "duree_validation_s": []}
    for e, x in sorted(epoques.items()):
        series["train_loss"].append((e, x["train_loss"]))
        series["duree_entrainement_s"].append((e, x["train_time_s"]))
        if "val_dice_moyen" in x:
            series["val_dice_moyen"].append((e, x["val_dice_moyen"]))
            series["duree_validation_s"].append((e, x["val_time_s"]))
            for i, d in enumerate(x["val_dice"], start=1):   # acc_0..acc_6 = étiquettes 1..7 (fond exclu)
                series.setdefault(f"val_dice_etiquette_{i}", []).append((e, d))
    params = {"algorithme": ALGO, "pli": PLI, "num_epochs": a.epoques, "monai": monai.__version__,
              "torch": torch.__version__, "modeles_auto3dseg": HASH_MODELES, "sha256_modeles": SHA256_MODELES,
              "gpu": gpu, "val_pli": ",".join(plis[PLI]["val"]), "sha256_liste": fiche05k["sha256_liste"],
              "sha256_splits_final": fiche05k["sha256_splits_final"], "batch_size": r.get("batch_size"),
              "num_crops_per_image": crops, "num_steps_per_image": r.get("num_steps_per_image"),
              "roi_size": r.get("roi"), "resample_resolution": hp.get("resample_resolution"),
              "commande": " ".join(sys.argv)}
    termine = etat_essai(dossier_modele, j) == "termine"
    statut = "interrompu" if interrompu else ("echec" if code else ("pli_non_confirme" if usage else
                                                                    ("termine" if termine else "en_cours")))
    tags = {"etape": "5", "type": "essai_a_blanc", "decision": "D37", "statut": statut,
            "mlflow.source.git.commit": prep.commit_git()}
    artefacts = [chemin_log] + [os.path.join(dossier_modele, f) for f in ("accuracy_history.csv", "progress.yaml")] + \
        [os.path.join(travail_essai, NOM_ALGO, "configs", "hyper_parameters.yaml"), os.path.join(travail_essai, "datastats.yaml"),
         chemin_liste, os.path.join(a.travail, prep05k.NOM_FICHE), chemin_fiche]
    essai.EXPERIENCE_MLFLOW = EXPERIENCE_MLFLOW      # tracer_mlflow (05b) lit ce nom au moment de l'appel
    try:
        id_run = essai.tracer_mlflow(a.mlflow, f"essai_{ALGO}_pli{PLI}", {k: v for k, v in params.items() if v is not None},
                                     tags, series, metriques, artefacts, id_run=suivi05c.lire_id_run(travail_essai))
        suivi05c.ecrire_id_run(travail_essai, id_run)
        print(f"\nEssai tracé dans MLflow : {a.mlflow} (expérience {EXPERIENCE_MLFLOW}, run {id_run}, statut {statut})")
    except Exception as err:
        print(f"\nATTENTION : traçage MLflow impossible ({err}). Relancer avec --tracer-seulement.")

    # --- Verdict ------------------------------------------------------------------------------------------
    if usage:
        print("Résultat : NON CONFIRMÉ, le pli 0 n'a peut-être pas utilisé nos plis. Ne pas utiliser ces mesures :")
        for u in usage:
            print("  -", u)
        sys.exit(6)
    if code or interrompu:
        print("Résultat : entraînement INTERROMPU ou en erreur. Pour reprendre, relancer la même cellule "
              "(au besoin dans une nouvelle session, après les sections 1, 2 et l'installation de la section 24).")
        sys.exit(3)
    print(f"Résultat : ESSAI {'TERMINÉ' if termine else 'EN COURS'}. Le lot de test n'a été ni lu ni utilisé.")


if __name__ == "__main__":
    main()
