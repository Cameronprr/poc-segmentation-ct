"""
Étape 5m — Entraînement d'UN pli de SegResNet (Auto3DSeg, MONAI 1.6.1, modèles 21ed8e5), décision D38 :
300 époques, validation toutes les 10 époques (dernière comprise), reprise COMPLÈTE après coupure, puis
évaluation finale des 3 cas de validation du pli avec model_final.pt (poids de la dernière époque).

Ce que fait le script (même style et mêmes garde-fous que 05c et 05l, dont il réutilise les fonctions) :
  1. garde-fous : dossiers hors du dépôt et sans espace ; MONAI 1.6.1 exactement ; GPU présent (avertit si
     ce n'est pas un L4) ; liste de 05k inchangée et TOUJOURS conforme à splits_final.json (05l) ;
     verrou (une seule session à la fois par pli), posé avant toute écriture, même avant --ecraser ;
  2. chaque pli dans SON dossier neuf : <travail>/entrainement_segresnet/pli_<k>/, créé par ce script
     avec une fiche (fiche_05m.json) qui porte un identifiant unique. Un dossier existant sans fiche est
     refusé ; une fiche aux réglages différents arrête le script (code 2, --ecraser) ;
  3. copie locale des SEULS 30 fichiers de la liste, archive des modèles épinglée (SHA-256), comme 05l ;
  4. génération par AutoRunner SANS entraînement, seulement si l'entraînement du pli n'a jamais été lancé
     (MONAI efface le dossier de l'algorithme quand il le régénère : points de reprise compris). AutoRunner
     génère les plis 0 à k ; seul segresnet_<k> est gardé. Réglages écrits dans hyper_parameters.yaml et
     contrôlés : fold = k, num_epochs = 300, num_epochs_per_validation = 10, num_epochs_per_saving = 10,
     early_stopping_fraction = 0, validate_final_original_res = false, identifiant du pli ; aucune trace
     de « continue », « pretrained_ckpt_name » ou « start_epoch » ; cas de validation = les 3 du pli dans
     splits_final.json (datafold_read, la fonction que lit le modèle) ;
  5. correctif de segmenter.py (copie du pli seulement) : empreinte SHA-256 contrôlée AVANT (code
     d'origine de l'archive 21ed8e5, sinon refus) et APRÈS (code corrigé attendu) ; module
     reprise_segresnet.py copié à côté. Cinq insertions, chacune sur un texte qui doit apparaître une seule
     fois (CORRECTIFS ci-dessous) ;
  6. entraînement (nouveau ou reprise) avec la commande de MONAI, journal training.log suivi en direct
     (05l). Toutes les --suivi-toutes-les minutes : progression, durée et unités restantes estimées, MLflow ;
  7. évaluation finale (une fois l'entraînement fini) : le modèle SegResNet lui-même, en mode
     « validate », prédit les 3 cas de validation du pli avec model_final.pt, à la résolution et dans
     l'orientation d'origine (transformations inversées). Ce script contrôle chaque prédiction (forme,
     affine, étiquettes 0 à 7), la réécrit en mmwhs_XXXX.nii.gz (entiers, en-tête de l'image d'origine),
     calcule le Dice par structure avec MONAI et le compare à celui du modèle. AUCUN post-traitement ;
  8. MLflow : même base que 05b/05c/05l, expérience de l'essai (« etape5_auto3dseg_segresnet », comme nnU-Net
     qui range essai et plis dans « etape5_nnunet »), un run par pli (tag type = entrainement).
  Tout l'affichage est recopié dans <dossier du pli>/sortie_05m_<SESSION>.txt (Drive), heures UTC et Paris.
  Le script ne lit jamais imagesTs ni labelsTs et ne prédit QUE les 3 cas de validation du pli.

Reprise après une coupure : relancer la même cellule. Ce qui est REPRIS (point_reprise_complet.pt, écrit
  toutes les 10 époques et à la dernière, par fichier temporaire puis renommage) : poids du réseau ; état de
  l'optimiseur AdamW ; calendrier du taux d'apprentissage ; GradScaler ; meilleur Dice de validation et son
  époque ; historique des validations ; époque où reprendre. Ce qui est REFAIT : au plus les 9 époques après
  le dernier point (≈ 3 min, plus une validation éventuelle). Ce qui n'est PAS repris : l'état des tirages
  au hasard (morceaux d'image tirés, augmentations), la mise en cache des images en mémoire, les compteurs
  de temps internes du modèle. Sans effet sur la qualité attendue (l'entraînement n'est pas déterministe :
  determ = false dans le modèle).
  Le mécanisme « continue » du modèle n'est JAMAIS utilisé (il réécrirait « continue: true » dans la
  configuration, voir 05l) ; le point de reprise est refusé s'il ne porte pas l'identifiant du pli, le
  même pli, le même nombre d'époques, les mêmes batch / ROI / crops (un autre GPU peut les changer).

Choix justifiés (à valider, ligne proposée pour le journal) :
  - early_stopping_fraction = 0 (0,001 dans le modèle) : l'arrêt anticipé déciderait de la fin de
    l'entraînement d'après les 3 cas de validation du pli, ceux qui jugent ensuite le modèle (biais exclu
    par D38) ; il empêcherait aussi d'atteindre les 300 époques de D38 ;
  - validate_final_original_res = false : la validation finale du modèle porte sur model.pt (meilleur Dice
    de validation), contraire à D38 ; elle est remplacée par l'évaluation finale avec model_final.pt ;
  - num_epochs_per_saving = 10 (1 dans le modèle) : un fichier de poids fait 349 Mo et un point complet
    (poids + AdamW) 1,05 Go ; à chaque époque (21 s), ce serait plus de 400 Go écrits dans Drive par pli ;
  - la dernière époque est ajoutée au calendrier des validations : range(10, 300, 10) s'arrête à 290.

Entrées :
  --pli           pli à entraîner, OBLIGATOIRE (0 à 4)
  --raw / --preprocessed / --split / --travail / --mlflow / --local   comme 05l
  --epoques       époques du pli (défaut 300, D38 ; toute autre valeur est signalée)
  --unites-par-heure  défaut 1,94 (L4, D25)
  --suivi-toutes-les  minutes entre deux bilans d'étape (progression, estimation, MLflow), défaut 15
  --ecraser       efface le dossier du pli et repart de zéro (explicite)
  --tracer-seulement  ne lance rien : relit le journal, le bilan, et met à jour MLflow
  --ignorer-verrou    comme 05c
Sorties (dans <travail>/entrainement_segresnet/pli_<k>/, Drive) :
  fiche_05m.json (réglages, identifiant du pli) ; segresnet_<k>/ (scripts corrigés, configs, model/ :
  training.log, model_final.pt, model.pt, point_reprise_complet.pt et .json, accuracy_history.csv,
  progress.yaml) ; validation_finale_brute/ (sortie du modèle) ; predictions_validation_finale/
  mmwhs_XXXX.nii.gz (les 3 cas du pli, géométrie de l'image d'origine) ; bilan_05m.json (contrôles, Dice) ;
  sessions_05m.json (GPU, durée, pic mémoire de chaque session) ; mlflow_run_id.txt ; sortie_05m_<SESSION>.txt.
  Verrou : <travail>/entrainement_segresnet/verrou_pli_<k>.
Codes de sortie :
  0 = pli terminé (entraînement, évaluation finale conforme, MLflow) ; 1 = garde-fou (rien n'est lancé, ou
  reprise refusée, ou réglage effectif non conforme) ; 2 = le dossier du pli a d'autres réglages
  (--ecraser, après en avoir parlé) ; 3 = interrompu ou en erreur : RELANCER LA MÊME CELLULE ; 5 = pas de
  GPU ; 6 = validation ou prédictions sur d'autres cas que les 3 du pli : ne pas utiliser ; 7 = prédictions
  finales non conformes (forme, affine, étiquettes, Dice différent de celui du modèle) : ne pas utiliser.

Vocabulaire :
  - point de reprise complet : fichier qui contient TOUT l'état de l'entraînement (voir plus haut) ;
  - model_final.pt : poids après la dernière époque (le modèle évalué, D38) ; model.pt : poids au meilleur
    Dice de validation (gardé pour information, pas utilisé : choisi sur les cas qui le jugent) ;
  - évaluation hors pli : chaque cas est prédit par le modèle du pli où il n'a PAS servi à l'entraînement ;
  - géométrie d'origine : même nombre de voxels et même « affine » (position, taille et orientation des
    voxels dans l'espace) que l'image imagesTr, pour superposer voxel à voxel avec nnU-Net ;
  - transformations inversées : le modèle recadre et réoriente l'image avant de prédire ; la prédiction est
    remise dans la géométrie d'origine en défaisant ces opérations (Invertd de MONAI).
"""
import argparse, datetime, glob, importlib.util, json, math, os, re, shutil, sys, threading, time, uuid

ICI = os.path.dirname(os.path.abspath(__file__))


def _charger(nom, fichier):
    spec = importlib.util.spec_from_file_location(nom, os.path.join(ICI, fichier))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)              # aucun de ces fichiers n'exécute quoi que ce soit à l'import
    return module


l05 = _charger("essai05l", "05l_essai_segresnet.py")      # apporte 05k, 05c, 05b et 05a
rs = _charger("reprise_segresnet", "reprise_segresnet.py")
suivi05c, essai, prep, prep05k = l05.suivi05c, l05.essai, l05.prep, l05.prep05k

ALGO = l05.ALGO
NB_EPOQUES_D38 = 300
VALIDATION_TOUTES_LES = 10                        # D38
SAUVEGARDE_TOUTES_LES = 10                        # model_final.pt et point de reprise complet
OPTIONS_D38 = {"num_epochs_per_validation": VALIDATION_TOUTES_LES, "num_epochs_per_saving": SAUVEGARDE_TOUTES_LES,
               "early_stopping_fraction": 0, "validate_final_original_res": False}
DOSSIER_PLIS = "entrainement_segresnet"
NOM_FICHE, NOM_SESSIONS, NOM_BILAN = "fiche_05m.json", "sessions_05m.json", "bilan_05m.json"
DOSSIER_BRUT, DOSSIER_PRED = "validation_finale_brute", "predictions_validation_finale"
STRUCTURES = {1: "myo_VG", 2: "OG", 3: "VG", 4: "OD", 5: "VD", 6: "aorte", 7: "AP"}   # comme 05f et 05i
ETIQUETTES_AUTORISEES = set(range(8))             # fond 0 et étiquettes 1 à 7 (comme R0)
TOLERANCE_AFFINE = 1e-3                           # mm, comme 03 (TOLERANCE_AFFINE) et 05i
TOLERANCE_DICE_MODELE = 1e-3                      # le modèle affiche son Dice par cas avec 3 décimales
# segmenter.py du modèle segresnet de l'archive 21ed8e5 (SHA-256 de l'archive vérifiée par 05l), avant et
# après le correctif ci-dessous. Toute autre empreinte : refus (code 1).
SHA256_SEGMENTER_ORIGINE = "0477831782c5855966f2a4dd3c5079f1bc93b7d2e6331e6a0badde6e40b41d43"
SHA256_SEGMENTER_CORRIGE = "8ee7331bfd384773416d3ce4d400f1506459ee7bf0af1c0a037761701dc27c81"

# --- Correctif de segmenter.py : (nom, texte d'origine qui doit apparaître UNE fois, remplacement) ---------
_IMPORT_UTILS = ('if __package__ in (None, ""):\n    from utils import auto_adjust_network_settings, logger_configure\n'
                 'else:\n    from .utils import auto_adjust_network_settings, logger_configure\n')
_DEBUT_BOUCLE = "        range_num_epochs = range(start_epoch, num_epochs)\n"
_PAS_LR = "            if lr_scheduler is not None:\n                lr_scheduler.step()\n"
CORRECTIFS = (
    ("import du module de reprise", _IMPORT_UTILS, _IMPORT_UTILS +
     "# >>> correctif 05m (D38) : reprise complète et écritures atomiques (scripts/reprise_segresnet.py)\n"
     'if __package__ in (None, ""):\n    import reprise_segresnet\nelse:\n    from . import reprise_segresnet\n'
     "# <<< correctif 05m\n"),
    ("écriture atomique des poids (model.pt, model_final.pt)",
     '        torch.save({"state_dict": state_dict, "config": config, **kwargs}, ckpt)\n',
     '        reprise_segresnet.sauver_atomique({"state_dict": state_dict, "config": config, **kwargs}, ckpt)'
     "  # correctif 05m\n"),
    ("copie atomique de model.pt vers model_final.pt",
     "                    shutil.copyfile(best_ckpt_path, intermediate_ckpt_path)  # if already saved once\n",
     "                    reprise_segresnet.copier_atomique(best_ckpt_path, intermediate_ckpt_path)  # correctif 05m\n"),
    ("restauration complète avant la boucle des époques", _DEBUT_BOUCLE,
     "        # >>> correctif 05m (D38) : reprise complète ; la dernière époque est ajoutée aux validations\n"
     "        start_epoch, best_metric, best_metric_epoch, val_acc_history, val_schedule_list = reprise_segresnet.restaurer(\n"
     "            self, optimizer, lr_scheduler, grad_scaler, num_epochs=num_epochs, num_crops_per_image=num_crops_per_image,\n"
     "            start_epoch=start_epoch, best_metric=best_metric, best_metric_epoch=best_metric_epoch,\n"
     "            val_acc_history=val_acc_history, val_schedule_list=val_schedule_list)\n"
     "        # <<< correctif 05m\n" + _DEBUT_BOUCLE),
    ("point de reprise complet après chaque boucle", _PAS_LR, _PAS_LR +
     "            # >>> correctif 05m (D38) : point de reprise complet (mêmes boucles que model_final.pt)\n"
     "            reprise_segresnet.sauver(\n"
     "                self, prochaine_epoque=epoch + 1, num_epochs=num_epochs, num_epochs_per_saving=num_epochs_per_saving,\n"
     "                num_crops_per_image=num_crops_per_image, optimizer=optimizer, lr_scheduler=lr_scheduler,\n"
     "                grad_scaler=grad_scaler, best_metric=best_metric, best_metric_epoch=best_metric_epoch,\n"
     "                val_acc_history=val_acc_history)\n"
     "            # <<< correctif 05m\n"),
)

# Lignes de training.log propres à 05m (module de reprise) et au mode « validate » du modèle.
RE_CALENDRIER = re.compile(r"^Calendrier des validations \(correctif 05m\) : \[([^\]]*)\]")
RE_REFUS = re.compile(r"^REPRISE REFUSEE \(correctif 05m\) : (.*)$")
RE_REPRISE = re.compile(r"^REPRISE COMPLETE \(correctif 05m\) : (.*)$")
RE_CHARGE = re.compile(r"^=> loaded checkpoint (\S+) \(epoch (-?\d+)\)")
RE_NB_VAL = re.compile(r"^validation files (\d+)")
RE_VAL_CAS = re.compile(r"^Val -?\d+/-?\d+ (\d+)/(\d+)\s+loss: \S+ acc \[([^\]]*)\]\s+time \S+ (.*)$")
RE_VAL_FIN = re.compile(r"^Validation complete, loss_avg: \S+ acc_avg: (\S+) acc \[([^\]]*)\]")
RE_CAS = re.compile(r"(mmwhs_\d{4})_0000\.nii(?:\.gz)?")


# --- Fonctions pures (testables sans GPU ni données) --------------------------------------------
def sha256_texte(texte):
    import hashlib
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()


def corriger_segmenter(texte):
    """Applique CORRECTIFS à un texte de segmenter.py. Lève ValueError si un texte d'origine n'apparaît pas
    exactement une fois (code différent de celui pour lequel le correctif a été écrit)."""
    for nom, ancien, nouveau in CORRECTIFS:
        n = texte.count(ancien)
        if n != 1:
            raise ValueError(f"correctif « {nom} » : texte d'origine trouvé {n} fois au lieu d'une")
        texte = texte.replace(ancien, nouveau)
    return texte


def nombres(texte):
    """« [ 0.912  nan 0.850] » → [0.912, nan, 0.85] (format numpy du modèle)."""
    return [float(x) for x in texte.replace(",", " ").split()]


def calendrier_attendu(nb_boucles, crops):
    """Validations que doit faire le modèle corrigé : celles de schedule_validation_epochs (copie 05l) avec
    num_epochs_per_validation = 10 // crops (segmenter.py l. 1221), plus la dernière boucle (module de reprise)."""
    par = max(1, VALIDATION_TOUTES_LES // max(1, crops))
    return rs.calendrier_avec_derniere(l05.calendrier_validation(nb_boucles, par), nb_boucles)


def lire_lignes_05m(texte):
    """Lignes du module de reprise : calendriers, reprises, refus (une entrée par lancement)."""
    r = {"calendriers": [], "reprises": [], "refus": []}
    for ligne in texte.splitlines():
        if (m := RE_CALENDRIER.match(ligne)):
            r["calendriers"].append([int(x) for x in m.group(1).replace(",", " ").split()])
        elif (m := RE_REPRISE.match(ligne)):
            r["reprises"].append(m.group(1))
        elif (m := RE_REFUS.match(ligne)):
            r["refus"].append(m.group(1))
    return r


def lr_par_epoque(texte):
    """{époque: taux d'apprentissage} lu sur les lignes « Final training » (la dernière valeur l'emporte)."""
    return {int(m.group(1)): float(m.group(6)) for l in texte.splitlines() if (m := l05.RE_ENTR.match(l))}


def lire_evaluation(texte):
    """Partie de training.log écrite par le mode « validate » : point chargé, nombre de cas, Dice par cas
    (7 valeurs, étiquettes 1 à 7) et Dice moyen calculés par le modèle."""
    r = {"charge": None, "nb_fichiers": None, "par_cas": {}, "moyenne": None, "autres_fichiers": []}
    for ligne in texte.splitlines():
        if (m := RE_CHARGE.match(ligne)):
            r["charge"] = (m.group(1), int(m.group(2)))
        elif (m := RE_NB_VAL.match(ligne)):
            r["nb_fichiers"] = int(m.group(1))
        elif (m := RE_VAL_CAS.match(ligne)):
            cas = RE_CAS.findall(m.group(4))
            if len(cas) == 1:
                r["par_cas"][cas[0]] = nombres(m.group(3))
            else:
                r["autres_fichiers"].append(m.group(4))
        elif (m := RE_VAL_FIN.match(ligne)):
            r["moyenne"] = float(m.group(1))
    return r


def egaux(a, b, tol):
    """Comparaison de deux Dice (None ou NaN = non défini)."""
    na, nb = a is None or a != a, b is None or b != b
    return (na and nb) or (not na and not nb and abs(a - b) <= tol)


def estimer_reste(faites, total, duree_par_epoque_s, unites_par_heure):
    """(heures, unités) restantes pour total − faites époques, ou (None, None) si la durée est inconnue."""
    if not duree_par_epoque_s:
        return None, None
    h = max(0, total - faites) * duree_par_epoque_s / 3600
    return h, h * unites_par_heure


def controler_config(hp, pli, identifiant):
    """Contrôles de hyper_parameters.yaml propres à 05m (en plus de controler_generation de 05l) : le pli ne
    doit jamais repartir d'un autre point par le mécanisme du modèle (« continue », point préentraîné)."""
    pb = []
    if hp.get("continue"):
        pb.append("hyper_parameters.yaml contient « continue: true » (reprise par le mécanisme du modèle)")
    for cle in ("pretrained_ckpt_name", "start_epoch"):
        if hp.get(cle) not in (None, 0):
            pb.append(f"hyper_parameters.yaml : {cle} = {hp.get(cle)!r} (attendu absent)")
    if (hp.get("finetune") or {}).get("enabled"):
        pb.append("hyper_parameters.yaml : finetune activé")
    if hp.get("identifiant_05m") != identifiant or hp.get("fold") != pli:
        pb.append(f"hyper_parameters.yaml : identifiant {hp.get('identifiant_05m')} / fold {hp.get('fold')} ≠ "
                  f"ceux du pli ({identifiant} / {pli})")
    return pb


def etat_du_pli(meta):
    """« neuf » (aucun point de reprise complet), « reprise » ou « entraine » (dernière boucle faite)."""
    if meta is None:
        return "neuf"
    return "entraine" if int(meta["prochaine_epoque"]) >= int(meta["num_epochs"]) else "reprise"


# --- Fichiers ---------------------------------------------------------------------------------
def lire_json(chemin, defaut=None):
    try:
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return defaut


def lire_meta_point(dossier_modele):
    """Résumé (sans tenseurs) du point de reprise complet, lu dans le .pt lui-même (il fait foi), ou None."""
    chemin = rs.chemin_point(dossier_modele)
    if not os.path.isfile(chemin):
        return None
    import torch
    try:
        etat = torch.load(chemin, map_location="cpu", weights_only=False, mmap=True)   # lecture paresseuse
    except Exception:
        etat = torch.load(chemin, map_location="cpu", weights_only=False)
    return dict(etat.get("meta", {}), format=etat.get("format"))


def lire_depuis(chemin, position=0):
    if not os.path.isfile(chemin):
        return ""
    with open(chemin, encoding="utf-8", errors="replace") as f:      # fichier refermé aussitôt
        f.seek(position)
        return f.read()


def installer_correctif(dossier_algo, module_depot):
    """segmenter.py du pli : d'origine → corrigé (contrôle avant/après) ; module de reprise copié à côté.
    Renvoie la liste des problèmes (vide = conforme)."""
    seg = os.path.join(dossier_algo, "scripts", "segmenter.py")
    module = os.path.join(dossier_algo, "scripts", os.path.basename(module_depot))
    if not os.path.isfile(seg):
        return [f"{seg} absent"]
    empreinte = prep.sha256_fichier(seg)
    if empreinte == SHA256_SEGMENTER_ORIGINE:
        with open(seg, "rb") as f:
            corrige = corriger_segmenter(f.read().decode("utf-8")).encode("utf-8")
        if sha256_texte(corrige.decode("utf-8")) != SHA256_SEGMENTER_CORRIGE:
            return ["le correctif appliqué ne donne pas le texte attendu (empreinte) : rien n'est modifié"]
        tmp = seg + ".tmp"
        with open(tmp, "wb") as f:
            f.write(corrige)
        os.replace(tmp, seg)
        print(f"segmenter.py du pli : code d'origine 21ed8e5 reconnu (SHA-256 {SHA256_SEGMENTER_ORIGINE[:12]}…), "
              f"correctif 05m appliqué ({len(CORRECTIFS)} insertions)")
    elif empreinte == SHA256_SEGMENTER_CORRIGE:
        print("segmenter.py du pli : correctif 05m déjà en place (empreinte attendue)")
    else:
        return [f"segmenter.py du pli : empreinte {empreinte} inconnue (ni l'original 21ed8e5 ni la version "
                "corrigée attendue) : refus de lancer"]
    if not os.path.isfile(module) or prep.sha256_fichier(module) != prep.sha256_fichier(module_depot):
        rs.copier_atomique(module_depot, module)
    pb = []
    if prep.sha256_fichier(seg) != SHA256_SEGMENTER_CORRIGE:
        pb.append("segmenter.py du pli : empreinte après correctif inattendue")
    if prep.sha256_fichier(module) != prep.sha256_fichier(module_depot):
        pb.append("module de reprise du pli ≠ scripts/reprise_segresnet.py du dépôt")
    return pb


def supprimer_plis_inutiles(dossier_pli, pli):
    """AutoRunner génère segresnet_0 à segresnet_<pli> ; on ne garde que segresnet_<pli>. Les autres n'ont
    jamais été lancés (pas de dossier model/) : sinon, refus."""
    for j in range(pli):
        d = os.path.join(dossier_pli, f"{ALGO}_{j}")
        if os.path.isdir(d):
            if os.path.isdir(os.path.join(d, "model")):
                sys.exit(f"ARRÊT : {d} contient un dossier model/ : il ne devrait pas exister dans ce pli.")
            shutil.rmtree(d)


# --- Évaluation finale : prédictions dans la géométrie d'origine, Dice MONAI ---------------------
def controler_prediction(chemin_image, chemin_pred):
    """(problèmes, données entières 3D, image nibabel) : même forme, même affine (TOLERANCE_AFFINE),
    étiquettes entières de 0 à 7."""
    import nibabel as nib
    import numpy as np
    img, pred = nib.load(chemin_image), nib.load(chemin_pred)
    donnees = np.asanyarray(pred.dataobj)
    while donnees.ndim > 3 and donnees.shape[-1] == 1:              # dimensions finales de taille 1
        donnees = donnees[..., 0]
    pb = []
    if donnees.shape != img.shape[:3]:
        return [f"forme {donnees.shape} ≠ image {img.shape[:3]}"], None, img
    ecart = float(np.abs(pred.affine - img.affine).max())
    if ecart > TOLERANCE_AFFINE:
        pb.append(f"affine différente de l'image (écart max {ecart:.5f} > {TOLERANCE_AFFINE})")
    if not np.issubdtype(donnees.dtype, np.integer) and not np.array_equal(donnees, np.round(donnees)):
        pb.append(f"valeurs non entières (type {donnees.dtype})")
    valeurs = set(np.unique(donnees).astype(int).tolist())
    if not valeurs <= ETIQUETTES_AUTORISEES:
        pb.append(f"étiquettes hors de 0 à 7 : {sorted(valeurs - ETIQUETTES_AUTORISEES)}")
    return pb, donnees.astype(np.uint8), img


def ecrire_prediction(donnees, img, chemin):
    """Écrit le masque en entiers 8 bits avec l'en-tête et l'affine de l'image d'origine (fichier temporaire
    puis renommage). Pas de facteur d'échelle : la valeur lue est l'étiquette."""
    import nibabel as nib
    import numpy as np
    entete = img.header.copy()
    entete.set_data_dtype(np.uint8)
    entete.set_slope_inter(1, 0)
    tmp = chemin[:-len(".nii.gz")] + f".tmp{os.getpid()}.nii.gz"
    nib.save(nib.Nifti1Image(donnees.astype(np.uint8), img.affine, entete), tmp)
    os.replace(tmp, chemin)


def dice_par_structure(pred, verite):
    """Dice de chaque structure 1 à 7 avec MONAI (compute_dice ; non défini (None) si la structure est
    absente de la vérité, comme le Dice du modèle et celui de nnU-Net)."""
    import numpy as np
    import torch
    from monai.metrics import compute_dice
    r = {}
    for e in STRUCTURES:
        p = torch.from_numpy((pred == e)[None, None].astype(np.float32))
        v = torch.from_numpy((verite == e)[None, None].astype(np.float32))
        d = float(compute_dice(p, v, include_background=True, ignore_empty=True)[0, 0])
        r[e] = None if d != d else d
    return r


def evaluer(algo, a, pli, val_pli, dossier_pli, dossier_modele, nb_boucles, crops):
    """Évaluation finale du pli avec model_final.pt. Renvoie (bilan, code, problèmes, suivi GPU, durée)."""
    import nibabel as nib
    import numpy as np
    chemin_log = os.path.join(dossier_modele, "training.log")
    final = os.path.join(dossier_modele, "model_final.pt")
    attendue = (nb_boucles - 1) * crops
    ep = l05.epoque_du_point(final)
    if ep != attendue:
        return None, 1, [f"model_final.pt contient l'époque {ep}, pas la dernière ({attendue})"], None, 0
    brut, sortie = os.path.join(dossier_pli, DOSSIER_BRUT), os.path.join(dossier_pli, DOSSIER_PRED)
    for d in (brut, sortie):                      # évaluation refaite entièrement si elle avait été coupée
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
    params = {"validate#enabled": True, "validate#ckpt_name": final, "validate#save_mask": True,
              "validate#invert": True, "validate#output_path": brut}
    print(f"\nÉVALUATION FINALE du pli {pli} avec model_final.pt (époque {ep}, la dernière) sur ses 3 cas de "
          f"validation {val_pli} : prédiction par le modèle lui-même (mode validate), puis contrôles. "
          f"La ligne « Lancement de l'entraînement » ci-dessous lance cette prédiction : aucun entraînement.", flush=True)
    position = os.path.getsize(chemin_log) if os.path.isfile(chemin_log) else 0
    code, interrompu, _, suivi_g, duree = l05.lancer_entrainement(algo, params, dossier_modele, dossier_pli)
    if code or interrompu:
        return None, 3, [f"évaluation finale {'interrompue' if interrompu else 'en erreur'} (code {code})"], suivi_g, duree
    ev = lire_evaluation(lire_depuis(chemin_log, position))
    pb_cas, pb = [], []
    if ev["charge"] is None or os.path.basename(ev["charge"][0]) != "model_final.pt" or ev["charge"][1] != attendue:
        pb.append(f"le modèle n'a pas confirmé avoir chargé model_final.pt de l'époque {attendue} (lu : {ev['charge']})")
    fichiers = {prep05k._cas_du_chemin(p): p for p in glob.glob(os.path.join(brut, "**", "*.nii.gz"), recursive=True)}
    if ev["nb_fichiers"] != len(val_pli) or set(ev["par_cas"]) != set(val_pli) or ev["autres_fichiers"]:
        pb_cas.append(f"le modèle a validé {ev['nb_fichiers']} cas {sorted(ev['par_cas'])} {ev['autres_fichiers']} "
                      f"au lieu de {val_pli}")
    if set(fichiers) != set(val_pli):
        pb_cas.append(f"prédictions écrites pour {sorted(fichiers)} au lieu de {val_pli}")
    if pb_cas:
        return None, 6, pb_cas, suivi_g, duree
    bilan = {"pli": pli, "cas": {}, "modele_evalue": "model_final.pt", "epoque_du_modele": ep,
             "sha256_model_final": prep.sha256_fichier(final), "post_traitement": "aucun",
             "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
             "commit": prep.commit_git(), "duree_evaluation_s": duree, "dice_moyen_modele": ev["moyenne"]}
    for c in sorted(val_pli):
        image = os.path.join(a.local, *prep05k.chemins_cas(c)[0].split("/"))
        verite = os.path.join(a.local, *prep05k.chemins_cas(c)[1].split("/"))
        p, donnees, img = controler_prediction(image, fichiers[c])
        if donnees is None:
            pb += [f"{c} : {x}" for x in p]
            continue
        cible = os.path.join(sortie, f"{c}.nii.gz")              # même nom que les prédictions de nnU-Net
        ecrire_prediction(donnees, img, cible)
        relu = nib.load(cible)                                   # contrôle de ce qui est réellement écrit
        d_relu = np.asanyarray(relu.dataobj)
        ecart = float(np.abs(relu.affine - img.affine).max())
        if relu.shape != img.shape[:3] or ecart > TOLERANCE_AFFINE or d_relu.dtype != np.uint8 \
                or not np.array_equal(d_relu, donnees):
            p.append(f"fichier réécrit non conforme (forme {relu.shape}, écart d'affine {ecart}, type {d_relu.dtype})")
        dice = dice_par_structure(donnees, np.asanyarray(nib.load(verite).dataobj))
        modele = dict(zip(STRUCTURES, ev["par_cas"][c]))
        ecarts = [STRUCTURES[e] for e in STRUCTURES if not egaux(dice[e], modele.get(e), TOLERANCE_DICE_MODELE)]
        if ecarts:
            p.append(f"Dice différent de celui du modèle (> {TOLERANCE_DICE_MODELE}) pour {ecarts} : géométrie suspecte")
        pb += [f"{c} : {x}" for x in p]
        definis = [d for d in dice.values() if d is not None]
        bilan["cas"][c] = {"dice": {STRUCTURES[e]: d for e, d in dice.items()},
                           "dice_modele": {STRUCTURES[e]: modele.get(e) for e in STRUCTURES},
                           "dice_moyen": sum(definis) / len(definis) if definis else None,
                           "etiquettes_presentes": sorted(set(np.unique(donnees).tolist())),
                           "forme": list(donnees.shape), "ecart_affine_max": ecart,
                           "fichier": os.path.relpath(cible, dossier_pli), "sha256": prep.sha256_fichier(cible)}
    par_structure = {}
    for nom in STRUCTURES.values():
        v = [x["dice"][nom] for x in bilan["cas"].values() if x["dice"][nom] is not None]
        par_structure[nom] = sum(v) / len(v) if v else None
    tous = [d for x in bilan["cas"].values() for d in x["dice"].values() if d is not None]
    bilan.update({"dice_moyen_par_structure": par_structure, "dice_moyen": sum(tous) / len(tous) if tous else None,
                  "problemes": pb, "conforme": not pb and len(bilan["cas"]) == len(val_pli)})
    rs.ecrire_json_atomique(bilan, os.path.join(dossier_pli, NOM_BILAN))
    return bilan, (0 if bilan["conforme"] else 7), pb, suivi_g, duree


def afficher_bilan(bilan):
    cas = sorted(bilan["cas"])
    print(f"\nDice HORS PLI avec model_final.pt (époque {bilan['epoque_du_modele']}), sans post-traitement, "
          f"3 cas de validation du pli {bilan['pli']} (PAS le test) :")
    print("cas        | " + " | ".join(f"{n:>6}" for n in STRUCTURES.values()) + " | moyenne")
    f = lambda d: f"{d:6.3f}" if d is not None else "     —"
    for c in cas:
        x = bilan["cas"][c]
        print(f"{c} | " + " | ".join(f(x["dice"][n]) for n in STRUCTURES.values()) + f" | {f(x['dice_moyen'])}")
    print("moyenne    | " + " | ".join(f(bilan["dice_moyen_par_structure"][n]) for n in STRUCTURES.values())
          + f" | {f(bilan['dice_moyen'])}")


# --- Suivi pendant l'entraînement ----------------------------------------------------------------
class SuiviPli(threading.Thread):
    """Toutes les `periode_s` secondes : appelle `fonction` (progression, estimation, MLflow). Une erreur
    est affichée sans jamais arrêter l'entraînement."""
    def __init__(self, periode_s, fonction):
        super().__init__(daemon=True)
        self.periode_s, self.fonction, self.arret = periode_s, fonction, threading.Event()

    def run(self):
        while not self.arret.wait(self.periode_s):
            try:
                self.fonction()
            except Exception as err:
                print(f"[{suivi05c.heure()}] ATTENTION : suivi non mis à jour ({type(err).__name__} : {err}) ; "
                      "l'entraînement continue.", flush=True)


def lire_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--pli", type=int, required=True, choices=range(prep.NB_PLIS))
    p.add_argument("--raw", default=f"{prep.DRIVE}/nnunet_raw")
    p.add_argument("--preprocessed", default=f"{prep.DRIVE}/nnunet_preprocessed")
    p.add_argument("--split", default=os.path.join(prep.DEPOT, "splits", "mmwhs_split.json"))
    p.add_argument("--travail", default=f"{prep.DRIVE}/auto3dseg")
    p.add_argument("--mlflow", default=f"{prep.DRIVE}/mlflow")
    p.add_argument("--local", default="/content/auto3dseg_donnees_local")
    p.add_argument("--id-dataset", type=int, default=101)
    p.add_argument("--epoques", type=int, default=NB_EPOQUES_D38)
    p.add_argument("--unites-par-heure", type=float, default=1.94)
    p.add_argument("--suivi-toutes-les", type=float, default=15.0)
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
    k, nom_ds = a.pli, f"Dataset{a.id_dataset:03d}_MMWHS"
    racine = os.path.join(a.travail, DOSSIER_PLIS)
    dossier_pli = os.path.join(racine, f"pli_{k}")
    nom_algo = f"{ALGO}_{k}"
    dossier_algo = os.path.join(dossier_pli, nom_algo)
    dossier_modele = os.path.join(dossier_algo, "model")
    chemin_log = os.path.join(dossier_modele, "training.log")
    module_depot = os.path.join(ICI, "reprise_segresnet.py")
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
    if a.epoques != NB_EPOQUES_D38:
        print(f"ATTENTION : --epoques {a.epoques} ≠ {NB_EPOQUES_D38} fixées par D38. Ce pli ne sera pas comparable aux autres.")
    if not a.tracer_seulement:                    # verrou AVANT toute écriture (y compris --ecraser)
        os.makedirs(racine, exist_ok=True)
        l05.Verrou(os.path.join(racine, f"verrou_pli_{k}"), a.ignorer_verrou)
    if a.ecraser and os.path.isdir(dossier_pli):
        shutil.rmtree(dossier_pli)                # --ecraser demandé explicitement
        print(f"Pli précédent effacé : {dossier_pli}")
    import monai
    if monai.__version__ != l05.VERSION_MONAI:
        sys.exit(f"ARRÊT : MONAI {monai.__version__} installé, {l05.VERSION_MONAI} attendu (notebook, section 25, installation).")
    chemin_liste, fiche05k, liste, plis = l05.verifier_liste(a, nom_ds, pli=k)
    cas = [prep05k._cas_du_chemin(e["image"]) for e in liste["training"]]
    val_pli = sorted(plis[k]["val"])

    # --- 2. Dossier du pli : neuf, ou créé par 05m avec les mêmes réglages ----------------------------
    reglages = {"monai": l05.VERSION_MONAI, "modeles": l05.HASH_MODELES, "sha256_modeles": l05.SHA256_MODELES,
                "pli": k, "num_epochs": a.epoques, **OPTIONS_D38,
                "sha256_segmenter_origine": SHA256_SEGMENTER_ORIGINE, "sha256_segmenter_corrige": SHA256_SEGMENTER_CORRIGE,
                "sha256_module_reprise": prep.sha256_fichier(module_depot),
                "sha256_liste": fiche05k["sha256_liste"], "sha256_splits_final": fiche05k["sha256_splits_final"],
                "dataroot": a.local}
    chemin_fiche = os.path.join(dossier_pli, NOM_FICHE)
    fiche = lire_json(chemin_fiche)
    if fiche is not None:
        if fiche.get("reglages") != reglages:
            ecarts = {c: (fiche.get("reglages", {}).get(c), v) for c, v in reglages.items()
                      if fiche.get("reglages", {}).get(c) != v}
            print(f"ARRÊT : le pli {k} existe dans {dossier_pli} avec d'autres réglages (ancien, demandé) : {ecarts}.")
            print("  Relancer avec --ecraser pour l'effacer et recommencer (après en avoir parlé).")
            sys.exit(2)
        identifiant = fiche["identifiant_05m"]
    elif a.tracer_seulement:
        sys.exit(f"ARRÊT : aucun entraînement 05m du pli {k} dans {dossier_pli} (fiche absente).")
    else:
        if os.path.isdir(dossier_pli) and os.listdir(dossier_pli):
            sys.exit(f"ARRÊT : {dossier_pli} existe et n'a pas été créé par 05m (pas de fiche). Un pli doit partir "
                     "d'un dossier NEUF : vérifier son contenu, puis relancer avec --ecraser.")
        identifiant = uuid.uuid4().hex
        os.makedirs(dossier_pli, exist_ok=True)
        rs.ecrire_json_atomique({"reglages": reglages, "identifiant_05m": identifiant, "val_pli": val_pli,
                                 "cree_le_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                                 "commit": prep.commit_git(), "decision": "D38"}, chemin_fiche)
        print(f"Dossier NEUF du pli {k} : {dossier_pli} (identifiant {identifiant})")
    journal_sortie.ouvrir(os.path.join(dossier_pli, f"sortie_05m_{suivi05c.SESSION}.txt"))
    print(f"Copie de cet affichage : {journal_sortie.chemin}")

    import torch
    gpu = "non utilisé"
    if not a.tracer_seulement:
        if not torch.cuda.is_available():
            print("ARRÊT : pas de GPU visible (Exécution → Modifier le type d'exécution → GPU L4).")
            sys.exit(5)
        gpu = torch.cuda.get_device_name(0)
        if "L4" not in gpu:
            print(f"ATTENTION : GPU {gpu}, pas un L4 : durées et coût estimés ne valent que pour le L4 ; une reprise sur "
                  "un autre GPU que celui du début du pli peut être refusée (batch / ROI différents).")
        print(f"GPU : {gpu} ; MONAI {monai.__version__} ; torch {torch.__version__}")
        # --- 3. Copie locale des 30 fichiers, modèles d'algorithmes épinglés ------------------------
        l05.copier_donnees(os.path.join(a.raw, nom_ds), liste, a.local)
        modeles = l05.preparer_modeles(a.travail, local_modeles)
        # --- 4. Génération, seulement si l'entraînement n'a jamais été lancé --------------------------
        if not os.path.isdir(dossier_modele):
            os.makedirs(local_mlflow, exist_ok=True)
            uri_interne = "sqlite:///" + os.path.join(local_mlflow, "mlflow.db").replace("\\", "/")
            t0 = time.time()
            os.chdir(dossier_pli)               # MONAI écrit runner.log... dans le dossier courant
            l05.generer(dossier_pli, chemin_liste, a.local, modeles, a.epoques, uri_interne, nb_plis=k + 1,
                        options={**OPTIONS_D38, "identifiant_05m": identifiant})
            supprimer_plis_inutiles(dossier_pli, k)
            print(f"Analyse des données et génération : {time.time() - t0:.0f} s (moins si déjà faites)")
        else:
            print("Entraînement déjà lancé dans ce pli : génération NON relancée (MONAI effacerait le dossier "
                  f"{nom_algo}, points de reprise compris).")
    algo, hp, problemes = l05.controler_generation(dossier_pli, a.local, chemin_liste, a.epoques, cas, pli=k,
                                                   attendus_en_plus={**OPTIONS_D38, "identifiant_05m": identifiant})
    problemes += l05.controler_pli(os.path.join(dossier_pli, os.path.basename(chemin_liste)), a.local, plis, pli=k)
    problemes += controler_config(hp, k, identifiant)
    if algo is not None and (int(algo.device_setting.get("n_devices", 1)) > 1 or int(algo.device_setting.get("NUM_NODES", 1)) > 1):
        problemes.append(f"plusieurs GPU ou nœuds dans device_setting {algo.device_setting} : non prévu")
    # --- 5. Correctif de segmenter.py (contrôlé avant / après) -----------------------------------------
    if not a.tracer_seulement and not problemes:
        problemes += installer_correctif(dossier_algo, module_depot)
    if problemes:
        sys.exit("ARRÊT : pli non conforme (rien n'est lancé) :\n  - " + "\n  - ".join(problemes))
    print(f"Génération conforme : un seul algorithme ({nom_algo}), fold = {k}, num_epochs = {a.epoques}, validation "
          f"toutes les {VALIDATION_TOUTES_LES} époques, arrêt anticipé désactivé, validation du pli {k} = {val_pli}")

    # --- 6. Où en est le pli ? ------------------------------------------------------------------
    meta = lire_meta_point(dossier_modele)
    if meta is not None:
        ecarts = {c: (meta.get(c), v) for c, v in (("identifiant_05m", identifiant), ("fold", k),
                                                   ("num_epochs_config", a.epoques), ("format", rs.FORMAT)) if meta.get(c) != v}
        if ecarts:
            sys.exit(f"ARRÊT : le point de reprise de {dossier_modele} n'appartient pas à ce pli (point, attendu) : {ecarts}.")
    etat = etat_du_pli(meta)
    resume = lire_json(os.path.join(dossier_modele, rs.NOM_RESUME))
    remplacer_depuis = None
    texte = lire_depuis(chemin_log)
    j = l05.lire_journal(texte)
    print(f"État du pli : {etat}")
    if etat == "neuf":
        if os.path.isfile(os.path.join(dossier_modele, "model_final.pt")):
            print("  model_final.pt présent sans point de reprise complet (coupure avant l'époque 10) : redémarrage à zéro.")
        print(f"NOUVEL ENTRAÎNEMENT du pli {k} : {a.epoques} époques, validation toutes les {VALIDATION_TOUTES_LES} (dernière "
              f"comprise), point de reprise complet toutes les {SAUVEGARDE_TOUTES_LES}")
    else:
        crops = int(meta["num_crops_per_image"])
        print(f"{'REPRISE COMPLÈTE' if etat == 'reprise' else 'ENTRAÎNEMENT FINI'} depuis {rs.NOM_POINT} (écrit le "
              f"{meta.get('date_utc')} UTC) : époques 0 à {meta['derniere_epoque_faite']} faites"
              + (f", reprise à l'époque {int(meta['prochaine_epoque']) * crops} sur {a.epoques}" if etat == "reprise" else ""))
        print(f"  taux d'apprentissage de l'époque suivante {meta['lr_prochaine_epoque']:.4e} ; meilleur Dice de validation "
              f"{meta['best_metric']:.4f} (époque {meta['best_metric_epoch']}) ; {meta['nb_validations']} validations faites ; "
              "état de l'optimiseur AdamW, du calendrier du taux d'apprentissage et du GradScaler repris")
        if resume and int(resume.get("prochaine_epoque", 0)) > int(meta["prochaine_epoque"]):
            remplacer_depuis = int(meta["prochaine_epoque"]) * crops   # Drive avait recopié le résumé, pas le point
            print(f"ATTENTION : le résumé indiquait {resume['prochaine_epoque']} boucles faites, le point {meta['prochaine_epoque']} : "
                  f"les époques à partir de {remplacer_depuis} seront refaites ; MLflow recevra leurs nouvelles valeurs.")
        if etat == "reprise":
            ep = j["epoques"]
            t_e = l05.moyenne([x["train_time_s"] for x in ep.values()])
            t_v = l05.moyenne([x["val_time_s"] for x in ep.values() if "val_time_s" in x]) or 0
            nb_b, fait = int(meta["num_epochs"]), int(meta["prochaine_epoque"])
            n_val = len(rs.validations_restantes(calendrier_attendu(nb_b, crops), fait))
            if t_e:
                h = ((nb_b - fait) * t_e + n_val * t_v) / 3600
                print(f"  Reste {nb_b - fait} boucles et {n_val} validations : au moins ≈ {h:.1f} h et ≈ "
                      f"{h * a.unites_par_heure:.1f} unités (sans écritures dans Drive, démarrage ni évaluation finale)")

    # --- Traçage MLflow (fonction appelée pendant et après l'entraînement) ------------------------------
    sessions_chemin = os.path.join(dossier_pli, NOM_SESSIONS)

    def tracer(statut, metriques_en_plus=None, afficher=True):
        texte_ = lire_depuis(chemin_log)
        j_ = l05.lire_journal(texte_)
        ep = j_["epoques"]
        res = lire_json(os.path.join(dossier_modele, rs.NOM_RESUME)) or {}
        crops_ = int(res.get("num_crops_per_image", 1))
        borne = int(res.get("prochaine_epoque", 0)) * crops_       # époques couvertes par un point : jamais refaites
        lr = lr_par_epoque(texte_)
        series = {"train_loss": [], "duree_entrainement_s": [], "val_dice_moyen": [], "duree_validation_s": [], "lr": []}
        for e, x in sorted(ep.items()):
            if e >= borne:
                continue
            series["train_loss"].append((e, x["train_loss"]))
            series["duree_entrainement_s"].append((e, x["train_time_s"]))
            if e in lr:
                series["lr"].append((e, lr[e]))
            if "val_dice_moyen" in x:
                series["val_dice_moyen"].append((e, x["val_dice_moyen"]))
                series["duree_validation_s"].append((e, x["val_time_s"]))
                for i, d in enumerate(x["val_dice"], start=1):
                    series.setdefault(f"val_dice_{STRUCTURES.get(i, i)}", []).append((e, d))
        sessions = lire_json(sessions_chemin, [])
        r = (j_["sessions"][-1]["reglages"] if j_["sessions"] else {})
        derniere = max(ep) if ep else None
        metriques = {"epoques_terminees": len(ep), "epoques_consolidees": borne,
                     "duree_entrainement_epoque_s": l05.moyenne([x["train_time_s"] for x in ep.values()]),
                     "duree_validation_s": l05.moyenne([x["val_time_s"] for x in ep.values() if "val_time_s" in x]),
                     "duree_sessions_h": sum(s.get("duree_s") or 0 for s in sessions) / 3600 if sessions else None,
                     "pic_memoire_gpu_mo": max([s["pic_mo"] for s in sessions if s.get("pic_mo")], default=None),
                     "meilleur_dice_validation": res.get("best_metric"), "epoque_meilleur_dice": res.get("best_metric_epoch"),
                     "dice_validation_derniere_epoque": ep[derniere].get("val_dice_moyen") if derniere is not None else None}
        metriques.update(metriques_en_plus or {})
        params = {"algorithme": ALGO, "pli": k, "num_epochs": a.epoques, "num_epochs_per_validation": VALIDATION_TOUTES_LES,
                  "num_epochs_per_saving": SAUVEGARDE_TOUTES_LES, "early_stopping_fraction": 0,
                  "validate_final_original_res": False, "modele_evalue": "model_final.pt", "post_traitement": "aucun",
                  "monai": monai.__version__, "torch": torch.__version__, "modeles_auto3dseg": l05.HASH_MODELES,
                  "sha256_modeles": l05.SHA256_MODELES, "sha256_segmenter_corrige": SHA256_SEGMENTER_CORRIGE,
                  "sha256_module_reprise": reglages["sha256_module_reprise"], "val_pli": ",".join(val_pli),
                  "sha256_liste": fiche05k["sha256_liste"], "sha256_splits_final": fiche05k["sha256_splits_final"],
                  "identifiant_05m": identifiant, "batch_size": r.get("batch_size"),
                  "num_crops_per_image": r.get("num_crops_per_image"), "num_steps_per_image": r.get("num_steps_per_image"),
                  "roi_size": r.get("roi"), "resample": hp.get("resample"), "resample_resolution": hp.get("resample_resolution")}
        tags = {"etape": "5", "type": "entrainement", "decision": "D38", "pli": str(k), "statut": statut,
                "mlflow.source.git.commit": prep.commit_git()}
        if gpu != "non utilisé":
            tags.update({f"session_{suivi05c.SESSION}.gpu": gpu, f"session_{suivi05c.SESSION}.commande": " ".join(sys.argv)[:500]})
        artefacts = [chemin_log, chemin_fiche, sessions_chemin, os.path.join(dossier_pli, NOM_BILAN),
                     os.path.join(dossier_modele, rs.NOM_RESUME)] + \
            [os.path.join(dossier_modele, f) for f in ("accuracy_history.csv", "progress.yaml")] + \
            [os.path.join(dossier_algo, "configs", "hyper_parameters.yaml")]
        essai.EXPERIENCE_MLFLOW = l05.EXPERIENCE_MLFLOW       # tracer_mlflow (05b) lit ce nom au moment de l'appel
        try:
            id_run = essai.tracer_mlflow(a.mlflow, f"entrainement_{ALGO}_pli{k}", {c: v for c, v in params.items() if v is not None},
                                         tags, series, metriques, artefacts, id_run=suivi05c.lire_id_run(dossier_pli),
                                         remplacer_depuis=remplacer_depuis)
            suivi05c.ecrire_id_run(dossier_pli, id_run)
            if afficher:
                print(f"[{suivi05c.heure()}] MLflow mis à jour : run {id_run} (pli {k}, statut {statut}, {len(ep)} époques "
                      f"faites, {borne} notées : celles couvertes par un point de reprise)", flush=True)
        except Exception as err:
            print(f"ATTENTION : traçage MLflow impossible ({err}). Relancer plus tard avec --tracer-seulement.", flush=True)
        return j_

    # --- 7. Entraînement (nouveau ou reprise) ------------------------------------------------------
    code, interrompu, refus, anomalies = 0, False, [], []
    if not a.tracer_seulement and etat != "entraine":
        position = os.path.getsize(chemin_log) if os.path.isfile(chemin_log) else 0
        mesures = {"t0": None, "e0": None}

        def bilan_etape():
            """Progression, estimation (durée réelle par époque mesurée dans cette session), MLflow."""
            texte_ = lire_depuis(chemin_log)
            ep = l05.lire_journal(texte_)["epoques"]
            if not ep:
                return
            fait = max(ep) + 1
            if mesures["t0"] is None:
                mesures.update(t0=time.time(), e0=fait)
            reelle = (time.time() - mesures["t0"]) / (fait - mesures["e0"]) if fait - mesures["e0"] >= 2 else None
            if reelle is None:
                t_e = l05.moyenne([x["train_time_s"] for x in ep.values()]) or 0
                t_v = l05.moyenne([x["val_time_s"] for x in ep.values() if "val_time_s" in x]) or 0
                reelle = t_e + t_v / VALIDATION_TOUTES_LES or None
            h, u = estimer_reste(fait, a.epoques, reelle, a.unites_par_heure)
            vals = [(e, x["val_dice_moyen"]) for e, x in sorted(ep.items()) if "val_dice_moyen" in x]
            print(f"[{suivi05c.heure()}] Pli {k} : {fait} époques faites sur {a.epoques}"
                  + (f" ; dernière validation : Dice {vals[-1][1]:.4f} (époque {vals[-1][0]})" if vals else "")
                  + (f" ; reste ≈ {h:.1f} h et ≈ {u:.1f} unités ({a.unites_par_heure} unités/h, indicatif, évaluation finale "
                     "en plus)" if h is not None else ""), flush=True)
            tracer("en_cours", afficher=True)

        torch.cuda.empty_cache()
        suivi = SuiviPli(a.suivi_toutes_les * 60, bilan_etape)
        suivi.start()
        code, interrompu, _, suivi_g, duree = l05.lancer_entrainement(algo, {}, dossier_modele, dossier_pli)
        suivi.arret.set()
        suivi.join(timeout=120)
        print(f"\nEntraînement {'interrompu' if interrompu else 'terminé'} (code {code}) le {suivi05c.heure()}, "
              f"après {duree / 3600:.2f} h")
        nouveau = lire_lignes_05m(lire_depuis(chemin_log, position))
        refus = nouveau["refus"]
        meta = lire_meta_point(dossier_modele)
        etat = etat_du_pli(meta)
        sessions = lire_json(sessions_chemin, [])
        sessions.append({"session": suivi05c.SESSION, "type": "entrainement", "gpu": gpu, "code": code,
                         "interrompu": interrompu, "duree_s": duree, "pic_mo": suivi_g.pic if suivi_g else None,
                         "total_mo": suivi_g.total if suivi_g else None,
                         "boucles_faites_a_la_fin": meta["prochaine_epoque"] if meta else 0, "reprises": nouveau["reprises"]})
        rs.ecrire_json_atomique(sessions, sessions_chemin)
        if suivi_g and suivi_g.pic:
            print(f"Pic de mémoire GPU de la session (nvidia-smi, carte entière) : {suivi_g.pic / 1024:.1f} Go sur "
                  f"{suivi_g.total / 1024:.1f} Go")
        if refus:
            tracer("reprise_refusee")
            print("ARRÊT : le modèle a REFUSÉ la reprise (rien n'a été entraîné) :\n  - " + "\n  - ".join(refus))
            print("  Le point de reprise ne correspond pas à cet entraînement (souvent : autre type de GPU). En parler avant "
                  "toute action ; --ecraser repartirait de zéro.")
            sys.exit(1)

    # --- Contrôles des réglages effectifs (journal de la dernière session d'entraînement) --------------
    texte = lire_depuis(chemin_log)
    j = l05.lire_journal(texte)
    session = j["sessions"][-1] if j["sessions"] else None
    usage = []
    if session:
        r = session["reglages"]
        if r.get("fichiers") and tuple(int(x) for x in r["fichiers"]) != (len(plis[k]["train"]), len(val_pli)):
            usage.append(f"segmenter : {r['fichiers'][0]} cas d'entraînement / {r['fichiers'][1]} de validation, attendu "
                         f"{len(plis[k]['train'])} / {len(val_pli)}")
        calendriers = lire_lignes_05m(texte)["calendriers"]
        crops = int(r.get("num_crops_per_image", 1))
        nb_boucles = int(r["num_epochs"]) if r.get("num_epochs") else None
        # Calendrier écrit juste avant la 1re boucle : contrôlé seulement si la session a fait au moins une époque
        # (une coupure pendant la mise en cache des images n'est pas une anomalie).
        if nb_boucles and session["epoques"] and (not calendriers or calendriers[-1] != calendrier_attendu(nb_boucles, crops)):
            anomalies.append(f"calendrier des validations {calendriers[-1] if calendriers else 'absent'} ≠ attendu "
                             f"{calendrier_attendu(nb_boucles, crops)} (toutes les {VALIDATION_TOUTES_LES}, dernière comprise)")
    if etat == "entraine":
        derniere = int(meta["derniere_epoque_faite"])
        if "val_dice_moyen" not in j["epoques"].get(derniere, {}):
            anomalies.append(f"la dernière époque ({derniere}) n'a pas été validée pendant l'entraînement")

    # --- 8. Évaluation finale avec model_final.pt ------------------------------------------------------
    bilan, code_eval, pb_eval = lire_json(os.path.join(dossier_pli, NOM_BILAN)), 0, []
    meta_eval = {}
    final = os.path.join(dossier_modele, "model_final.pt")
    if etat == "entraine" and not usage:
        a_jour = bool(bilan and bilan.get("conforme") and os.path.isfile(final)
                      and bilan.get("sha256_model_final") == prep.sha256_fichier(final))
        if a_jour:
            print("Évaluation finale déjà faite et conforme pour ce model_final.pt (bilan_05m.json) : non refaite.")
        elif not a.tracer_seulement:
            crops = int(meta["num_crops_per_image"])
            bilan, code_eval, pb_eval, suivi_e, duree_e = evaluer(algo, a, k, val_pli, dossier_pli, dossier_modele,
                                                                  int(meta["num_epochs"]), crops)
            sessions = lire_json(sessions_chemin, [])
            sessions.append({"session": suivi05c.SESSION, "type": "evaluation_finale", "gpu": gpu, "code": code_eval,
                             "duree_s": duree_e, "pic_mo": suivi_e.pic if suivi_e else None})
            rs.ecrire_json_atomique(sessions, sessions_chemin)
            if pb_eval:
                print("Problèmes de l'évaluation finale :\n  - " + "\n  - ".join(pb_eval))
        if bilan and bilan.get("cas"):
            afficher_bilan(bilan)
            meta_eval = {"dice_final_moyen": bilan["dice_moyen"], "duree_evaluation_finale_s": bilan.get("duree_evaluation_s")}
            for nom, d in bilan["dice_moyen_par_structure"].items():
                meta_eval[f"dice_final_{nom}"] = d
            for c, x in bilan["cas"].items():
                meta_eval[f"dice_final_moyen_{c}"] = x["dice_moyen"]
                for nom, d in x["dice"].items():
                    meta_eval[f"dice_final_{c}_{nom}"] = d

    # --- Bilan, MLflow, verdict ------------------------------------------------------------------------
    ep = j["epoques"]
    if ep:
        print(f"\nÉpoques faites : {len(ep)} sur {a.epoques}. Dernières validations (Dice moyen des 3 cas du pli, PAS le test) :")
        for e, x in [(e, x) for e, x in sorted(ep.items()) if "val_dice_moyen" in x][-5:]:
            print(f"  époque {e:>4} : {x['val_dice_moyen']:.4f}  ({' '.join(f'{d:.3f}' for d in x['val_dice'])})")
    termine = etat == "entraine" and bool(bilan and bilan.get("conforme")) and not usage and not anomalies
    statut = ("interrompu" if interrompu else "echec" if code else "validation_non_confirmee" if usage else
              "evaluation_non_conforme" if code_eval in (6, 7) else "anomalie" if anomalies else
              "termine" if termine else "en_cours")
    tracer(statut, meta_eval)
    if usage or code_eval == 6:
        print("Résultat : NON CONFIRMÉ, le pli n'a peut-être pas été validé sur ses 3 cas. Ne pas utiliser :")
        for u in usage + (pb_eval if code_eval == 6 else []):
            print("  -", u)
        sys.exit(6)
    if code_eval == 7:
        print("Résultat : prédictions finales NON CONFORMES (voir ci-dessus). Ne pas les utiliser ; en parler.")
        sys.exit(7)
    if anomalies or code_eval == 1:
        print("Résultat : réglage effectif NON CONFORME. Ne pas utiliser avant examen :")
        for x in anomalies + (pb_eval if code_eval == 1 else []):
            print("  -", x)
        sys.exit(1)
    if code or interrompu or code_eval == 3 or (not a.tracer_seulement and etat != "entraine"):
        print("Résultat : session INTERROMPUE ou en erreur. Pour reprendre, relancer la même cellule (au besoin dans une "
              "nouvelle session, après les sections 1, 2 et la cellule d'installation de la section 25).")
        sys.exit(3)
    print(f"Résultat : pli {k} {'TERMINÉ' if termine else 'en cours'}. Le lot de test n'a été ni lu ni utilisé ; seuls les "
          f"3 cas de validation du pli ({', '.join(val_pli)}) ont été prédits.")


if __name__ == "__main__":
    main()
