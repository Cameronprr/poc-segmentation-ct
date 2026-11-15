"""
Reprise COMPLÈTE de l'entraînement SegResNet (modèle d'algorithme Auto3DSeg 21ed8e5), décision D38.

Ce module n'est PAS lancé seul. Le script 05m le copie dans <dossier du pli>/segresnet_<k>/scripts/, à côté
du fichier segmenter.py du modèle SegResNet, et ajoute à segmenter.py quelques lignes qui l'appellent
(« correctif 05m », voir CORRECTIFS dans 05m). Il fait trois choses :
  1. restaurer() : au début de l'entraînement, s'il existe un point de reprise complet dans le dossier
     model/ du pli, il remet en place TOUT l'état : poids du réseau, état de l'optimiseur AdamW (ses
     moyennes internes), calendrier du taux d'apprentissage (scheduler), GradScaler (précision mixte),
     meilleur Dice de validation et son époque, historique des validations, et l'époque où reprendre.
     Il ajoute aussi la DERNIÈRE époque au calendrier des validations (le modèle ne la valide pas quand
     on demande une validation toutes les 10 époques : range(10, 300, 10) s'arrête à 290) ;
  2. sauver() : à la fin d'une époque sur 10 (et à la dernière), il écrit ce point de reprise complet ;
  3. sauver_atomique() / copier_atomique() : toute écriture passe par un fichier temporaire, puis un
     renommage. Une coupure pendant l'écriture laisse donc intact le fichier précédent.

Sans ce module, l'option « continue » du modèle ne reprend que les poids, le numéro d'époque et un
« meilleur Dice » approximatif (segmenter.py, l. 1020-1029 et 1296-1324) : l'optimiseur, le calendrier
du taux d'apprentissage et le GradScaler repartiraient de zéro.

Vocabulaire :
  - point de reprise : fichier qui contient l'état de l'entraînement, pour repartir après une coupure ;
  - optimiseur AdamW : la règle qui corrige les poids à chaque pas ; il garde deux moyennes par poids,
    perdues si l'on ne les sauvegarde pas (les premiers pas après une reprise seraient alors faussés) ;
  - calendrier du taux d'apprentissage (scheduler) : la taille des corrections diminue au fil des
    époques (montée sur 3 époques, puis descente en cosinus) ; il faut savoir où l'on en est ;
  - GradScaler : réglage automatique de la précision mixte (calculs en 16 bits) ; il garde un facteur
    d'échelle appris au fil des pas ;
  - écriture atomique : le fichier final est soit l'ancien complet, soit le nouveau complet, jamais un
    mélange ; le renommage (os.replace) est l'étape qui bascule de l'un à l'autre d'un seul coup.
"""
import datetime, json, os, shutil, time

FORMAT = 1                                         # version du contenu du point de reprise
NOM_POINT = "point_reprise_complet.pt"             # dans model/ du pli (même dossier que model_final.pt)
NOM_RESUME = "point_reprise_complet.json"          # résumé lisible (sans tenseurs), écrit APRÈS le .pt
MARQUE = "(correctif 05m)"                         # repère des lignes de ce module dans training.log
# Les mots-repères (REPRISE COMPLETE, REPRISE REFUSEE, DEPART A ZERO, Calendrier des validations) sont
# sans accents : training.log est écrit dans l'encodage du système, la détection n'en dépend donc pas.


# --- Écritures atomiques (testables sans GPU) ---------------------------------------------------
def _fsync(f):
    try:
        f.flush()
        os.fsync(f.fileno())                       # force l'écriture sur le disque avant le renommage
    except OSError:
        pass


def sauver_atomique(objet, chemin):
    """torch.save dans un fichier temporaire du même dossier, puis renommage d'un seul coup."""
    import torch
    tmp = f"{chemin}.tmp{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            torch.save(objet, f)
            _fsync(f)
        os.replace(tmp, chemin)
    finally:
        if os.path.exists(tmp):                    # écriture ratée : on ne laisse pas de demi-fichier
            os.remove(tmp)


def copier_atomique(source, cible):
    """Copie de fichier (model.pt → model_final.pt) par fichier temporaire puis renommage."""
    tmp = f"{cible}.tmp{os.getpid()}"
    try:
        with open(source, "rb") as fs, open(tmp, "wb") as fc:
            shutil.copyfileobj(fs, fc, 16 * 1024 * 1024)
            _fsync(fc)
        os.replace(tmp, cible)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def ecrire_json_atomique(objet, chemin):
    tmp = f"{chemin}.tmp{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(objet, f, indent=2, ensure_ascii=False)
            _fsync(f)
        os.replace(tmp, chemin)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# --- Fonctions pures (testables sans GPU) -------------------------------------------------------
def calendrier_avec_derniere(calendrier, nb_boucles):
    """Calendrier des validations du modèle + la dernière boucle si elle n'y est pas. Le modèle valide
    à la boucle e quand e + 1 ≥ prochaine valeur : la valeur nb_boucles fait valider la dernière boucle."""
    x = [int(v) for v in calendrier]
    if not x or x[-1] < nb_boucles:
        x.append(int(nb_boucles))
    return x


def validations_restantes(calendrier, prochaine):
    """Valeurs du calendrier encore à faire quand les boucles 0..prochaine−1 sont faites. La validation v
    a lieu à la boucle v − 1 : elle est faite si v ≤ prochaine. (Le modèle, lui, garde v ≥ start_epoch,
    l. 1302 : il refait une validation dès la première boucle après la reprise.)"""
    return [v for v in calendrier if v > prochaine]


def meta_courante(config, nb_boucles, crops, amp):
    """Réglages qui doivent être IDENTIQUES entre la session qui a écrit le point et celle qui reprend."""
    return {"identifiant_05m": config.get("identifiant_05m"), "fold": int(config["fold"]),
            "num_epochs_config": int(config["num_epochs"]), "num_epochs": int(nb_boucles),
            "num_crops_per_image": int(crops), "batch_size": int(config["batch_size"]),
            "roi_size": [int(r) for r in config["roi_size"]], "amp": bool(amp),
            "learning_rate": float(config["learning_rate"])}


def ecarts_meta(lue, attendue):
    """{réglage: (valeur du point, valeur actuelle)} pour chaque réglage différent."""
    return {k: (lue.get(k), v) for k, v in attendue.items() if lue.get(k) != v}


def chemin_point(dossier_modele):
    return os.path.join(dossier_modele, NOM_POINT)


# --- Journal : mêmes fichier et format que segmenter.py (training.log) --------------------------
def _ecrire(message):
    try:
        from monai.apps.auto3dseg.auto_runner import logger
        logger.debug(message)                      # segmenter.py : print = logger.debug → training.log
    except Exception:
        print(message, flush=True)


def _modele(seg):
    return seg.model.module if hasattr(seg.model, "module") else seg.model


# --- Appelées par segmenter.py corrigé --------------------------------------------------------
def restaurer(seg, optimizer, lr_scheduler, grad_scaler, *, num_epochs, num_crops_per_image, start_epoch,
              best_metric, best_metric_epoch, val_acc_history, val_schedule_list):
    """Appelée dans Segmenter.train(), juste avant la boucle des époques (optimiseur et calendrier du taux
    d'apprentissage déjà créés). Renvoie (start_epoch, best_metric, best_metric_epoch, val_acc_history,
    val_schedule_list) à utiliser par la boucle. Refuse (RuntimeError) toute reprise incohérente."""
    import torch
    config = seg.config
    calendrier = calendrier_avec_derniere(val_schedule_list, num_epochs)
    _ecrire(f"Calendrier des validations {MARQUE} : {calendrier}")
    if config.get("pretrained_ckpt_name") is not None or int(start_epoch) != 0:
        _refuser(f"le modèle a déjà chargé un point par son option « continue » ou « pretrained_ckpt_name » "
                 f"(start_epoch {start_epoch}, pretrained_ckpt_name {config.get('pretrained_ckpt_name')}) : "
                 "les deux mécanismes de reprise ne doivent pas se mélanger")
    chemin = chemin_point(config["ckpt_path"])
    if not os.path.isfile(chemin):
        _ecrire(f"DEPART A ZERO {MARQUE} : aucun point de reprise complet ({chemin})")
        return start_epoch, best_metric, best_metric_epoch, val_acc_history, calendrier
    etat = torch.load(chemin, map_location="cpu", weights_only=False)
    meta = etat.get("meta", {})
    ecarts = ecarts_meta(meta, meta_courante(config, num_epochs, num_crops_per_image, grad_scaler.is_enabled()))
    if etat.get("format") != FORMAT or ecarts:
        _refuser(f"le point {chemin} ne correspond pas à cet entraînement (format {etat.get('format')}, "
                 f"écarts (point, actuel) : {ecarts}). Un GPU différent peut changer batch / ROI / crops.")
    _modele(seg).load_state_dict(etat["modele"], strict=True)
    optimizer.load_state_dict(etat["optimiseur"])          # moyennes d'AdamW et taux courant
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(etat["calendrier_lr"])
    if grad_scaler.is_enabled():
        grad_scaler.load_state_dict(etat["grad_scaler"])
    prochaine = int(meta["prochaine_epoque"])
    reste = validations_restantes(calendrier, prochaine)
    _ecrire(f"REPRISE COMPLETE {MARQUE} : boucles 0 à {prochaine - 1} faites, reprise à la boucle {prochaine} sur "
            f"{num_epochs} ; taux d'apprentissage {optimizer.param_groups[0]['lr']:.4e} ; meilleur Dice "
            f"{float(meta['best_metric']):.4f} à l'époque {int(meta['best_metric_epoch'])} ; optimiseur AdamW, "
            f"calendrier du taux d'apprentissage{' et GradScaler' if grad_scaler.is_enabled() else ''} repris ; "
            f"{len(etat['val_acc_history'])} validations dans l'historique ; validations restantes {reste}")
    historique = [tuple(x) for x in etat["val_acc_history"]]
    return prochaine, float(meta["best_metric"]), int(meta["best_metric_epoch"]), historique, reste


def sauver(seg, *, prochaine_epoque, num_epochs, num_epochs_per_saving, num_crops_per_image, optimizer,
           lr_scheduler, grad_scaler, best_metric, best_metric_epoch, val_acc_history):
    """Appelée à la fin de chaque boucle, APRÈS lr_scheduler.step() : l'état écrit est celui qui précède
    la boucle prochaine_epoque. N'écrit qu'aux mêmes boucles que model_final.pt (toutes les
    num_epochs_per_saving, et la dernière)."""
    config = seg.config
    if seg.global_rank != 0 or config.get("ckpt_path") is None or not config.get("ckpt_save", True):
        return
    if not (prochaine_epoque % num_epochs_per_saving == 0 or prochaine_epoque >= num_epochs):
        return
    t0 = time.time()
    meta = meta_courante(config, num_epochs, num_crops_per_image, grad_scaler.is_enabled())
    meta.update({"prochaine_epoque": int(prochaine_epoque),
                 "derniere_epoque_faite": int((prochaine_epoque - 1) * num_crops_per_image),
                 "best_metric": float(best_metric), "best_metric_epoch": int(best_metric_epoch),
                 "lr_prochaine_epoque": float(optimizer.param_groups[0]["lr"]),
                 "nb_validations": len(val_acc_history),
                 "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
    etat = {"format": FORMAT, "meta": meta, "modele": _modele(seg).state_dict(),
            "optimiseur": optimizer.state_dict(),
            "calendrier_lr": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "grad_scaler": grad_scaler.state_dict(), "val_acc_history": [list(x) for x in val_acc_history]}
    chemin = chemin_point(config["ckpt_path"])
    sauver_atomique(etat, chemin)
    meta["duree_ecriture_s"] = round(time.time() - t0, 2)
    ecrire_json_atomique(meta, os.path.join(config["ckpt_path"], NOM_RESUME))   # après le .pt : jamais en avance
    _ecrire(f"Point de reprise complet écrit {MARQUE} : boucles 0 à {prochaine_epoque - 1} faites ; taux "
            f"d'apprentissage suivant {meta['lr_prochaine_epoque']:.4e} ; meilleur Dice {meta['best_metric']:.4f} "
            f"(époque {meta['best_metric_epoch']}) ; écriture {meta['duree_ecriture_s']:.1f} s")


def _refuser(raison):
    _ecrire(f"REPRISE REFUSEE {MARQUE} : {raison}")
    raise RuntimeError(f"REPRISE REFUSEE {MARQUE} : {raison}")
