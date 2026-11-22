"""
Étape 6, préalable à la phase 0 bis (décision D44) — Refaire les 5 prédictions de TotalSegmentator, puis
mesurer la reproductibilité sur mmwhs_1004.

Pourquoi : dans poc_ct/predictions/totalsegmentator, seul mmwhs_1004.nii.gz a survécu à l'incident du Drive
plein ; les 4 autres prédictions de l'étape 4 et tracabilite_etape4.json ont disparu. D44 : refaire les
5 prédictions avec EXACTEMENT la même version (TotalSegmentator 2.18.0), la même tâche (heartchambers_highres),
les mêmes réglages et la même licence qu'à l'étape 4, dans un NOUVEAU dossier ; comparer le nouveau 1004
à l'ancien, voxel par voxel.

Ce que fait le script :
  1. garde-fous, AVANT tout calcul : --sortie différent du dossier d'origine (ni dedans, ni au-dessus),
     hors du dépôt et de tout dossier nnU-Net, VIDE ou inexistant (jamais d'écrasement) ; aucun chemin vers
     labelsTs ou labelsTr ; imagesTs = exactement les 5 scanners du découpage figé (lu, jamais modifié) ;
     ancien mmwhs_1004.nii.gz présent ; place libre sur Drive ;
  2. versions : TotalSegmentator 2.18.0 et nnU-Net v2 2.8.1, sinon ARRÊT (une autre version peut donner
     d'autres masques) ;
  3. mêmes réglages qu'à l'étape 4 : ils sont LUS dans scripts/04_totalsegmentator.py (ses valeurs par défaut,
     avec la ligne de commande de la section 10 : --images et --sortie seulement), puis la même préparation
     (nnUNet_def_n_proc avant l'import, statistiques d'usage coupées, licence par TOTALSEG_LICENCE, D21) ;
  4. pour chaque scanner, un à la fois : prédiction écrite d'abord sur le disque LOCAL de Colab, contrôlée
     (comme 04 : même forme et même affine que le scanner, étiquettes 1 à 7), puis COPIÉE sur Drive avec
     écriture forcée (os.fsync), RELUE depuis Drive (taille et SHA-256 identiques à la copie locale, image
     décodée, forme, affine, étiquettes) ;
  5. tracabilite_etape4.json (même nom qu'à l'étape 4, attendu par 06a et etape6_commun ; contenu marqué D44,
     GPU utilisé) écrit de la même façon, relu en JSON ; il ne contient JAMAIS la clé de licence (vérifié
     avant l'écriture) ;
  6. récapitulatif : les 6 fichiers attendus dans --sortie, avec leur taille ; tout manque = ARRÊT (code 7) ;
  7. comparaison du nouveau mmwhs_1004.nii.gz avec l'ancien (LECTURE SEULE) : forme, affine, voxels
     différents au total et par étiquette (matrice de confusion), SHA-256 des deux fichiers. Verdict :
     « IDENTIQUES » ou « ÉCART MESURÉ ». Résultat écrit en JSON dans --comparaison (PAS dans --sortie :
     06a s'arrête sur un NIfTI en trop et signale tout autre fichier), puis une ligne pour compléter D44.
  Le script ne lit AUCUN masque de vérité (labelsTs reste fermé jusqu'à la phase C) et ne calcule AUCUN score
  contre la vérité. Le dossier d'origine n'est jamais écrit : son contenu est relevé avant et après.

Entrées :
  --images       dossier imagesTs (défaut /content/drive/MyDrive/poc_ct/nnunet_raw/Dataset101_MMWHS/imagesTs)
  --sortie       NOUVEAU dossier des prédictions (défaut .../poc_ct/predictions/totalsegmentator_d44)
  --origine      dossier de l'étape 4, LECTURE SEULE de mmwhs_1004.nii.gz (défaut .../poc_ct/predictions/totalsegmentator)
  --comparaison  dossier du résultat de la comparaison (défaut .../poc_ct/etape6/d44_reproductibilite)
  --split        découpage figé (défaut : splits/mmwhs_split.json du dépôt)
  Variable d'environnement TOTALSEG_LICENCE : clé de licence (jamais en argument, jamais affichée ni écrite).
  Aucune option de réglage (tâche, appareil, processus) : ce sont ceux de l'étape 4, lus dans 04.
Sorties :
  --sortie      : mmwhs_1004.nii.gz ... mmwhs_1014.nii.gz et tracabilite_etape4.json (6 fichiers, rien d'autre) ;
  --comparaison : comparaison_1004_d44.json.
Matériel et durée : GPU L4 (l'étape 4 était sur T4 : 576 s pour 5 cas) ; environ 10 à 20 min avec le
  téléchargement des poids.
Codes de sortie :
  0 = 5 prédictions et traçabilité écrites, relues, conformes ; comparaison écrite (IDENTIQUES ou ÉCART MESURÉ) ;
  1 = garde-fou (chemin interdit, labelsTs, sortie = dossier d'origine, imagesTs incomplet, réglages de 04
      changés, place insuffisante...). Rien n'est écrit ;
  2 = --sortie n'est pas vide, ou la comparaison existe déjà : rien n'est écrasé ;
  3 = ERREUR PENDANT LA PRÉDICTION : --sortie est incomplet (voir le message pour relancer) ;
  4 = licence absente ou refusée ; 5 = pas de GPU ; 9 = version de TotalSegmentator ou de nnU-Net différente.
      Rien n'est écrit ;
  6 = tout est écrit, mais CONTRÔLE EN ÉCHEC (géométrie différente du scanner ou structure absente) ;
  7 = écriture sur Drive NON CONFIRMÉE (fichier absent, taille ou empreinte relue différente, JSON illisible,
      récapitulatif incomplet) ;
  8 = comparaison impossible (ancien 1004 absent ou illisible). Les prédictions déjà vérifiées restent valables.

Vocabulaire :
  - fsync : demande au système d'écrire vraiment le fichier, au lieu de le garder en mémoire. Dans Colab,
    Drive est un « montage » : le fichier est d'abord écrit dans un cache de la machine, puis envoyé aux
    serveurs de Google en arrière-plan. fsync et la relecture prouvent que le montage a le fichier COMPLET ;
    seule l'interface web de Drive (ou drive.flush_and_unmount(), en fin de section) prouve qu'il est arrivé
    chez Google. D'où la consigne finale ;
  - matrice de confusion : tableau qui compte, pour chaque étiquette de l'ancien fichier (ligne), combien de
    ses voxels ont chaque étiquette dans le nouveau (colonne). Tout hors de la diagonale = voxel différent ;
  - les octets d'un .nii.gz peuvent différer (en-tête, compression) alors que les voxels sont identiques :
    la conclusion porte sur les VOXELS ; les SHA-256 sont données pour information.
"""
# Au niveau du module : imports légers seulement (les processus enfants de nnU-Net, en mode spawn, relisent ce
# fichier). Les imports lourds (torch, nibabel, totalsegmentator) sont faits dans main(), après les garde-fous.
import argparse, datetime, hashlib, importlib.util, json, os, shutil, sys, tempfile, time
from importlib.metadata import version as _version, PackageNotFoundError

ICI = os.path.dirname(os.path.abspath(__file__))


def _charger_04():
    """Le script de l'étape 4, chargé comme module : il n'exécute rien à l'import (main() est protégé)."""
    spec = importlib.util.spec_from_file_location("ts04", os.path.join(ICI, "04_totalsegmentator.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ts04 = _charger_04()

# --- Paramètres fixes (D44) ------------------------------------------------------------------------
DRIVE = "/content/drive/MyDrive/poc_ct"
DEFAUTS = {"images": f"{DRIVE}/nnunet_raw/Dataset101_MMWHS/imagesTs",
           "sortie": f"{DRIVE}/predictions/totalsegmentator_d44",
           "origine": f"{DRIVE}/predictions/totalsegmentator",
           "comparaison": f"{DRIVE}/etape6/d44_reproductibilite"}
VERSION_TOTALSEG, VERSION_NNUNET = "2.18.0", "2.8.1"     # étape 4 (04 : « vérifié dans 2.18.0 et 2.8.1 »)
REGLAGES_ETAPE4 = {"tache": "heartchambers_highres", "appareil": "gpu", "processus": 1, "statistiques_usage": False}
GPU_ETAPE4 = "Tesla T4"                                   # README, étape 4 : 576 s pour 5 cas
CAS_COMPARE = "1004"
TAILLE_ANCIEN_1004 = 728942                               # octets, relevé dans D44
NOM_COMPARAISON = "comparaison_1004_d44.json"
PLACE_MIN_OCTETS = 200 * 1024 * 1024                      # 5 prédictions ≈ 5 Mo : large marge
ETIQUETTES_AUTORISEES = set(range(8))                     # 0 (fond) à 7, comme R0


# --- Utilitaires (aucun effet à l'import) ---------------------------------------------------------
arret, sans_cle, sha256_fichier, commande, version_paquet = (ts04.arret, ts04.sans_cle, ts04.sha256_fichier,
                                                            ts04.commande, ts04.version_paquet)


def maintenant():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def composants(chemin):
    return [c.lower() for c in os.path.abspath(chemin).replace("\\", "/").split("/") if c]


def dedans(chemin, dossier):
    """Vrai si chemin est dossier lui-même ou un de ses sous-dossiers (chemins résolus)."""
    c, d = os.path.realpath(chemin), os.path.realpath(dossier)
    try:
        return os.path.commonpath([c, d]) == d
    except ValueError:                                    # lecteurs différents (Windows)
        return False


def verifier_jamais_verite(chemins):
    """Aucun chemin vers une vérité terrain (labelsTs, labelsTr) : arrêt avant toute lecture."""
    for option, c in chemins.items():
        if {"labelsts", "labelstr"} & set(composants(c)) or {"labelsts", "labelstr"} & set(composants(os.path.realpath(c))):
            arret(1, f"{option} ({c}) désigne un dossier de vérité terrain : interdit (labelsTs reste fermé).")


def verifier_dossier_donnees(chemin, option, images):
    """Mêmes règles que 04 pour --sortie : hors du dépôt, hors de nnunet_raw/preprocessed/results et de imagesTs."""
    if dedans(chemin, ts04.DEPOT):
        arret(1, f"{option} ({chemin}) est dans le dépôt : les données restent dans Google Drive.")
    interdits = [d for d in os.path.abspath(chemin).replace("\\", "/").split("/") if ts04.DOSSIERS_INTERDITS.match(d)]
    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if os.environ.get(var) and dedans(chemin, os.environ[var]):
            interdits.append(f"${var}")
    if interdits or dedans(chemin, images):
        arret(1, f"{option} ({chemin}) est dans un dossier de données d'entraînement ({interdits or images}) : "
                 "règle 5, une prédiction de TotalSegmentator ne doit jamais pouvoir servir à l'entraînement.")


def verifier_sortie(sortie, origine):
    """Le dossier d'origine n'est JAMAIS écrit ; la nouvelle sortie est vide (pas d'écrasement silencieux)."""
    if dedans(sortie, origine) or dedans(origine, sortie):
        arret(1, f"--sortie ({sortie}) est le dossier d'origine, ou le contient, ou est dedans ({origine}). "
                 "Le dossier de l'étape 4 ne doit jamais être modifié (D44).")
    if os.path.exists(sortie) and not os.path.isdir(sortie):
        arret(1, f"--sortie ({sortie}) existe et n'est pas un dossier.")
    if os.path.isdir(sortie) and os.listdir(sortie):
        arret(2, f"--sortie ({sortie}) contient déjà {sorted(os.listdir(sortie))}. Rien n'a été écrasé. "
                 "Si une tentative précédente a été coupée : dans Drive, RENOMMER ce dossier (par exemple "
                 "totalsegmentator_d44_interrompu_<date>), ne rien effacer, puis relancer la cellule.")


def lister(dossier):
    """Noms, tailles et dates du dossier (sans ouvrir les fichiers) : preuve que le dossier d'origine n'a pas changé."""
    return {n: [os.stat(os.path.join(dossier, n)).st_size, os.stat(os.path.join(dossier, n)).st_mtime]
            for n in sorted(os.listdir(dossier))}


def synchroniser_dossier(dossier):
    """fsync du dossier (enregistre le nouveau nom). Pas toujours possible (Windows, certains montages) : sans effet alors."""
    try:
        fd = os.open(dossier, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def ecrire_force(chemin, donnees_ou_source):
    """Écrit (octets) ou copie (chemin d'un fichier local) vers chemin : fichier « .partiel », flush, os.fsync,
    renommage d'un seul coup, fsync du dossier. Renvoie (taille, SHA-256) de ce qui a été écrit.
    Une erreur du système (Drive plein, montage perdu : l'incident de D44) = ARRÊT 7, jamais un code 1."""
    tmp = chemin + ".partiel"
    h, taille = hashlib.sha256(), 0
    try:
        with open(tmp, "wb") as fo:
            def ecrire(bloc):
                nonlocal taille
                fo.write(bloc)
                h.update(bloc)
                taille += len(bloc)
            if isinstance(donnees_ou_source, bytes):
                ecrire(donnees_ou_source)
            else:
                with open(donnees_ou_source, "rb") as fi:
                    for bloc in iter(lambda: fi.read(1 << 20), b""):
                        ecrire(bloc)
            fo.flush()
            os.fsync(fo.fileno())                         # écriture forcée (cache du montage Drive)
        os.replace(tmp, chemin)
    except OSError as err:
        arret(7, f"écriture de {chemin} IMPOSSIBLE ({err!r}) : Drive plein ou montage perdu ? Le dossier est "
                 "INCOMPLET (un fichier « .partiel » peut rester) : dans Drive, le RENOMMER (…_interrompu_<date>), "
                 "ne rien effacer, libérer de la place, puis relancer.")
    synchroniser_dossier(os.path.dirname(os.path.abspath(chemin)))
    return taille, h.hexdigest()


def verifier_present(chemin, taille, sha):
    """Relecture depuis Drive : présent, taille > 0, taille et SHA-256 relues = celles écrites. Arrêt 7 sinon."""
    if not os.path.isfile(chemin):
        arret(7, f"{chemin} ABSENT juste après l'écriture : écriture sur Drive non confirmée.")
    lue = os.stat(chemin).st_size
    if lue <= 0 or lue != taille:
        arret(7, f"{chemin} : taille relue {lue} octets, écrite {taille} : écriture sur Drive non confirmée.")
    relu = sha256_fichier(chemin)
    if relu != sha:
        arret(7, f"{chemin} : SHA-256 relue {relu} ≠ écrite {sha} : écriture sur Drive non confirmée.")
    return lue


def controler_nifti(chemin_pred, chemin_scanner, etiquettes):
    """Contrôle de 04 (ce n'est pas un score) : image décodée, même forme et même affine que le scanner,
    étiquettes présentes, aucune valeur hors de 0 à 7."""
    import nibabel as nib
    import numpy as np
    scanner, pred = nib.load(chemin_scanner), nib.load(chemin_pred)
    donnees = np.asanyarray(pred.dataobj)
    valeurs = sorted(int(v) for v in np.unique(donnees))
    presentes = [v for v in valeurs if v != 0]
    return {"forme": list(pred.shape), "meme_forme": scanner.shape[:3] == pred.shape[:3],
            "meme_affine": bool(np.allclose(scanner.affine, pred.affine, atol=1e-3)),
            "ecart_affine_max": float(np.max(np.abs(scanner.affine - pred.affine))),
            "etiquettes_presentes": presentes,
            "structures_absentes": [etiquettes[k] for k in etiquettes if k not in presentes],
            "valeurs_hors_0_7": [v for v in valeurs if v not in ETIQUETTES_AUTORISEES]}


def conforme(c):
    return c["meme_forme"] and c["meme_affine"] and not c["structures_absentes"] and not c["valeurs_hors_0_7"]


# --- Comparaison voxel par voxel (fonctions pures, testées sur de petits tableaux) -----------------
def comparer_voxels(ancien, nouveau):
    """Compare deux masques d'étiquettes (tableaux numpy entiers). Renvoie un dict : même forme, nombre de voxels,
    voxels différents, matrice de confusion (lignes = ancien, colonnes = nouveau) et décompte par étiquette."""
    import numpy as np
    if tuple(ancien.shape) != tuple(nouveau.shape):
        return {"meme_forme": False, "forme_ancien": list(ancien.shape), "forme_nouveau": list(nouveau.shape)}
    a, b = np.asarray(ancien).astype(np.int64).ravel(), np.asarray(nouveau).astype(np.int64).ravel()
    etiq = sorted(set(np.unique(a).tolist()) | set(np.unique(b).tolist()))
    rang = {e: i for i, e in enumerate(etiq)}
    ia = np.searchsorted(etiq, a)                         # indice de l'étiquette (etiq est trié)
    ib = np.searchsorted(etiq, b)
    k = len(etiq)
    conf = np.bincount(ia * k + ib, minlength=k * k).reshape(k, k)
    par_etiquette = {}
    for e in etiq:
        i = rang[e]
        n_a, n_b = int(conf[i, :].sum()), int(conf[:, i].sum())
        par_etiquette[str(e)] = {"voxels_ancien": n_a, "voxels_nouveau": n_b,
                                 "perdus": n_a - int(conf[i, i]),          # e dans l'ancien, autre dans le nouveau
                                 "gagnes": n_b - int(conf[i, i])}          # e dans le nouveau, autre dans l'ancien
    return {"meme_forme": True, "forme": list(ancien.shape), "voxels_total": int(a.size),
            "voxels_differents": int(a.size - np.trace(conf)), "etiquettes": etiq,
            "matrice_confusion": {"lignes_ancien_colonnes_nouveau": etiq, "valeurs": conf.tolist()},
            "par_etiquette": par_etiquette}


def verdict(comparaison, meme_affine):
    """« IDENTIQUES » si même forme, même affine et aucun voxel différent ; sinon « ÉCART MESURÉ »."""
    if comparaison.get("meme_forme") and meme_affine and comparaison.get("voxels_differents") == 0:
        return "IDENTIQUES"
    return "ÉCART MESURÉ"


def comparer_1004(ancien, nouveau):
    """Lit les deux fichiers (lecture seule), compare en-têtes et voxels. Renvoie le contenu du JSON."""
    import nibabel as nib
    import numpy as np
    ia, ib = nib.load(ancien), nib.load(nouveau)
    meme_affine = bool(np.allclose(ia.affine, ib.affine, rtol=0, atol=1e-6))
    res = comparer_voxels(np.asanyarray(ia.dataobj), np.asanyarray(ib.dataobj))
    v = verdict(res, meme_affine)
    sha_a, sha_b = sha256_fichier(ancien), sha256_fichier(nouveau)
    return {"objet": "reproductibilité de TotalSegmentator sur mmwhs_1004 (D44) : prédiction de l'étape 4 contre "
                     "prédiction refaite ; aucune vérité lue, aucun score contre la vérité",
            "verdict": v, "conclusion_sur": "les voxels (les octets d'un .nii.gz peuvent différer : en-tête, compression)",
            "ancien": {"chemin": os.path.abspath(ancien), "sha256": sha_a,
                       "taille_octets": os.stat(ancien).st_size, "taille_attendue_D44": TAILLE_ANCIEN_1004,
                       "type_voxels": str(ia.get_data_dtype())},
            "nouveau": {"chemin": os.path.abspath(nouveau), "sha256": sha_b,
                        "taille_octets": os.stat(nouveau).st_size, "type_voxels": str(ib.get_data_dtype())},
            "octets_identiques": sha_a == sha_b,
            "meme_affine": meme_affine, "affine_exactement_egale": bool(np.array_equal(ia.affine, ib.affine)),
            "ecart_affine_max": float(np.max(np.abs(ia.affine - ib.affine))), "voxels": res}


def ligne_journal(c, gpu):
    v = c["voxels"]
    if c["verdict"] == "IDENTIQUES":
        detail = (f"IDENTIQUES : reproductibilité démontrée sur 1004 ({v['voxels_total']} voxels, 0 différent, "
                  f"même forme, même affine)")
    elif not v.get("meme_forme"):
        detail = f"ÉCART MESURÉ : formes différentes (ancien {v['forme_ancien']}, nouveau {v['forme_nouveau']})"
    else:
        par = " ; ".join(f"{e} : −{x['perdus']} / +{x['gagnes']}" for e, x in v["par_etiquette"].items()
                         if x["perdus"] or x["gagnes"])
        detail = (f"ÉCART MESURÉ : {v['voxels_differents']} voxels différents sur {v['voxels_total']} "
                  f"({100 * v['voxels_differents'] / v['voxels_total']:.4f} %) ; affine identique : "
                  f"{'oui' if c['meme_affine'] else 'NON'} ; par étiquette (perdus / gagnés) : {par}")
    return (f"Complément de D44 ({c['date_utc'][:10]}) : prédictions de TotalSegmentator refaites (2.18.0, "
            f"heartchambers_highres, GPU {gpu}) ; comparaison de mmwhs_1004 avec l'étape 4 : {detail}. "
            f"SHA-256 ancien {c['ancien']['sha256']}, nouveau {c['nouveau']['sha256']} "
            f"(octets {'identiques' if c['octets_identiques'] else 'différents'} ; la conclusion porte sur les voxels).")


def ecrire_json_verifie(chemin, contenu, cle):
    """JSON écrit avec fsync, puis relu (taille, SHA-256, décodage JSON identique). Jamais la clé de licence."""
    texte = json.dumps(contenu, indent=2, ensure_ascii=False)
    if cle and cle in texte:
        arret(1, f"la clé de licence apparaît dans le contenu de {os.path.basename(chemin)} : rien n'est écrit.", cle)
    taille, sha = ecrire_force(chemin, texte.encode("utf-8"))
    lue = verifier_present(chemin, taille, sha)
    try:
        with open(chemin, encoding="utf-8") as f:
            relu = json.load(f)
    except ValueError as err:
        arret(7, f"{chemin} illisible en JSON après écriture ({err}).")
    if relu != json.loads(texte):
        arret(7, f"{chemin} : contenu relu différent du contenu écrit.")
    return lue, sha


def lire_arguments(argv=None):
    p = argparse.ArgumentParser(description="D44 : refaire les prédictions de TotalSegmentator, comparer 1004.")
    for nom, defaut in DEFAUTS.items():
        p.add_argument(f"--{nom}", default=defaut)
    p.add_argument("--split", default=os.path.join(ts04.DEPOT, "splits", "mmwhs_split.json"))
    return p.parse_args(argv)


def reglages_de_04(images, sortie):
    """Réglages de l'étape 4 LUS dans 04 : ses valeurs par défaut, avec la ligne de commande de la section 10
    (--images et --sortie seulement). Si 04 a changé depuis l'étape 4 : arrêt."""
    argv = sys.argv
    try:
        sys.argv = ["04_totalsegmentator.py", "--images", images, "--sortie", sortie]
        a = ts04.lire_arguments()
    finally:
        sys.argv = argv
    lus = {"tache": a.tache, "appareil": a.appareil, "processus": a.processus, "statistiques_usage": a.statistiques_usage}
    if lus != REGLAGES_ETAPE4 or a.ecraser:
        arret(1, f"réglages par défaut de 04 = {lus}, attendus ceux de l'étape 4 {REGLAGES_ETAPE4} : 04 a changé, en parler.")
    return a


# --- Programme principal --------------------------------------------------------------------------
def main():
    a = lire_arguments()
    cle = (os.environ.get("TOTALSEG_LICENCE") or "").strip()       # jamais affichée
    images, sortie, origine, dossier_comp = (os.path.abspath(x) for x in (a.images, a.sortie, a.origine, a.comparaison))
    print(f"D44 — prédictions de TotalSegmentator refaites ; début {maintenant()} (UTC)")
    print(f"Commit du dépôt : {commande('git', 'rev-parse', 'HEAD')}"
          + (" (+ fichiers modifiés)" if commande("git", "status", "--porcelain") else " (propre)"))

    # --- 1. Garde-fous (rien n'est lu ni écrit sur Drive avant qu'ils passent) -------------------------
    verifier_jamais_verite({"--images": images, "--sortie": sortie, "--origine": origine, "--comparaison": dossier_comp})
    if os.path.basename(images) != "imagesTs":
        arret(1, f"--images doit être le dossier imagesTs (scanners de test), pas {images}.")
    verifier_dossier_donnees(sortie, "--sortie", images)
    verifier_dossier_donnees(dossier_comp, "--comparaison", images)
    verifier_sortie(sortie, origine)
    if dedans(dossier_comp, sortie) or dedans(dossier_comp, origine):
        arret(1, f"--comparaison ({dossier_comp}) ne doit être ni dans --sortie ni dans le dossier d'origine.")
    chemin_comp = os.path.join(dossier_comp, NOM_COMPARAISON)
    if os.path.exists(chemin_comp):
        arret(2, f"{chemin_comp} existe déjà : rien n'est écrasé. En parler avant de relancer.")
    ancien_1004 = os.path.join(origine, ts04.NOM_SORTIE.format(n=CAS_COMPARE))
    if not os.path.isfile(ancien_1004):
        arret(8, f"ancien {ancien_1004} introuvable : la comparaison prévue par D44 serait impossible. Rien n'est lancé.")
    origine_avant = lister(origine)
    a4 = reglages_de_04(images, sortie)
    with open(a.split, encoding="utf-8") as f:                      # lecture seule du découpage figé
        split = json.load(f)
    cas_ts, cas_tr = split["test_sous_cle"], split["entrainement_validation_croisee"]
    if set(cas_ts) & set(cas_tr) or len(set(cas_ts)) != len(cas_ts) or len(cas_ts) != ts04.NB_CAS_TEST:
        arret(1, f"découpage incohérent (cas répété, présent dans les deux lots, ou test ≠ {ts04.NB_CAS_TEST} cas).")
    numeros = sorted(c.split("_")[-1] for c in cas_ts)
    attendus = {f"mmwhs_{n}_0000.nii.gz" for n in numeros}
    presents = {x for x in os.listdir(images) if x.endswith(".nii.gz")} if os.path.isdir(images) else set()
    if presents != attendus:
        arret(1, f"imagesTs ne correspond pas au lot de test. Manquants : {sorted(attendus - presents)} ; "
                 f"en trop : {sorted(presents - attendus)}.")
    print(f"Lot de test vérifié : {numeros} ; ancien {CAS_COMPARE} présent ({origine_avant[os.path.basename(ancien_1004)][0]} octets)")
    if origine_avant[os.path.basename(ancien_1004)][0] != TAILLE_ANCIEN_1004:
        print(f"ATTENTION : l'ancien {CAS_COMPARE} ne fait pas {TAILLE_ANCIEN_1004} octets (taille relevée dans D44).")
    parent = sortie
    while not os.path.isdir(parent) and os.path.dirname(parent) != parent:
        parent = os.path.dirname(parent)
    libre = shutil.disk_usage(parent).free
    print(f"Place libre vue par le montage Drive : {libre / 2**30:.1f} Go")
    if libre < PLACE_MIN_OCTETS:
        arret(1, f"moins de {PLACE_MIN_OCTETS // 2**20} Mo libres sur Drive : libérer de la place (incident D44).")

    # --- 2. Versions (avant tout import lourd) -------------------------------------------------------
    versions = {"TotalSegmentator": version_paquet("TotalSegmentator"), "nnunetv2": version_paquet("nnunetv2")}
    if versions != {"TotalSegmentator": VERSION_TOTALSEG, "nnunetv2": VERSION_NNUNET}:
        arret(9, f"versions installées {versions}, attendues TotalSegmentator {VERSION_TOTALSEG} et nnunetv2 "
                 f"{VERSION_NNUNET} (étape 4). Relancer la cellule d'installation de la section.")
    print(f"Versions : TotalSegmentator {VERSION_TOTALSEG}, nnU-Net v2 {VERSION_NNUNET} (comme l'étape 4)")

    # --- 3. Même préparation que 04 (lignes 189 à 233 de 04, reprises telles quelles) ---------------
    os.environ["nnUNet_def_n_proc"] = str(a4.processus)            # AVANT l'import de totalsegmentator
    import torch
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.config import (setup_totalseg, set_config_key, set_license_number,
                                         has_valid_license_offline, get_version)
    from totalsegmentator.map_to_binary import class_map, commercial_models
    if get_version() != VERSION_TOTALSEG:
        arret(9, f"TotalSegmentator importé en version {get_version()}, attendu {VERSION_TOTALSEG}.")
    if not torch.cuda.is_available():
        arret(5, "aucun GPU visible. Menu Exécution → Modifier le type d'exécution → GPU L4, puis relancer.")
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU : {gpu}" + ("" if "L4" in gpu else "  ATTENTION : L4 attendu (D44) ; le GPU sera noté dans la traçabilité."))
    setup_totalseg()
    set_config_key("send_usage_stats", False)                      # D21 : jamais d'envoi (numéro de licence)
    print("Statistiques d'usage envoyées à TotalSegmentator : non")
    if a4.tache in commercial_models:
        if cle:
            print(f"Clé de licence reçue par TOTALSEG_LICENCE : {cle[:4]}… ({len(cle)} caractères, masquée)")
            try:
                set_license_number(cle)
            except SystemExit:
                arret(4, "clé de licence refusée par TotalSegmentator (voir le message juste au-dessus).", cle)
        etat, _ = has_valid_license_offline()                      # message jamais affiché (il contient la clé)
        if etat != "yes":
            arret(4, f"la tâche {a4.tache} demande une licence (état : {etat}) : secret Colab TOTALSEG_LICENCE.", cle)
    etiquettes = {int(k): v for k, v in class_map[a4.tache].items()}
    # Même dictionnaire d'options que 04 (ligne 250), construit avec les mêmes valeurs.
    options = {"task": a4.tache, "ml": True, "device": a4.appareil,
               "nr_thr_resamp": a4.processus, "nr_thr_saving": a4.processus}
    print(f"Options passées à TotalSegmentator (identiques à l'étape 4) : {options}")

    # --- 4. Prédictions : local → contrôle → copie forcée sur Drive → relecture --------------------------
    os.makedirs(sortie, exist_ok=True)
    local = tempfile.mkdtemp(prefix="totalseg_d44_", dir="/content" if os.path.isdir("/content") else None)
    resultats, echecs, debut = [], [], time.time()
    for n in numeros:
        entree = os.path.join(images, f"mmwhs_{n}_0000.nii.gz")
        nom = ts04.NOM_SORTIE.format(n=n)
        print(f"\n=== mmwhs_{n} : prédiction en cours ({maintenant()} UTC) ===", flush=True)
        t0 = time.time()
        try:
            empreinte = sha256_fichier(entree)
            totalsegmentator(entree, os.path.join(local, nom), **options)
            ctl = controler_nifti(os.path.join(local, nom), entree, etiquettes)
        except (Exception, SystemExit) as err:
            print(f"\nERREUR PENDANT LA PRÉDICTION (mmwhs_{n}) : {sans_cle(repr(err), cle)}")
            print(f"Le dossier {sortie} est INCOMPLET. Dans Drive, le RENOMMER (…_interrompu_<date>), ne rien effacer, "
                  "corriger la cause, puis relancer.")
            sys.exit(3)
        duree = round(time.time() - t0, 1)
        dest = os.path.join(sortie, nom)
        taille, sha = ecrire_force(dest, os.path.join(local, nom))
        verifier_present(dest, taille, sha)
        relu = controler_nifti(dest, entree, etiquettes)          # relu DEPUIS DRIVE et décodé
        if relu != ctl:
            arret(7, f"{dest} relu depuis Drive ne donne pas le même contrôle que la copie locale.")
        if not conforme(relu):
            echecs.append(f"mmwhs_{n}")
        resultats.append({"cas": f"mmwhs_{n}", "scanner_sha256": empreinte, "fichier": nom, "duree_s": duree,
                          "meme_forme": relu["meme_forme"], "meme_affine": relu["meme_affine"],
                          "etiquettes_presentes": relu["etiquettes_presentes"],
                          "structures_absentes": relu["structures_absentes"], "valeurs_hors_0_7": relu["valeurs_hors_0_7"],
                          "prediction_sha256": sha, "prediction_taille_octets": taille})
        print(f"mmwhs_{n} : {duree} s ; écrit et RELU sur Drive : {taille} octets, SHA-256 {sha[:12]}… identique ; "
              f"forme {relu['forme']} identique au scanner : {'oui' if relu['meme_forme'] else 'NON'} ; affine identique : "
              f"{'oui' if relu['meme_affine'] else 'NON'} ; étiquettes {relu['etiquettes_presentes']}"
              + (f" ; ABSENTES : {relu['structures_absentes']}" if relu["structures_absentes"] else "")
              + (f" ; HORS 0-7 : {relu['valeurs_hors_0_7']}" if relu["valeurs_hors_0_7"] else ""), flush=True)

    # --- 5. Traçabilité (même nom qu'à l'étape 4 ; jamais la clé) ------------------------------------
    poids = []
    try:                                                           # poids téléchargés : noms, tailles, empreintes
        from totalsegmentator.config import get_weights_dir
        for racine, _, noms in os.walk(str(get_weights_dir())):
            for x in sorted(noms):
                if x.endswith(".pth"):
                    c = os.path.join(racine, x)
                    poids.append({"fichier": os.path.relpath(c, str(get_weights_dir())), "taille_octets": os.stat(c).st_size,
                                  "sha256": sha256_fichier(c)})
    except Exception as err:
        poids = f"non relevés ({sans_cle(repr(err), cle)})"
    trace = {
        "etape": "4 refaite selon D44 (prédictions de l'étape 4 perdues lors de l'incident du Drive plein) ; "
                 "prédictions seulement, aucun score (D19)",
        "decision": "D44", "script": "scripts/04c_refaire_totalseg_d44.py",
        "reglages_repris_de": "scripts/04_totalsegmentator.py : valeurs par défaut lues dans son analyseur d'arguments, "
                              "ligne de commande de la section 10 (--images, --sortie)",
        "date_utc": maintenant(), "totalsegmentator_version": get_version(), "nnunetv2_version": version_paquet("nnunetv2"),
        "torch_version": torch.__version__, "python_version": sys.version.split()[0], "gpu": gpu,
        "gpu_etape4": GPU_ETAPE4, "remarque_materiel": "GPU différent de l'étape 4 : de petits écarts de calcul sont "
                                                      "possibles ; la comparaison de mmwhs_1004 en donne la mesure",
        "code_commit": commande("git", "rev-parse", "HEAD") or "inconnu",
        "code_modifie_localement": bool(commande("git", "status", "--porcelain")),
        "tache": a4.tache, "options": options,
        "processus": {"nr_thr_resamp": a4.processus, "nr_thr_saving": a4.processus,
                      "nnUNet_def_n_proc": os.environ["nnUNet_def_n_proc"]},
        "licence": "requise, fournie par TOTALSEG_LICENCE (non enregistrée)" if a4.tache in commercial_models else "non requise",
        "statistiques_usage_envoyees": False,
        "split": {"fichier": os.path.relpath(a.split, ts04.DEPOT), "sha256": sha256_fichier(a.split)},
        "images": images, "etiquettes_totalsegmentator": etiquettes,
        "correspondance_prevue_mmwhs_etape6": ts04.CORRESPONDANCE_MMWHS,
        "dossier_origine": {"chemin": origine, "non_modifie": "contenu relevé avant et après (voir la sortie)"},
        "poids_modeles": poids, "cas": resultats, "duree_totale_s": round(time.time() - debut, 1),
        "rappel": "Aucun masque de vérité terrain lu, aucun score calculé. Ces prédictions ne servent jamais à "
                  "l'entraînement (règle 5, licence).",
    }
    chemin_trace = os.path.join(sortie, ts04.FICHIER_TRACE)
    taille_trace, sha_trace = ecrire_json_verifie(chemin_trace, trace, cle)
    print(f"\nTraçabilité écrite et RELUE en JSON : {chemin_trace} ({taille_trace} octets, SHA-256 {sha_trace[:12]}…)")

    # --- 6. Récapitulatif des 6 fichiers attendus ------------------------------------------------------
    attendus_sortie = {r["fichier"]: r["prediction_taille_octets"] for r in resultats}
    attendus_sortie[ts04.FICHIER_TRACE] = taille_trace
    presents_sortie = set(os.listdir(sortie))
    print(f"\nRÉCAPITULATIF de {sortie} :")
    manque = []
    for nom_f, t in sorted(attendus_sortie.items()):
        chemin = os.path.join(sortie, nom_f)
        lue = os.stat(chemin).st_size if os.path.isfile(chemin) else None
        ok = lue is not None and lue > 0 and lue == t
        manque += [] if ok else [nom_f]
        print(f"  {'OK   ' if ok else 'MANQUE'} {nom_f:<26} {lue if lue is not None else '—':>12} octets")
    autres = sorted(presents_sortie - set(attendus_sortie))
    if autres:
        print(f"  ATTENTION : fichiers en trop (06a s'arrêtera sur un NIfTI en trop) : {autres}")
    if manque or len(attendus_sortie) != 6:
        arret(7, f"récapitulatif incomplet : {manque}. Ne rien effacer ; envoyer la sortie dans la conversation.")
    print("  6 fichiers sur 6 présents, tailles relues = tailles écrites.")

    # --- 7. Comparaison de 1004 (ancien en lecture seule) -------------------------------------------------
    try:
        comp = comparer_1004(ancien_1004, os.path.join(sortie, ts04.NOM_SORTIE.format(n=CAS_COMPARE)))
    except Exception as err:
        arret(8, f"comparaison de {CAS_COMPARE} impossible ({sans_cle(repr(err), cle)}). Les 5 prédictions sont "
                 "écrites et vérifiées.")
    comp.update({"date_utc": maintenant(), "gpu": gpu, "gpu_etape4": GPU_ETAPE4, "decision": "D44",
                 "code_commit": trace["code_commit"], "tracabilite_d44_sha256": sha_trace})
    origine_apres = lister(origine)
    comp["dossier_origine_inchange"] = origine_apres == origine_avant
    if not comp["dossier_origine_inchange"]:
        print(f"ATTENTION : le contenu du dossier d'origine a changé pendant le script : avant {origine_avant}, après {origine_apres}")
    v = comp["voxels"]
    print(f"\nCOMPARAISON mmwhs_{CAS_COMPARE} (ancien de l'étape 4 / nouveau D44) — aucune vérité lue :")
    print(f"  SHA-256 ancien  : {comp['ancien']['sha256']} ({comp['ancien']['taille_octets']} octets)")
    print(f"  SHA-256 nouveau : {comp['nouveau']['sha256']} ({comp['nouveau']['taille_octets']} octets)")
    print(f"  octets {'identiques' if comp['octets_identiques'] else 'différents (possible même si les voxels sont identiques : en-tête, compression)'}")
    print(f"  même forme : {'oui' if v.get('meme_forme') else 'NON'} ; même affine : {'oui' if comp['meme_affine'] else 'NON'} "
          f"(écart max {comp['ecart_affine_max']:.2e})")
    if v.get("meme_forme"):
        print(f"  voxels différents : {v['voxels_differents']} sur {v['voxels_total']}")
        print(f"  {'étiquette':>9} {'ancien':>12} {'nouveau':>12} {'perdus':>9} {'gagnés':>9}")
        for e, x in v["par_etiquette"].items():
            print(f"  {e:>9} {x['voxels_ancien']:>12} {x['voxels_nouveau']:>12} {x['perdus']:>9} {x['gagnes']:>9}")
    print(f"  Dossier d'origine inchangé : {'oui' if comp['dossier_origine_inchange'] else 'NON'}")
    print(f"\nVERDICT : {'IDENTIQUES : reproductibilité démontrée sur 1004' if comp['verdict'] == 'IDENTIQUES' else 'ÉCART MESURÉ (chiffres ci-dessus)'}")
    os.makedirs(dossier_comp, exist_ok=True)
    taille_c, _ = ecrire_json_verifie(chemin_comp, comp, cle)
    print(f"Comparaison écrite et RELUE en JSON : {chemin_comp} ({taille_c} octets)")
    print(f"\nLigne proposée pour compléter D44 au journal :\n{ligne_journal(comp, gpu)}")

    # --- 8. Consigne finale ------------------------------------------------------------------------------
    print("\nÀ FAIRE MAINTENANT : ouvrir https://drive.google.com, dossier poc_ct/predictions/totalsegmentator_d44, "
          "et vérifier que les 6 fichiers ci-dessus y sont visibles avec ces tailles (clic droit → Informations "
          "sur le fichier) ; puis lancer la dernière cellule de la section (envoi forcé vers Drive).")
    if echecs:
        print(f"\nCONTRÔLE EN ÉCHEC pour {echecs} : géométrie différente du scanner, structure absente ou valeur hors 0-7.")
        sys.exit(6)
    print(f"Contrôle des prédictions : CONFORME ; fin {maintenant()} (UTC)")


# Garde indispensable : les processus enfants (mode spawn) relisent ce fichier sous le nom « __mp_main__ ».
if __name__ == "__main__":
    main()
