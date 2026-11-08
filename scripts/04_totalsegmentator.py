"""
Étape 4 — Ligne de base TotalSegmentator sur les 5 scanners de test (prédictions seulement, décision D19).

Ce que fait le script :
  1. vérifie les garde-fous : --sortie hors du dépôt et hors de tout dossier nnU-Net (nnunet_raw,
     Dataset1xx...) ; --images est bien un dossier « imagesTs » qui contient EXACTEMENT les 5 scanners
     du lot de test du découpage figé (splits/mmwhs_split.json, lu seulement) ;
  2. coupe l'envoi des statistiques d'usage de TotalSegmentator (elles contiennent le numéro de
     licence, donc un identifiant personnel) et vérifie la licence de la tâche. La clé est lue
     UNIQUEMENT dans la variable d'environnement TOTALSEG_LICENCE : jamais en argument (visible dans la
     liste des processus), jamais affichée, jamais écrite dans le dépôt, dans Drive ou dans la traçabilité ;
  3. lance TotalSegmentator (son API Python officielle, réglages par défaut de la tâche, sauf le nombre
     de processus, réduit pour la mémoire de Colab) sur le GPU, UN scanner à la fois, et enregistre pour
     chaque cas un fichier multi-étiquettes mmwhs_XXXX.nii.gz ;
  4. contrôle chaque prédiction : même taille et même affine que le scanner, étiquettes présentes ;
  5. écrit un fichier de traçabilité tracabilite_etape4.json (versions, tâche, options, date, durées,
     empreintes des scanners d'entrée).
  Le script ne lit AUCUN masque de vérité terrain (labelsTs reste fermé jusqu'à l'étape 6, D18/D19)
  et ne calcule AUCUN score (ni Dice, ni comparaison). Les étiquettes ne sont PAS renumérotées vers
  MM-WHS ici : ce sera fait à l'étape 6 (voir CORRESPONDANCE_MMWHS plus bas, pour référence).

Entrées :
  --images    dossier imagesTs de Dataset101_MMWHS (dans Drive)
  --sortie    dossier des prédictions dans Drive, HORS de nnunet_raw (règle 5 : un masque produit par
              TotalSegmentator ne doit jamais pouvoir servir à l'entraînement)
  --split     découpage figé (défaut : ../splits/mmwhs_split.json à côté du script)
  --tache     tâche TotalSegmentator (défaut : heartchambers_highres)
  --appareil  gpu (défaut) ou cpu (très lent : seulement pour un essai)
  --processus nombre de processus enfants de nnU-Net pour le prétraitement et l'enregistrement
              (défaut : 1 ; plus = un peu plus rapide mais plus de mémoire vive)
  --ecraser   autorise à remplacer des prédictions déjà présentes dans --sortie
  --statistiques-usage  réactive volontairement l'envoi des statistiques d'usage (coupé par défaut)
  Variable d'environnement TOTALSEG_LICENCE : clé de licence (tâches soumises à licence).
Sorties (uniquement dans --sortie) :
  mmwhs_XXXX.nii.gz (une par cas de test, étiquettes 1..7 de TotalSegmentator) ; tracabilite_etape4.json
Codes de sortie :
  0 = 5 prédictions écrites et contrôlées ;
  1 = garde-fou : --sortie interdit, découpage incohérent, scanners manquants ou en trop. Rien n'est écrit ;
  2 = des prédictions existent déjà dans --sortie (relancer avec --ecraser). Rien n'est écrit ;
  3 = ERREUR PENDANT LA PRÉDICTION : --sortie est incomplet (pas de traçabilité), relancer avec
      --ecraser une fois la cause corrigée ;
  4 = licence absente ou refusée. Rien n'est écrit ;
  5 = pas de GPU visible (menu Exécution → Modifier le type d'exécution → GPU T4). Rien n'est écrit ;
  6 = prédictions et traçabilité écrites, mais CONTRÔLE EN ÉCHEC (géométrie différente du scanner ou
      structure absente) : à examiner avant l'étape 6.

Vocabulaire :
  - TotalSegmentator : outil public déjà entraîné (des modèles nnU-Net) qui segmente de nombreuses
    structures sur scanner. On s'en sert tel quel, sans l'entraîner : c'est la « ligne de base »,
    le score à battre par nos propres modèles de l'étape 5 ;
  - tâche (option task / -ta) : un des modèles de TotalSegmentator. « heartchambers_highres » segmente
    7 structures cardiaques à haute résolution ; elle est soumise à licence (gratuite pour un usage
    non commercial) ;
  - prédiction : le masque produit par un modèle. Ce n'est PAS une vérité terrain ;
  - fichier multi-étiquettes : un seul volume où chaque voxel vaut 0 (fond) ou le numéro d'une structure
    (option ml=True de TotalSegmentator, équivalent de --ml en ligne de commande) ;
  - recadrage (crop) : la tâche lance d'abord un modèle grossier (3 mm) pour trouver le cœur, puis
    applique le modèle haute résolution seulement dans une boîte autour du cœur ;
  - empreinte (SHA-256) : suite de 64 caractères calculée sur tout le fichier ; elle change si le
    fichier change d'un seul octet. Elle prouve, plus tard, quels scanners ont été traités ;
  - processus enfant, mode « spawn » : nnU-Net confie le prétraitement et l'enregistrement à d'autres
    programmes Python lancés en parallèle (des processus enfants). En mode « spawn », chaque enfant
    repart de zéro et relit ce fichier : c'est pourquoi tout le traitement est rangé dans main(),
    exécutée seulement par le programme principal (if __name__ == "__main__"). Chaque enfant consomme
    aussi de la mémoire vive : d'où --processus 1 par défaut.
"""
# Au niveau du module : seulement des imports légers, des constantes et des fonctions. Les imports
# lourds (torch, nibabel, totalsegmentator) sont faits dans main(), après les garde-fous.
import argparse, datetime, glob, hashlib, json, os, re, subprocess, sys, time
from importlib.metadata import version, PackageNotFoundError

# --- Paramètres fixes -------------------------------------------------------------------------
# Étiquettes produites par la tâche heartchambers_highres (TotalSegmentator 2.18.0,
# totalsegmentator/map_to_binary.py, class_map["heartchambers_highres"]). Relues dans le paquet à
# l'exécution et enregistrées dans le fichier de traçabilité.
#
# Correspondance PRÉVUE pour l'étape 6 (NON appliquée ici) : étiquette TotalSegmentator → MM-WHS.
# Les définitions ne sont pas toutes identiques : voir les remarques.
CORRESPONDANCE_MMWHS = {
    1: 205,  # heart_myocardium      → myocarde du VG. À vérifier : myocarde du VG seul, ou aussi paroi du VD ?
    2: 420,  # heart_atrium_left     → oreillette gauche. MM-WHS : cavité ; auricule et départ des veines
             #                         pulmonaires peut-être traités différemment.
    3: 500,  # heart_ventricle_left  → cavité du VG.
    4: 550,  # heart_atrium_right    → oreillette droite. Limite avec les veines caves possiblement différente.
    5: 600,  # heart_ventricle_right → ventricule droit (MM-WHS : cavité seule).
    6: 820,  # aorta                 → aorte ascendante. MM-WHS s'arrête au bord supérieur des oreillettes ;
             #                         TotalSegmentator nomme la classe « aorta » : la portion d'aorte
             #                         descendante proche du cœur peut être incluse. Comparaison biaisée.
    7: 850,  # pulmonary_artery      → artère pulmonaire. MM-WHS : tronc, de la valve à la bifurcation ;
             #                         TotalSegmentator peut inclure le début des branches droite et gauche.
}
NOM_SORTIE = "mmwhs_{n}.nii.gz"            # même nom que les masques nnU-Net : comparaison simple à l'étape 6
FICHIER_TRACE = "tracabilite_etape4.json"
NB_CAS_TEST = 5                            # D20 : 5 patients à un seul examen
DOSSIERS_INTERDITS = re.compile(r"^(nnunet_raw|nnunet_preprocessed|nnunet_results|Dataset\d{3}_.*)$", re.I)
DEPOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# --- Fonctions utilitaires (aucun effet à l'import) --------------------------------------------
def sans_cle(texte, cle):
    """Masque la clé si elle apparaît dans un message (ex. message d'erreur d'une bibliothèque)."""
    texte = str(texte)
    return texte.replace(cle, "[clé masquée]") if cle else texte


def arret(code, message, cle=""):
    print(f"ARRÊT : {sans_cle(message, cle)}")
    sys.exit(code)


def sha256_fichier(chemin):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def commande(*args):
    try:
        return subprocess.run(args, cwd=DEPOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "inconnu"


def version_paquet(nom):
    try:
        return version(nom)
    except PackageNotFoundError:
        return "inconnu"


def lire_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--sortie", required=True)
    p.add_argument("--split", default=os.path.join(DEPOT, "splits", "mmwhs_split.json"))
    p.add_argument("--tache", default="heartchambers_highres")
    p.add_argument("--appareil", default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--processus", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--ecraser", action="store_true")
    p.add_argument("--statistiques-usage", action="store_true")
    return p.parse_args()


# --- Programme principal ----------------------------------------------------------------------
def main():
    a = lire_arguments()
    # Clé nettoyée (espaces, retour à la ligne d'un copier-coller) AVANT tout usage. Jamais affichée.
    cle = (os.environ.get("TOTALSEG_LICENCE") or "").strip()

    # --- 1. Garde-fous (rien n'est écrit avant qu'ils soient tous passés) -----------------------
    sortie, images = os.path.abspath(a.sortie), os.path.abspath(a.images)
    if os.path.commonpath([sortie, DEPOT]) == DEPOT:
        arret(1, "--sortie est dans le dépôt. Les prédictions doivent rester dans Google Drive.")
    # Règle 5 : aucune prédiction TotalSegmentator dans un dossier que nnU-Net ou Auto3DSeg pourraient lire.
    interdits = [d for d in sortie.split(os.sep) if DOSSIERS_INTERDITS.match(d)]
    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        racine = os.environ.get(var)
        if racine and os.path.commonpath([sortie, os.path.abspath(racine)]) == os.path.abspath(racine):
            interdits.append(f"${var}")
    if interdits or os.path.commonpath([sortie, images]) == images:
        arret(1, f"--sortie ({sortie}) est dans un dossier de données d'entraînement ({interdits or images}). "
                 "Choisir par exemple /content/drive/MyDrive/poc_ct/predictions/totalsegmentator.")
    if os.path.basename(images) != "imagesTs":
        arret(1, f"--images doit être le dossier imagesTs (scanners de test), pas {images}.")

    with open(a.split, encoding="utf-8") as f:              # lecture seule du découpage figé
        split = json.load(f)
    cas_ts, cas_tr = split["test_sous_cle"], split["entrainement_validation_croisee"]
    if set(cas_ts) & set(cas_tr) or len(set(cas_ts)) != len(cas_ts) or len(cas_ts) != NB_CAS_TEST:
        arret(1, f"découpage incohérent (cas répété, présent dans les deux lots, ou test ≠ {NB_CAS_TEST} cas).")
    numeros = sorted(c.split("_")[-1] for c in cas_ts)       # ex. "ct_train_1004" → "1004"
    attendus = {f"mmwhs_{n}_0000.nii.gz" for n in numeros}
    presents = {os.path.basename(x) for x in glob.glob(os.path.join(images, "*.nii.gz"))}
    if presents != attendus:
        arret(1, f"imagesTs ne correspond pas au lot de test du découpage figé. "
                 f"Manquants : {sorted(attendus - presents)} ; en trop : {sorted(presents - attendus)}.")
    print(f"Lot de test vérifié : {len(attendus)} scanners = {numeros}")

    sorties = {n: os.path.join(sortie, NOM_SORTIE.format(n=n)) for n in numeros}
    chemin_trace = os.path.join(sortie, FICHIER_TRACE)
    deja = [x for x in list(sorties.values()) + [chemin_trace] if os.path.exists(x)]
    if deja and not a.ecraser:
        print(f"ARRÊT : des prédictions existent déjà ({len(deja)} fichiers dans {sortie}). Rien n'a été écrasé.")
        print("Relancer avec --ecraser pour les remplacer volontairement.")
        sys.exit(2)

    # --- 2. Processus, GPU, statistiques d'usage, licence ---------------------------------------
    # nnU-Net v2 lit nnUNet_def_n_proc UNE fois, à l'import de nnunetv2.configuration : il faut donc la
    # fixer avant d'importer totalsegmentator. Dans nnunetv2 2.8.1, elle règle le nombre de fils de calcul
    # (threads) de torch pendant la prédiction et l'export (défaut 8), pas le nombre de processus.
    # Les processus enfants héritent de cette variable.
    os.environ["nnUNet_def_n_proc"] = str(a.processus)
    import nibabel as nib
    import numpy as np
    import torch
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.config import (setup_totalseg, set_config_key, set_license_number,
                                         has_valid_license_offline, get_version)
    from totalsegmentator.map_to_binary import class_map, commercial_models

    if a.appareil == "gpu" and not torch.cuda.is_available():
        # Sans ce contrôle, TotalSegmentator passerait en silence sur le processeur (des heures).
        arret(5, "aucun GPU visible. Menu Exécution → Modifier le type d'exécution → GPU T4, puis relancer.")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "aucun"

    # Fichier de réglages de TotalSegmentator : ~/.totalsegmentator/config.json, dans la machine Colab
    # (effacé en fin de session, jamais dans Drive ni dans le dépôt).
    setup_totalseg()
    # Statistiques d'usage : TotalSegmentator les envoie par défaut après chaque prédiction, avec le numéro
    # de licence. On les coupe, sauf demande explicite. (TotalSegmentator 2.18.0 relit ce réglage dans
    # config.json juste avant chaque envoi : increase_prediction_counter() puis send_usage_stats().)
    set_config_key("send_usage_stats", bool(a.statistiques_usage))
    print(f"Statistiques d'usage envoyées à TotalSegmentator : {'OUI (demandé)' if a.statistiques_usage else 'non'}")

    licence_requise = a.tache in commercial_models
    if licence_requise:
        if cle:
            # Seul le préfixe est affiché, pour reconnaître le type de clé (ex. « aca_ » = académique).
            print(f"Clé de licence reçue par TOTALSEG_LICENCE : {cle[:4]}… ({len(cle)} caractères, masquée)")
            try:
                # Vérification en ligne par le serveur TotalSegmentator, puis mémorisation dans config.json.
                set_license_number(cle)
            except SystemExit:
                arret(4, "clé de licence refusée par TotalSegmentator (voir le message juste au-dessus).", cle)
        # On n'affiche JAMAIS le message de has_valid_license_offline() : il contient la clé en clair quand
        # elle est jugée invalide. Seul l'état est affiché.
        etat, _ = has_valid_license_offline()
        if etat != "yes":
            arret(4, f"la tâche {a.tache} demande une licence (état : {etat}). Licence gratuite pour un usage "
                     "non commercial : https://backend.totalsegmentator.com/license-academic/ . "
                     "Saisir ensuite la clé dans la cellule du notebook (secret Colab ou case masquée).", cle)

    # --- 3. Prédictions, un cas à la fois -------------------------------------------------------
    os.makedirs(sortie, exist_ok=True)
    if os.path.exists(chemin_trace):
        # --ecraser : l'ancienne traçabilité est retirée AVANT de commencer, pour qu'une relance qui échoue
        # (code 3) ne laisse pas croire que les prédictions présentes sont complètes et contrôlées.
        os.remove(chemin_trace)
    etiquettes = {int(k): v for k, v in class_map[a.tache].items()}
    # Nombre de processus (vérifié dans TotalSegmentator 2.18.0 et nnunetv2 2.8.1) :
    #  - nr_thr_resamp (défaut 1) est transmis à nnU-Net comme num_processes_preprocessing : nombre de
    #    processus enfants qui préparent le scanner (nnU-Net en lance au plus un par fichier, donc 1 ici) ;
    #  - nr_thr_saving (défaut 6 !) est transmis comme num_processes_segmentation_export : taille du
    #    groupe de processus enfants (mode spawn) qui remettent la prédiction à la taille du scanner et
    #    l'enregistrent. 6 enfants qui chargent chacun torch et une prédiction pleine résolution saturent
    #    les ~12 Go de Colab. On le ramène à --processus (1 par défaut).
    #  Le modèle de recadrage (3 mm) utilise déjà 1 processus d'enregistrement (fixé dans TotalSegmentator).
    options = {"task": a.tache, "ml": True, "device": a.appareil,
               "nr_thr_resamp": a.processus, "nr_thr_saving": a.processus}   # le reste : réglages par défaut
    resultats, debut_total = [], time.time()
    for n in numeros:
        entree = os.path.join(images, f"mmwhs_{n}_0000.nii.gz")
        print(f"\n=== mmwhs_{n} : prédiction en cours ===", flush=True)
        t0 = time.time()
        try:
            empreinte = sha256_fichier(entree)
            totalsegmentator(entree, sorties[n], **options)
            # Contrôle de la prédiction (ce n'est pas un score) : même grille de voxels que le scanner.
            scanner, pred = nib.load(entree), nib.load(sorties[n])
            meme_forme = scanner.shape[:3] == pred.shape[:3]
            meme_affine = bool(np.allclose(scanner.affine, pred.affine, atol=1e-3))
            presentes = sorted(int(v) for v in np.unique(np.asarray(pred.dataobj)) if v != 0)
        except (Exception, SystemExit) as err:   # SystemExit : TotalSegmentator s'arrête parfois par sys.exit
            print(f"\nERREUR PENDANT LA PRÉDICTION (mmwhs_{n}) : {sans_cle(repr(err), cle)}")
            print(f"Le dossier {sortie} est INCOMPLET. Corriger la cause, puis relancer avec --ecraser.")
            sys.exit(3)
        duree = round(time.time() - t0, 1)
        absentes = [etiquettes[k] for k in etiquettes if k not in presentes]
        resultats.append({"cas": f"mmwhs_{n}", "scanner_sha256": empreinte, "fichier": os.path.basename(sorties[n]),
                          "duree_s": duree, "meme_forme": meme_forme, "meme_affine": meme_affine,
                          "etiquettes_presentes": presentes, "structures_absentes": absentes})
        print(f"mmwhs_{n} : {duree} s ; forme identique au scanner : {'oui' if meme_forme else 'NON'} ; "
              f"affine identique : {'oui' if meme_affine else 'NON'} ; étiquettes {presentes}"
              + (f" ; ABSENTES : {absentes}" if absentes else ""))
        del scanner, pred

    # --- 4. Traçabilité (jamais la clé) ---------------------------------------------------------
    trace = {
        "etape": "4 — ligne de base TotalSegmentator (prédictions seulement, D19)",
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "totalsegmentator_version": get_version(), "nnunetv2_version": version_paquet("nnunetv2"),
        "torch_version": torch.__version__, "python_version": sys.version.split()[0], "gpu": gpu,
        "code_commit": commande("git", "rev-parse", "HEAD") or "inconnu",
        "code_modifie_localement": bool(commande("git", "status", "--porcelain")),
        "tache": a.tache, "options": options,
        "processus": {"nr_thr_resamp": a.processus, "nr_thr_saving": a.processus,
                      "nnUNet_def_n_proc": os.environ["nnUNet_def_n_proc"]},
        "licence": "requise, fournie par TOTALSEG_LICENCE (non enregistrée)" if licence_requise else "non requise",
        "statistiques_usage_envoyees": bool(a.statistiques_usage),
        "split": {"fichier": os.path.relpath(a.split, DEPOT), "sha256": sha256_fichier(a.split)},
        "images": images, "etiquettes_totalsegmentator": etiquettes,
        "correspondance_prevue_mmwhs_etape6": CORRESPONDANCE_MMWHS,
        "cas": resultats, "duree_totale_s": round(time.time() - debut_total, 1),
        "rappel": "Aucun masque de vérité terrain lu, aucun score calculé. Ces prédictions ne servent "
                  "jamais à l'entraînement (règle 5, licence).",
    }
    with open(chemin_trace, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    # --- 5. Récapitulatif -----------------------------------------------------------------------
    print(f"\nPrédictions écrites dans {sortie} ({len(resultats)} cas, {trace['duree_totale_s']} s au total).")
    print(f"Traçabilité : {chemin_trace} (TotalSegmentator {trace['totalsegmentator_version']}, "
          f"nnU-Net {trace['nnunetv2_version']}, GPU {gpu}, {a.processus} processus)")
    print("Aucun score n'a été calculé : la comparaison avec la vérité terrain se fera à l'étape 6, une seule fois.")
    echecs = [r["cas"] for r in resultats
              if not (r["meme_forme"] and r["meme_affine"]) or r["structures_absentes"]]
    if echecs:
        print(f"\nCONTRÔLE EN ÉCHEC pour {echecs} : géométrie différente du scanner ou structure absente "
              "(détail ci-dessus et dans la traçabilité). À examiner avant l'étape 6.")
        sys.exit(6)
    print("Contrôle des prédictions : CONFORME")


# Garde indispensable : les processus enfants (mode spawn) relisent ce fichier sous le nom
# « __mp_main__ » ; sans ce test, ils relanceraient tout le traitement.
if __name__ == "__main__":
    main()
