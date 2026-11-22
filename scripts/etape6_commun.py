"""
Étape 6 — Module commun aux scripts 06a, 06b, 06c, 06d (protocole figé docs/protocole_etape6.md, D42).

Ce que fait le module : il n'exécute RIEN à l'import (aucun fichier lu, aucun argument lu). Il apporte :
  1. les vérifications au démarrage du § 9 du protocole, à appeler AVANT toute lecture d'image ;
  2. la GARDE « jamais labelsTs hors phase C » : toute ouverture d'image, de masque ou de prédiction des
     scripts de l'étape 6 passe par verifier_chemin() (ou par ouvrir_nifti(), sha256(), lire_json(), qui
     l'appellent) ; un chemin interdit pour la phase en cours = arrêt immédiat, rien n'est lu ;
  3. les constantes de l'étape (empreintes de référence, 5 cas de test, dossiers Drive, fichiers de
     référence du dépôt) ;
  4. le MANIFESTE (phase B : écriture ; phase C : vérification) ;
  5. la traçabilité commune (commit git, versions, heures UTC et Paris, sortie recopiée sur Drive, verrou),
     reprise de 05a, 05b, 05c et 05l (mêmes fonctions, pas de nouvelle implémentation).

Entrées : uniquement les fichiers du dépôt (protocole, seuils, découpage, config/*.json) et les chemins que
  les scripts lui passent. Sorties : uniquement les fichiers que les scripts lui demandent d'écrire.

=========================================================================================================
INTERFACE PUBLIQUE (les scripts 06x n'ont besoin que de ce qui suit)
=========================================================================================================
Phases : "0bis", "A", "B", "C" (PHASES). Toute fonction qui lit un fichier de données prend la phase.

Arrêt
  arret(code, message)                     affiche « ARRÊT (code N) : ... » et quitte (sys.exit(code)).
  Codes communs (CODES) : 1 garde-fou général ; 10 protocole ; 11 seuils ; 12 quatre valeurs ; 13 découpage ;
  14 versions ; 15 modèles ; 16 empreintes de TotalSegmentator ; 17 manifeste ; 18 lecture interdite
  (garde) ; 19 écrasement refusé.

Vérifications du § 9
  verifier_demarrage(phase, depot=DEPOT, chemin_split=None) -> dict   points 1 à 4, toujours ; renvoie
      {protocole_sha256, seuils_sha256, split_sha256, cas_test, quatre_valeurs}.
  verifier_versions(monai=False, nnunet=False, archive_auto3dseg=None) -> dict      point 5.
  chemins_modeles(results, travail_a3d) -> dict     chemins par défaut des modèles (point 6).
  verifier_modeles(chemins, phase) -> dict          point 6 : présence + SHA-256 {nom: {chemin, sha256, taille_octets}}.
  verifier_empreintes_totalseg(dossier_predictions, phase, depot=DEPOT) -> dict      point 7 :
      compare au fichier du dépôt config/empreintes_totalseg_etape4.json (absent = arrêt) les 5 prédictions
      ET tracabilite_etape4.json (entrée « tracabilite_etape4 » de la référence) ; tout écart = arrêt 16 ;
      renvoie {"1004": {"chemin", "sha256"}, ...} (les 5 cas seulement).
  cas_test(chemin_split=None) -> tuple      les 5 cas de test lus dans le découpage et contrôlés.

Garde (lecture des données)
  verifier_chemin(chemin, phase) -> str     chemin absolu si autorisé, sinon arrêt (code 18).
  classer_chemin(chemin) -> str             "verite_test_brute" | "verite_test" | "image_test" | "cas_test" | "autre".
  autoriser_ouverture_test(chemin_registre) phase C : à appeler APRÈS verifier_manifeste() et APRÈS l'écriture
                                            du registre d'ouverture ; sans cet appel, labelsTs reste refusé.
  sha256(chemin, phase=None) -> str         SHA-256 en binaire, par blocs, sans décoder (passe par la garde ;
                                            phase=None : fichiers du dépôt ou de modèles seulement).
  ouvrir_nifti(chemin, phase)               image nibabel (nibabel.load) après la garde.
  lire_json(chemin, phase=None) -> objet    JSON lu après la garde.

Manifeste (phase B, puis C)
  ecrire_manifeste(chemin, modeles, predictions, controle, versions, reglages_inference, extra=None, depot=DEPOT) -> str
      vérifie le schéma, recalcule chaque empreinte, ajoute statut « figé », écrit de façon atomique,
      refuse tout écrasement par un contenu différent ; renvoie la SHA-256 du manifeste.
  contenu_reference_manifeste(chemin, sha) -> dict  contenu du futur config/empreinte_manifeste_etape6.json.
  verifier_manifeste(chemin, depot=DEPOT) -> dict   phase C : compare au fichier du dépôt, puis recalcule
                                            l'empreinte de chaque fichier listé ; le moindre écart = arrêt.

Traçabilité
  demarrer_sortie() -> journal              à appeler en tout premier : l'affichage est gardé en mémoire ;
  ouvrir_sortie(journal, dossier, script) -> chemin   ... puis écrit dans <dossier>/sortie_<script>_<SESSION>.txt
                                            (à appeler APRÈS les vérifications : rien n'est écrit si elles échouent).
  poser_verrou(dossier, nom, ignorer=False) verrou de 05c/05l (une seule session à la fois), retiré à la fin.
  heure() -> str                            « 16/11 14:52:10 UTC (16/11 15:52:10 Paris) ».
  maintenant() -> dict                      {"utc": ISO, "paris": ISO}.
  commit_git() -> str                       commit du dépôt (« +modifié » si des fichiers ont changé).
  versions_bibliotheques() -> dict          Python, PyTorch, numpy, nibabel, scipy, MONAI, nnU-Net, TotalSegmentator.
  tracabilite(phase, script) -> dict        bloc commun : commit, session, heures, versions, empreintes de référence.
  ecrire_json_atomique(chemin, contenu)     fichier temporaire puis renommage (jamais de fichier à moitié écrit).
  dans_depot(chemin) -> bool ; verifier_hors_depot(chemin, option)   les données restent hors du dépôt.

Constantes : SHA256_PROTOCOLE, SHA256_SEUILS, SHA256_SPLIT, CAS_TEST_ATTENDUS, DEPOT, DRIVE, DOSSIER_ETAPE6,
  DOSSIERS_PHASES, PREDICTIONS_TOTALSEG, REF_EMPREINTES_TOTALSEG, REF_MANIFESTE, NOM_MANIFESTE,
  MODELES_PREDICTION ("A", "N0", "S0", "T"), VERSION_MONAI, VERSION_NNUNET, SHA256_ARCHIVE_AUTO3DSEG.

Règles de la garde (protocole § 10.1, cahier des charges « Règles de lecture du test ») :
  - « verite_test » (un dossier labelsTs dans le chemin, ou un masque d'un cas de test dans labelsTr) :
    phase C seulement, et seulement après autoriser_ouverture_test() ;
  - « verite_test_brute » (masque d'origine MM-WHS d'un cas de test, ct_train_1004_label...) : JAMAIS
    (R-O1 : la vérité du test n'est lue que dans labelsTs, par le seul script d'évaluation) ;
  - « image_test » (dossier imagesTs, ou scanner d'origine d'un cas de test) : phases B et C ;
  - « cas_test » (autre fichier dont le nom désigne un cas de test : prédictions) : phases 0bis, B, C ;
    jamais en phase A, qui ne travaille que sur les cas d'entraînement ;
  - « autre » : toutes les phases. Phase None (fichiers du dépôt, modèles) : « autre » seulement.
  Le chemin est examiné tel qu'écrit ET après résolution des liens (os.path.realpath).

Vocabulaire :
  - empreinte SHA-256 : 64 caractères calculés sur tous les octets d'un fichier ; un seul octet changé
    donne une autre empreinte. Elle prouve qu'un fichier n'a pas changé ;
  - manifeste : liste figée des fichiers (prédictions, résultats du contrôle, modèles) avec leur empreinte,
    écrite avant l'ouverture du test (phase B) et vérifiée avant de lire la vérité (phase C) ;
  - écriture atomique : on écrit un fichier temporaire, puis on le renomme d'un seul coup ;
  - verrou : petit fichier qui empêche deux sessions Colab de travailler en même temps au même endroit.
"""
import datetime, hashlib, importlib.util, json, os, re, sys
from importlib.metadata import version as _version, PackageNotFoundError

ICI = os.path.dirname(os.path.abspath(__file__))


def _charger(nom, fichier):
    spec = importlib.util.spec_from_file_location(nom, os.path.join(ICI, fichier))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)          # 05l (et 05k, 05c, 05b, 05a qu'il apporte) n'exécute rien à l'import
    return module


l05 = _charger("essai05l", "05l_essai_segresnet.py")   # verrou ; apporte 05c (heures, sortie) et 05a (commit)
suivi05c, prep = l05.suivi05c, l05.prep

# --- Constantes de référence (protocole § 2 et § 9 ; D36, D42) -------------------------------------
SHA256_PROTOCOLE = "ade98f69f6587035fc17f5ab96bc208bbdcfab50812e5fa3317f8b4958dab5d3"   # D42
SHA256_SEUILS = "c1b247070765d193a46e90b538cc655787dbb65f8fdd13f113a39efd3ef90672"      # D36
SHA256_SPLIT = "25881b214823758ddb708b6b46616c2422671c872035683852b1f045f4c4a05c"       # D20 (relevé par 05j)
CAS_TEST_ATTENDUS = ("1004", "1010", "1011", "1013", "1014")                             # § 3.1, D20
VERSION_MONAI, VERSION_NNUNET = "1.6.1", "2.8.1"                                         # § 5.2, § 9 point 5
SHA256_ARCHIVE_AUTO3DSEG = l05.SHA256_MODELES                                            # archive 21ed8e5 (D37)
HASH_MODELES_AUTO3DSEG = l05.HASH_MODELES
STATUT_FIGE = "figé"
PHASES = ("0bis", "A", "B", "C")
MODELES_PREDICTION = ("A", "N0", "S0", "T")                                              # § 3.3

# --- Fichiers du dépôt ---------------------------------------------------------------------------
DEPOT = prep.DEPOT
FICHIER_PROTOCOLE = os.path.join("docs", "protocole_etape6.md")
FICHIER_SEUILS = os.path.join("config", "seuils_plausibilite.json")
FICHIER_SPLIT = os.path.join("splits", "mmwhs_split.json")
# Nom gardé (« etape4 ») même après D44 : il désigne la prédiction T de l'étape 4 ; son contenu portera sur les
# prédictions refaites selon D44 (dossier PREDICTIONS_TOTALSEG ci-dessous).
REF_EMPREINTES_TOTALSEG = os.path.join("config", "empreintes_totalseg_etape4.json")     # écrit à la main après 0 bis
REF_MANIFESTE = os.path.join("config", "empreinte_manifeste_etape6.json")               # écrit à la main après B

# --- Dossiers Drive (chemins par défaut ; les scripts les reçoivent en paramètres) -----------------
DRIVE = prep.DRIVE                                                     # /content/drive/MyDrive/poc_ct
DOSSIER_ETAPE6 = f"{DRIVE}/etape6"
DOSSIERS_PHASES = {"0bis": f"{DOSSIER_ETAPE6}/phase0bis", "A": f"{DOSSIER_ETAPE6}/phaseA",
                   "B": f"{DOSSIER_ETAPE6}/phaseB", "C": f"{DOSSIER_ETAPE6}/phaseC"}
# D44 : les prédictions de l'étape 4 (predictions/totalsegmentator, section 10) ont été perdues (Drive plein,
# seul mmwhs_1004 a survécu). Elles sont refaites, mêmes version, tâche et réglages, par 04c (section 26 bis) dans
# ce NOUVEAU dossier ; l'ancien dossier n'est plus lu par l'étape 6 (il garde l'ancien 1004, conservé à part).
PREDICTIONS_TOTALSEG = f"{DRIVE}/predictions/totalsegmentator_d44"     # --sortie de 04c (D44)
NOM_PREDICTION = "mmwhs_{cas}.nii.gz"                                  # nom des prédictions (04, 04c, nnU-Net, 05m)
NOM_TRACE_ETAPE4 = "tracabilite_etape4.json"                           # même nom écrit par 04c (D44)
NOM_EMPREINTES_0BIS = "empreintes_totalseg.json"
NOM_MANIFESTE = "manifeste_etape6.json"
NOM_REGISTRE = "registre_ouverture.json"

CODES = {"garde_fou": 1, "protocole": 10, "seuils": 11, "quatre_valeurs": 12, "decoupage": 13, "versions": 14,
         "modeles": 15, "totalseg": 16, "manifeste": 17, "lecture_interdite": 18, "ecrasement": 19}

_ETAT = {"manifeste_verifie": None, "ouverture_autorisee": False}     # garde de la phase C (même processus)


# =================================================================================================
# Utilitaires
# =================================================================================================
def arret(code, message):
    """Arrêt propre : message en clair, code de sortie connu. Rien d'autre n'est lu ni écrit."""
    print(f"ARRÊT (code {code}) : {message}", flush=True)
    sys.exit(code)


def _sha256_brut(chemin):
    """SHA-256 des octets du fichier, lu en binaire par blocs de 1 Mo (aucun décodage)."""
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def dans_depot(chemin):
    return prep.dans(os.path.abspath(chemin), DEPOT)


def verifier_hors_depot(chemin, option):
    """Les données et résultats restent dans Drive : un dossier de sortie dans le dépôt = arrêt."""
    if dans_depot(chemin):
        arret(CODES["garde_fou"], f"{option} ({os.path.abspath(chemin)}) est dans le dépôt : rien n'est versionné.")
    return os.path.abspath(chemin)


def _phase_valide(phase, permettre_none=False):
    if phase is None and permettre_none:
        return
    if phase not in PHASES:
        arret(CODES["garde_fou"], f"phase inconnue {phase!r} (attendu : {', '.join(PHASES)}).")


# =================================================================================================
# Garde « jamais labelsTs hors phase C »
# =================================================================================================
_RE_CAS_TEST = re.compile(r"(?:mmwhs|ct_train)_(" + "|".join(CAS_TEST_ATTENDUS) + r")(?!\d)", re.I)
_RE_BRUT_LABEL = re.compile(r"ct_train_(" + "|".join(CAS_TEST_ATTENDUS) + r")_label", re.I)
_RE_BRUT_IMAGE = re.compile(r"ct_train_(" + "|".join(CAS_TEST_ATTENDUS) + r")_image|mmwhs_("
                            + "|".join(CAS_TEST_ATTENDUS) + r")_0000", re.I)       # scanner de test, même copié ailleurs


def _composants(chemin):
    return [c.lower() for c in re.split(r"[\\/]+", chemin) if c]


def classer_chemin(chemin):
    """Catégorie d'un chemin (voir l'en-tête). Examine le chemin écrit ET le chemin résolu (liens)."""
    pire = "autre"
    ordre = ("autre", "cas_test", "image_test", "verite_test", "verite_test_brute")
    for c in {os.path.abspath(chemin), os.path.realpath(chemin)}:
        parties, nom = _composants(c), os.path.basename(c)
        if _RE_BRUT_LABEL.search(nom):
            classe = "verite_test_brute"
        elif "labelsts" in parties or ("labelstr" in parties and _RE_CAS_TEST.search(nom)):
            classe = "verite_test"
        elif "imagests" in parties or _RE_BRUT_IMAGE.search(nom):
            classe = "image_test"
        elif _RE_CAS_TEST.search(nom):
            classe = "cas_test"
        else:
            classe = "autre"
        pire = max(pire, classe, key=ordre.index)
    return pire


_AUTORISE = {"autre": {None, "0bis", "A", "B", "C"}, "cas_test": {"0bis", "B", "C"},
             "image_test": {"B", "C"}, "verite_test": {"C"}, "verite_test_brute": set()}


def verifier_chemin(chemin, phase):
    """Passage obligé avant toute ouverture d'image, de masque ou de prédiction. Arrêt (code 18) si interdit."""
    _phase_valide(phase, permettre_none=True)
    classe = classer_chemin(chemin)
    if phase not in _AUTORISE[classe]:
        arret(CODES["lecture_interdite"], f"lecture REFUSÉE en phase {phase} : {chemin} est classé « {classe} ». "
              "La vérité du test n'est lue qu'en phase C (R-O1), imagesTs qu'en phases B et C. Rien n'a été lu.")
    if classe == "verite_test" and not _ETAT["ouverture_autorisee"]:
        arret(CODES["lecture_interdite"], f"lecture REFUSÉE : {chemin} (vérité du test) avant vérification du "
              "manifeste et écriture du registre d'ouverture (autoriser_ouverture_test). Rien n'a été lu.")
    return os.path.abspath(chemin)


def autoriser_ouverture_test(chemin_registre):
    """Phase C : ouvre la garde de labelsTs, seulement si le manifeste a été vérifié dans ce processus et si
    le registre d'ouverture existe déjà sur Drive (§ 10.1 R-O1, cahier des charges phase C)."""
    if not _ETAT["manifeste_verifie"]:
        arret(CODES["lecture_interdite"], "ouverture refusée : le manifeste n'a pas été vérifié (verifier_manifeste).")
    if not os.path.isfile(chemin_registre):
        arret(CODES["lecture_interdite"], f"ouverture refusée : registre d'ouverture absent ({chemin_registre}).")
    _ETAT["ouverture_autorisee"] = True


def sha256(chemin, phase=None):
    """SHA-256 d'un fichier, en binaire, sans décodage, après la garde."""
    return _sha256_brut(verifier_chemin(chemin, phase))


def ouvrir_nifti(chemin, phase):
    """Image NIfTI ouverte avec nibabel, après la garde (nibabel lit les voxels plus tard, à la demande)."""
    import nibabel as nib
    return nib.load(verifier_chemin(chemin, phase))


def lire_json(chemin, phase=None):
    with open(verifier_chemin(chemin, phase), encoding="utf-8") as f:
        return json.load(f)


# =================================================================================================
# Vérifications au démarrage (§ 9)
# =================================================================================================
def _fichier_depot(depot, relatif, code):
    chemin = os.path.join(depot, relatif)
    if not os.path.isfile(chemin):
        arret(code, f"{relatif} introuvable dans le dépôt ({chemin}).")
    return chemin


def cas_test(chemin_split=None):
    """Les 5 cas de test lus dans le découpage figé, contrôlés contre CAS_TEST_ATTENDUS (point 4)."""
    chemin = chemin_split or os.path.join(DEPOT, FICHIER_SPLIT)
    with open(chemin, encoding="utf-8") as f:
        dec = json.load(f)
    test = [str(c) for c in dec.get("test_sous_cle", [])]
    numeros = tuple(sorted(c.rsplit("_", 1)[-1] for c in test))
    entr = {c.rsplit("_", 1)[-1] for c in dec.get("entrainement_validation_croisee", [])}
    if len(test) != len(set(test)) or numeros != CAS_TEST_ATTENDUS or not all(re.fullmatch(r"ct_train_\d{4}", c) for c in test):
        arret(CODES["decoupage"], f"lot de test du découpage = {test}, attendu exactement {list(CAS_TEST_ATTENDUS)}.")
    if entr & set(numeros):
        arret(CODES["decoupage"], f"cas à la fois en test et en entraînement : {sorted(entr & set(numeros))}.")
    return numeros


def _connexite_morceaux(pp):
    """Connexité RÉELLEMENT utilisée par morceaux() de 05i, mesurée par le code sur de petits masques :
    deux voxels qui ne se touchent que par un coin = 1 morceau en connexité 26 (2 en 6 ou 18) ;
    deux voxels qui ne se touchent que par une arête = 1 morceau en 18 et 26 (2 en 6) ;
    deux voxels séparés d'un voxel vide = toujours 2 morceaux."""
    import numpy as np
    m = np.zeros((3, 3, 3), bool)
    m[0, 0, 0] = m[1, 1, 1] = True
    coin = pp.morceaux(m, 1.0)["n"]
    m = np.zeros((3, 3, 3), bool)
    m[0, 0, 0] = m[1, 1, 0] = True
    arete = pp.morceaux(m, 1.0)["n"]
    m = np.zeros((3, 3, 3), bool)
    m[0, 0, 0] = m[2, 0, 0] = True
    ecart = pp.morceaux(m, 1.0)["n"]
    if ecart != 2:
        return None
    return 26 if coin == 1 else (18 if arete == 1 else 6)


def _seuil_morceau_mesure(pp, valeur):
    """Seuil de taille RÉELLEMENT appliqué par morceaux() : un morceau de 1 voxel dont le volume vaut
    exactement la valeur n'est pas compté (« plus de »), un voxel à peine plus gros l'est."""
    import numpy as np
    m = np.zeros((3, 3, 3), bool)
    m[1, 1, 1] = True
    return pp.morceaux(m, valeur)["n_gros"] == 0 and pp.morceaux(m, valeur * 1.000001)["n_gros"] == 1


def verifier_demarrage(phase, depot=DEPOT, chemin_split=None):
    """Points 1 à 4 du § 9, dans l'ordre. Lit UNIQUEMENT des fichiers du dépôt. Arrêt au premier échec.
    Renvoie {"protocole_sha256", "seuils_sha256", "split_sha256", "cas_test", "quatre_valeurs"}."""
    _phase_valide(phase)
    print(f"Vérifications au démarrage (protocole § 9, points 1 à 4), phase {phase} :")
    # 1. Protocole figé
    s = _sha256_brut(_fichier_depot(depot, FICHIER_PROTOCOLE, CODES["protocole"]))
    if s != SHA256_PROTOCOLE:
        arret(CODES["protocole"], f"empreinte de {FICHIER_PROTOCOLE} = {s}, attendu {SHA256_PROTOCOLE} (D42). "
              "Le protocole a changé (ou ses fins de ligne) : ne rien lancer, en parler.")
    print(f"  1. protocole figé : SHA-256 conforme ({s[:8]}…{s[-6:]})")
    # 2. Fichier de seuils figé
    chemin_seuils = _fichier_depot(depot, FICHIER_SEUILS, CODES["seuils"])
    s2 = _sha256_brut(chemin_seuils)
    if s2 != SHA256_SEUILS:
        arret(CODES["seuils"], f"empreinte de {FICHIER_SEUILS} = {s2}, attendu {SHA256_SEUILS} (D36).")
    with open(chemin_seuils, encoding="utf-8") as f:
        seuils = json.load(f)
    if seuils.get("statut") != STATUT_FIGE:
        arret(CODES["seuils"], f"{FICHIER_SEUILS} : statut = {seuils.get('statut')!r}, attendu « {STATUT_FIGE} ».")
    print(f"  2. seuils : SHA-256 conforme ({s2[:8]}…{s2[-6:]}), statut « {STATUT_FIGE} »")
    # 3. Les quatre valeurs non relues par le code
    cp = _charger("controle_plausibilite", "controle_plausibilite.py")
    pp = cp.pp                                        # le module 05i tel que le contrôle l'utilise
    r0, r8 = seuils.get("R0", {}), seuils.get("R8", {})
    try:
        connexite = _connexite_morceaux(pp)
    except ImportError as err:
        arret(CODES["quatre_valeurs"], f"numpy et scipy sont nécessaires pour vérifier la connexité ({err}).")
    valeurs = {
        "tolerance_affine_mm": (r0.get("tolerance_affine_mm"), cp.TOLERANCE_R0_MM),
        "etiquettes_autorisees": (r0.get("etiquettes_autorisees"), list(cp.ETIQUETTES_AUTORISEES)),
        "morceau_min_ml": (r8.get("morceau_min_ml"), pp.MORCEAU_ML),
        "connexite": (r8.get("connexite"), connexite)}
    ecarts = [f"{k} : fichier figé {f!r}, code {c!r}" for k, (f, c) in valeurs.items() if f != c]
    if cp.MORCEAU_ML != pp.MORCEAU_ML:
        ecarts.append(f"MORCEAU_ML du module de contrôle ({cp.MORCEAU_ML}) ≠ 05i ({pp.MORCEAU_ML})")
    if not _seuil_morceau_mesure(pp, r8.get("morceau_min_ml") or 0):
        ecarts.append("morceaux() de 05i n'applique pas « plus de morceau_min_ml » (mesuré sur un masque test)")
    if ecarts:
        arret(CODES["quatre_valeurs"], "valeurs du fichier figé différentes du code :\n  - " + "\n  - ".join(ecarts))
    print("  3. quatre valeurs identiques entre le fichier figé et le code : tolérance R0 "
          f"{cp.TOLERANCE_R0_MM} mm, étiquettes {list(cp.ETIQUETTES_AUTORISEES)}, morceau > {pp.MORCEAU_ML} mL, "
          f"connexité {connexite} (mesurée en appelant morceaux() de 05i)")
    # 4. Découpage
    chemin_split = chemin_split or _fichier_depot(depot, FICHIER_SPLIT, CODES["decoupage"])
    s4 = _sha256_brut(chemin_split)
    if s4 != SHA256_SPLIT:
        arret(CODES["decoupage"], f"empreinte de {chemin_split} = {s4}, attendu {SHA256_SPLIT} (D20).")
    test = cas_test(chemin_split)
    print(f"  4. découpage : SHA-256 conforme ({s4[:8]}…{s4[-6:]}) ; lot de test = {', '.join(test)}")
    return {"protocole_sha256": s, "seuils_sha256": s2, "split_sha256": s4, "cas_test": list(test),
            "quatre_valeurs": {k: c for k, (f, c) in valeurs.items()}}


def verifier_versions(monai=False, nnunet=False, archive_auto3dseg=None):
    """Point 5 : MONAI 1.6.1, nnU-Net v2 2.8.1, archive des modèles Auto3DSeg 21ed8e5 (SHA-256 de D37),
    selon ce que la phase utilise. Renvoie les versions lues."""
    lues, pb = versions_bibliotheques(), []
    if monai and lues.get("monai") != VERSION_MONAI:
        pb.append(f"MONAI {lues.get('monai')}, attendu {VERSION_MONAI}")
    if nnunet and lues.get("nnunetv2") != VERSION_NNUNET:
        pb.append(f"nnU-Net v2 {lues.get('nnunetv2')}, attendu {VERSION_NNUNET}")
    if archive_auto3dseg is not None:
        if not os.path.isfile(archive_auto3dseg):
            pb.append(f"archive des modèles Auto3DSeg absente ({archive_auto3dseg})")
        else:
            s = sha256(archive_auto3dseg, None)
            lues["sha256_archive_auto3dseg"] = s
            if s != SHA256_ARCHIVE_AUTO3DSEG:
                pb.append(f"archive Auto3DSeg : SHA-256 {s}, attendu {SHA256_ARCHIVE_AUTO3DSEG} ({HASH_MODELES_AUTO3DSEG})")
    if pb:
        arret(CODES["versions"], "versions non conformes (protocole § 9, point 5) :\n  - " + "\n  - ".join(pb))
    print("  5. versions conformes : " + ", ".join(
        x for x, ok in ((f"MONAI {lues.get('monai')}", monai), (f"nnU-Net {lues.get('nnunetv2')}", nnunet),
                        (f"archive Auto3DSeg {HASH_MODELES_AUTO3DSEG}", archive_auto3dseg is not None)) if ok))
    return lues


def chemins_modeles(results, travail_a3d):
    """Chemins des modèles du § 3.3 (point 6), d'après 05c, 05i et 05m."""
    pp = _charger("postraitement05i_chemins", "05i_postraitement_nnunet.py")
    d = pp.dossier_modele(results)
    chemins = {f"nnunet_pli_{k}": os.path.join(d, f"fold_{k}", "checkpoint_final.pth") for k in pp.PLIS}
    chemins["nnunet_postprocessing"] = os.path.join(d, pp.NOM_CV, "postprocessing.pkl")
    chemins["segresnet_pli_0"] = os.path.join(travail_a3d, "entrainement_segresnet_1250", "pli_0", "segresnet_0",
                                              "model", "model_final.pt")
    return chemins


def verifier_modeles(chemins, phase):
    """Point 6 : chaque fichier de modèle présent ; empreintes relevées. Renvoie {nom: {chemin, sha256, taille_octets}}."""
    manquants = [f"{n} : {c}" for n, c in chemins.items() if not os.path.isfile(c)]
    if manquants:
        arret(CODES["modeles"], "modèles absents (entraînement non terminé ?) :\n  - " + "\n  - ".join(manquants))
    releves = {}
    for nom, c in chemins.items():
        releves[nom] = {"chemin": os.path.abspath(c), "sha256": sha256(c, phase), "taille_octets": os.path.getsize(c)}
        print(f"  6. {nom} : {releves[nom]['sha256']}")
    return releves


def verifier_empreintes_totalseg(dossier_predictions, phase, depot=DEPOT):
    """Point 7 : les 5 prédictions de TotalSegmentator ET leur fichier de traçabilité (tracabilite_etape4.json)
    doivent avoir l'empreinte inscrite dans le dépôt (config/empreintes_totalseg_etape4.json, recopié depuis la
    phase 0 bis). Depuis D44, ce sont les prédictions refaites par 04c (dossier PREDICTIONS_TOTALSEG).
    Fichier de référence absent, entrée absente ou empreinte différente = arrêt (code 16).
    Renvoie {cas: {"chemin", "sha256"}} pour les 5 cas."""
    ref_chemin = os.path.join(depot, REF_EMPREINTES_TOTALSEG)
    if not os.path.isfile(ref_chemin):
        arret(CODES["totalseg"], f"{REF_EMPREINTES_TOTALSEG} absent du dépôt : faire la phase 0 bis (06a), recopier "
              "son bloc dans ce fichier et dans le journal, puis relancer.")
    with open(ref_chemin, encoding="utf-8") as f:
        ref = json.load(f)
    fichiers = ref.get("fichiers", {})
    if sorted(fichiers) != list(CAS_TEST_ATTENDUS):
        arret(CODES["totalseg"], f"{REF_EMPREINTES_TOTALSEG} : cas {sorted(fichiers)}, attendu {list(CAS_TEST_ATTENDUS)}.")
    pb, sortie = [], {}
    for cas in CAS_TEST_ATTENDUS:
        nom = NOM_PREDICTION.format(cas=cas)
        attendu = fichiers[cas]
        chemin = os.path.join(dossier_predictions, nom)
        if attendu.get("nom") != nom or not re.fullmatch(r"[0-9a-f]{64}", str(attendu.get("sha256", ""))):
            pb.append(f"{cas} : entrée de référence mal formée ({attendu})")
        elif not os.path.isfile(chemin):
            pb.append(f"{cas} : {chemin} absent")
        else:
            s = sha256(chemin, phase)
            sortie[cas] = {"chemin": os.path.abspath(chemin), "sha256": s}
            if s != attendu["sha256"]:
                pb.append(f"{cas} : SHA-256 {s} ≠ référence {attendu['sha256']}")
    # Traçabilité de l'étape 4 (tâche, version de TotalSegmentator) : relevée en 0 bis, elle ne doit pas changer.
    attendu, chemin = ref.get("tracabilite_etape4") or {}, os.path.join(dossier_predictions, NOM_TRACE_ETAPE4)
    if attendu.get("nom") != NOM_TRACE_ETAPE4 or not re.fullmatch(r"[0-9a-f]{64}", str(attendu.get("sha256", ""))):
        pb.append(f"{NOM_TRACE_ETAPE4} : entrée « tracabilite_etape4 » absente ou mal formée dans la référence ({attendu})")
    elif not os.path.isfile(chemin):
        pb.append(f"{NOM_TRACE_ETAPE4} : {chemin} absent")
    else:
        s = sha256(chemin, phase)
        if s != attendu["sha256"]:
            pb.append(f"{NOM_TRACE_ETAPE4} : SHA-256 {s} ≠ référence {attendu['sha256']}")
    if pb:
        arret(CODES["totalseg"], "prédictions de TotalSegmentator (ou leur traçabilité) différentes de la phase 0 bis :\n  - " + "\n  - ".join(pb))
    print(f"  7. prédictions de TotalSegmentator : 5 empreintes (et {NOM_TRACE_ETAPE4}) identiques à {REF_EMPREINTES_TOTALSEG}")
    return sortie


# =================================================================================================
# Manifeste (§ 10.2, phase B point 5 ; phase C point 1)
# =================================================================================================
def _serialiser(contenu):
    return (json.dumps(contenu, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")


def ecrire_json_atomique(chemin, contenu):
    """Écrit le JSON (UTF-8, clés triées) dans un fichier temporaire, puis le renomme d'un seul coup."""
    os.makedirs(os.path.dirname(os.path.abspath(chemin)), exist_ok=True)
    tmp = chemin + ".tmp"
    with open(tmp, "wb") as f:
        f.write(_serialiser(contenu))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, chemin)


def _fichiers_listes(m):
    """(rubrique, clé, entrée) pour chaque fichier listé dans le manifeste."""
    for nom, e in m["modeles"].items():
        yield "modeles", nom, e
    for mod in MODELES_PREDICTION:
        for cas in CAS_TEST_ATTENDUS:
            yield "predictions", f"{mod}/{cas}", m["predictions"][mod][cas]
        yield "controle", mod, m["controle"][mod]


def _controler_schema(m):
    pb = []
    if not isinstance(m.get("modeles"), dict) or not m["modeles"]:
        pb.append("modeles : dictionnaire non vide attendu {nom: {chemin, sha256}}")
    p, c = m.get("predictions"), m.get("controle")
    if not isinstance(p, dict) or sorted(p) != sorted(MODELES_PREDICTION):
        pb.append(f"predictions : clés {sorted(p) if isinstance(p, dict) else p}, attendu {list(MODELES_PREDICTION)}")
    else:
        pb += [f"predictions[{k}] : cas {sorted(v)}, attendu {list(CAS_TEST_ATTENDUS)}"
               for k, v in p.items() if not isinstance(v, dict) or sorted(v) != list(CAS_TEST_ATTENDUS)]
    if not isinstance(c, dict) or sorted(c) != sorted(MODELES_PREDICTION):
        pb.append(f"controle : clés {sorted(c) if isinstance(c, dict) else c}, attendu {list(MODELES_PREDICTION)}")
    if not pb:
        pb += [f"{r}[{k}] : entrée sans chemin ({e})" for r, k, e in _fichiers_listes(m)
               if not isinstance(e, dict) or not e.get("chemin")]
    return pb


def ecrire_manifeste(chemin, modeles, predictions, controle, versions, reglages_inference, extra=None, depot=DEPOT):
    """Phase B : écrit le manifeste figé et renvoie sa SHA-256.
    modeles = {nom: {"chemin", "sha256"?}} ; predictions = {"A"|"N0"|"S0"|"T": {"1004": {"chemin", "sha256"?}, ...}} ;
    controle = {"A"|"N0"|"S0"|"T": {"chemin", "sha256"?}}. Chaque empreinte est RECALCULÉE ici (lecture
    binaire) ; une empreinte fournie différente = arrêt. Refuse d'écraser un manifeste différent, et refuse
    tout manifeste différent de celui déjà inscrit dans le dépôt (config/empreinte_manifeste_etape6.json)."""
    if os.path.exists(chemin):
        arret(CODES["ecrasement"], f"{chemin} existe déjà : un manifeste figé n'est jamais réécrit. S'il est complet, "
              "relever son empreinte ; sinon en parler (déviation éventuelle, § 11).")
    if os.path.isfile(os.path.join(depot, REF_MANIFESTE)):
        arret(CODES["ecrasement"], f"{REF_MANIFESTE} existe déjà dans le dépôt : le manifeste est figé, on ne le refait pas.")
    m = {"objet": "manifeste de l'étape 6 (protocole § 10.2, phase B, point 5)",
         "protocole_sha256": SHA256_PROTOCOLE, "seuils_sha256": SHA256_SEUILS, "split_sha256": SHA256_SPLIT,
         "commit": commit_git(), "date_utc": maintenant()["utc"], "versions": versions,
         "reglages_inference": reglages_inference, "cas_test": list(CAS_TEST_ATTENDUS),
         "modeles": modeles, "predictions": predictions, "controle": controle, "statut": STATUT_FIGE}
    pb = [f"extra : la clé {k!r} est réservée" for k in (extra or {}) if k in m]
    m.update({k: v for k, v in (extra or {}).items() if k not in m})
    pb += _controler_schema(m)
    if pb:
        arret(CODES["manifeste"], "manifeste mal formé :\n  - " + "\n  - ".join(pb))
    ecarts = []
    for rubrique, cle, e in _fichiers_listes(m):
        s = sha256(e["chemin"], "B")
        if e.get("sha256") not in (None, s):
            ecarts.append(f"{rubrique}[{cle}] : empreinte fournie {e['sha256']} ≠ fichier {s}")
        e["chemin"], e["sha256"] = os.path.abspath(e["chemin"]), s
    if ecarts:
        arret(CODES["manifeste"], "le fichier a changé depuis le relevé de son empreinte :\n  - " + "\n  - ".join(ecarts))
    ecrire_json_atomique(chemin, m)
    s = _sha256_brut(chemin)
    print(f"Manifeste écrit (statut « {STATUT_FIGE} ») : {chemin}\n  SHA-256 du manifeste : {s}")
    return s


def contenu_reference_manifeste(chemin, sha):
    """Contenu du futur config/empreinte_manifeste_etape6.json (recopié à la main dans le dépôt après B)."""
    return {"objet": "empreinte du manifeste de l'étape 6 (protocole § 10.2), vérifiée au démarrage de la phase C",
            "manifeste_chemin_drive": os.path.abspath(chemin), "manifeste_sha256": sha,
            "protocole_sha256": SHA256_PROTOCOLE, "statut": STATUT_FIGE}


def verifier_manifeste(chemin, depot=DEPOT):
    """Phase C, AVANT toute lecture de labelsTs : 1) fichier de référence du dépôt présent ; 2) SHA-256 du
    manifeste identique ; 3) statut figé et empreintes de référence ; 4) empreinte de CHAQUE fichier listé
    recalculée. Le moindre écart = arrêt (code 17). Renvoie le manifeste lu."""
    ref_chemin = os.path.join(depot, REF_MANIFESTE)
    if not os.path.isfile(ref_chemin):
        arret(CODES["manifeste"], f"{REF_MANIFESTE} absent du dépôt : la phase C ne démarre pas (§ 10.2).")
    with open(ref_chemin, encoding="utf-8") as f:
        ref = json.load(f)
    if not os.path.isfile(chemin):
        arret(CODES["manifeste"], f"manifeste introuvable : {chemin}.")
    s = sha256(chemin, "C")
    if s != ref.get("manifeste_sha256"):
        arret(CODES["manifeste"], f"SHA-256 du manifeste {s} ≠ référence du dépôt {ref.get('manifeste_sha256')}.")
    m = lire_json(chemin, "C")
    pb = _controler_schema(m)
    for cle, attendu in (("statut", STATUT_FIGE), ("protocole_sha256", SHA256_PROTOCOLE),
                         ("seuils_sha256", SHA256_SEUILS), ("split_sha256", SHA256_SPLIT)):
        if m.get(cle) != attendu:
            pb.append(f"{cle} = {m.get(cle)!r}, attendu {attendu!r}")
    if pb:
        arret(CODES["manifeste"], "manifeste non conforme :\n  - " + "\n  - ".join(pb))
    ecarts = []
    for rubrique, cle, e in _fichiers_listes(m):
        if not os.path.isfile(e["chemin"]):
            ecarts.append(f"{rubrique}[{cle}] : fichier absent ({e['chemin']})")
            continue
        s2 = sha256(e["chemin"], "C")
        if s2 != e.get("sha256"):
            ecarts.append(f"{rubrique}[{cle}] : {s2} ≠ manifeste {e.get('sha256')}")
    if ecarts:
        arret(CODES["manifeste"], "fichiers modifiés depuis la phase B :\n  - " + "\n  - ".join(ecarts))
    n = sum(1 for _ in _fichiers_listes(m))
    print(f"Manifeste vérifié : SHA-256 conforme au dépôt ({s[:8]}…{s[-6:]}), {n} fichiers aux empreintes identiques.")
    _ETAT["manifeste_verifie"] = s
    return m


# =================================================================================================
# Traçabilité (reprise de 05a, 05b, 05c, 05l)
# =================================================================================================
heure = suivi05c.heure
commit_git = prep.commit_git
SESSION = suivi05c.SESSION


def maintenant():
    t = datetime.datetime.now(datetime.timezone.utc)
    return {"utc": t.isoformat(timespec="seconds"), "paris": t.astimezone(suivi05c.PARIS).isoformat(timespec="seconds")}


def versions_bibliotheques():
    """Versions lues dans les métadonnées des paquets installés (sans les importer)."""
    v = {"python": sys.version.split()[0]}
    for paquet, cle in (("torch", "torch"), ("numpy", "numpy"), ("nibabel", "nibabel"), ("scipy", "scipy"),
                        ("monai", "monai"), ("nnunetv2", "nnunetv2"), ("TotalSegmentator", "totalsegmentator")):
        try:
            v[cle] = _version(paquet)
        except PackageNotFoundError:
            v[cle] = "absent"
    return v


def tracabilite(phase, script):
    return {"phase": phase, "script": script, "commit": commit_git(), "session": SESSION, **{f"heure_{k}": x for k, x in maintenant().items()},
            "versions": versions_bibliotheques(), "protocole_sha256": SHA256_PROTOCOLE,
            "seuils_sha256": SHA256_SEUILS, "split_sha256": SHA256_SPLIT}


def demarrer_sortie():
    """Copie de tout l'affichage (journal de 05c) : gardée en mémoire jusqu'à ouvrir_sortie()."""
    journal = suivi05c.Journal()
    sys.stdout, sys.stderr = suivi05c.Copie(sys.stdout, journal), suivi05c.Copie(sys.stderr, journal)
    return journal


def ouvrir_sortie(journal, dossier, script):
    chemin = os.path.join(dossier, f"sortie_{script}_{SESSION}.txt")
    journal.ouvrir(chemin)
    return chemin


def poser_verrou(dossier, nom, ignorer=False):
    """Verrou de 05c/05l : arrêt (code 1) si une autre session l'a rafraîchi il y a moins de 15 min."""
    os.makedirs(dossier, exist_ok=True)
    return l05.Verrou(os.path.join(dossier, nom), ignorer)
