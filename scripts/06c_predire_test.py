"""
Étape 6, phase B — Prédictions sur les 5 scanners de test, contrôle R0, contrôle de plausibilité, manifeste
(protocole figé docs/protocole_etape6.md : § 3.3, § 4, § 7.1, § 9, § 10.2 phase B, P7 ; D42).

Pourquoi trois sous-commandes : nnU-Net 2.8.1 (section 12 du notebook) et MONAI 1.6.1 (sections 24-25)
s'installent dans deux sessions Colab distinctes (l'installation de nnU-Net peut changer la version de
PyTorch). Chaque sous-commande refait TOUTES les vérifications au démarrage dont elle a besoin, écrit sa
propre trace, et reprend proprement après une coupure. Le manifeste n'est écrit qu'à la fin, par la 3e.

  nnunet     (GPU L4, session « nnU-Net ») : prédictions A et N0.
     A  = ensemble des 5 plis nnU-Net (checkpoint_final), nnUNetv2_predict -f 0 1 2 3 4, réglages par défaut,
          puis post-traitement D33 avec le postprocessing.pkl de 05i (nnUNetv2_apply_postprocessing) ;
          la sortie BRUTE de l'ensemble est gardée à part (phaseB/nnunet/A_brut, listée en « extra ») ;
     N0 = pli 0 seul (-f 0), sans post-traitement.
  segresnet  (GPU L4, session « MONAI 1.6.1 ») : prédiction S0 = SegResNet pli 0, model_final.pt
          (auto3dseg/entrainement_segresnet_1250/pli_0), lancé EXACTEMENT comme l'évaluation finale de 05m
          (mode « validate » du modèle, même point de contrôle, save_mask, invert : géométrie de l'image
          d'origine), puis contrôle et réécriture avec les fonctions de 05m (controler_prediction,
          ecrire_prediction). Seule différence : la liste des cas (les 5 scanners de test, SANS masque).
          Aucun post-traitement.
  controle   (sans GPU ; n'importe quelle session, une session « processeur » suffit) : après les deux autres.
          R0 (forme, affine à 0,01 mm, étiquettes 0 à 7) sur les 20 prédictions (A, N0, S0 et T, celle de
          l'étape 4 refaite selon D44 par 04c, jamais par ce script) ; un défaut = ARRÊT (code 20), rien n'est écrit. Puis contrôle de
          plausibilité (module figé controle_plausibilite : mesurer, appliquer, resume ; seuils figés),
          règles R0 à R8 sur les 4 modèles (pour T, la restriction de Q7 aux structures 1 à 5 se fait À LA
          LECTURE, en phase C, § 4 (c)). Enfin le MANIFESTE (ecrire_manifeste), sa SHA-256 et le bloc à
          recopier dans config/empreinte_manifeste_etape6.json.

Même commit pour les trois sous-commandes (décision de l'utilisateur, phase B) : chaque sous-commande lit
  d'abord l'état du dépôt (git) : commit et état « propre / modifié ». « Modifié » = un fichier suivi par git a
  changé, ou un fichier non suivi existe dans scripts/ ou config/ (les fichiers ignorés par .gitignore ne
  comptent pas). Dépôt modifié = ARRÊT 24 dans les trois sous-commandes. Le commit est écrit dans la trace
  (rubrique « depot », chaque étape, chaque session) ; si trace_nnunet.json ou trace_segresnet.json existe
  déjà avec un AUTRE commit (ou sans commit noté), ARRÊT 24, avant toute lecture d'image et toute écriture :
  ainsi « segresnet » refuse un commit différent de celui de « nnunet » (et inversement), une reprise refuse un
  commit différent de celui des étapes déjà faites, et « controle » exige que les deux traces aient le commit
  du dépôt courant. Exception : une fois le manifeste écrit, « controle » ne fait que le réafficher (il porte
  son propre commit, affiché à côté du commit courant) ; seule la propreté du dépôt est alors exigée.

Vérifications au démarrage (§ 9), avant toute lecture d'image, dans chaque sous-commande :
  1 à 4 (verifier_demarrage("B")) et 7 (empreintes de TotalSegmentator, config/empreintes_totalseg_etape4.json) ;
  5 : nnunet → nnU-Net 2.8.1 ; segresnet → MONAI 1.6.1 et archive 21ed8e5 (SHA-256 de D37) ; controle → les
      versions relevées dans les deux traces ;
  6 : modèles présents, empreintes relevées (nnunet : 5 checkpoint_final.pth et postprocessing.pkl ;
      segresnet : model_final.pt, bilan_05m.json conforme pour ce fichier, segmenter.py corrigé de 05m ;
      controle : tous, identiques à ceux relevés au moment de la prédiction).
  Toute ouverture de fichier passe par la garde verifier_chemin(..., "B") : imagesTs autorisé, labelsTs
  REFUSÉ (le script ne construit jamais ce chemin).

Réglages d'inférence (P7) relevés et écrits dans les traces puis dans le manifeste (reglages_inference),
sans rien aligner : nnU-Net → options réelles de nnUNetv2_predict (lues dans son analyseur d'arguments pour
NOTRE ligne de commande), axes de miroir du point de contrôle, fenêtre (patch) et espacement de plans.json,
précision (autocast) ; SegResNet → hyper_parameters.yaml, fenêtre « ROI size » lue dans le journal, fenêtre
glissante et absence de miroir lues dans segmenter.py. Les différences N0 / S0 sont affichées comme LIMITE
de la comparaison (b) et écrites dans reglages_inference["differences_N0_S0"].

Entrées (paramètres ; défauts = Drive /content/drive/MyDrive/poc_ct/...) :
  sous-commande   nnunet | segresnet | controle (obligatoire)
  --raw           nnunet_raw : SEUL <raw>/Dataset101_MMWHS/imagesTs est lu (jamais labelsTs)
  --preprocessed  nnunet_preprocessed (variable d'environnement de nnU-Net, rien n'y est lu)
  --results       nnunet_results (modèles nnU-Net, lus)
  --travail       auto3dseg (SegResNet pli 0, archive des modèles 21ed8e5, lus)
  --predictions-ts prédictions de TotalSegmentator refaites selon D44 (lues ; défaut .../predictions/totalsegmentator_d44)
  --sortie        dossier de la phase B (défaut .../etape6/phaseB)
  --local         copie locale des 5 scanners de test pour nnU-Net et SegResNet (défaut /content/etape6_images_test)
  --processus     processus du post-traitement de nnU-Net (-np, défaut 2)
  --ignorer-verrou  comme 05c (seulement si tu es SÛR qu'aucune autre session ne tourne)
Sorties (dans --sortie, Drive) :
  nnunet/A_brut/mmwhs_XXXX.nii.gz (+ plans.json, dataset.json, predict_from_raw_data_args.json de nnU-Net) ;
  predictions/A, predictions/N0, predictions/S0 : mmwhs_XXXX.nii.gz (contrat avec la phase C) ;
  segresnet/ : liste_test.json, sortie brute du modèle (S0_brut/), journal/training.log ;
  trace_nnunet.json, trace_segresnet.json (versions, GPU, commandes, durées, réglages, empreintes) ;
  controle/controle_<M>.json (M = A, N0, S0, T) ; manifeste_etape6.json (statut « figé ») ;
  sortie_06c_<sous-commande>_<SESSION>.txt (copie de l'affichage, heures UTC et Paris) ; verrou_06c (retiré à la fin).

Format de controle/controle_<M>.json (relu par 06d ; JSON UTF-8, clés triées, écrit par ecrire_json_atomique) :
  {"modele": M, "seuils_sha256": "<SHA-256 du fichier de seuils figé>", "objet": texte, "statut_alertes": texte
   (§ 7.1 : officielles pour A, informatives pour N0, S0, T), "regles_appliquees": ["R0", ..., "R8"],
   "serialisation": texte, "cas": {"1004": {"mesures": ..., "resultat": ..., "resume": ...}, ...}}
  - "mesures"  = sortie de controle_plausibilite.mesurer(cas, image, masque), convertie par en_json() :
                 dictionnaires à clés entières (étiquettes 1 à 7 : "voxels", "volumes_ml", "morceaux") →
                 clés chaînes "1" à "7" ; scalaires numpy → nombres JSON (int, float) ; tuples → listes ;
                 None → null ; un float non fini serait écrit NaN / Infinity (jetons acceptés par json.load
                 de Python). Inverse exact : restaurer_mesures() (clés "1".."7" → entiers 1..7), ou lire_controle() ;
  - "resultat" = sortie de appliquer(mesures, seuils) avec TOUTES les règles R0 à R8 (même conversion ; aucun
                 dictionnaire à clés entières) : {"statut": "OK"|"ALERTE", "regles_declenchees": [{"regle",
                 "categorie", "libelle", "objet", "mesure", "seuil"}], "regles_verifiees", "non_evaluees"} ;
  - "resume"   = texte de resume(resultat).
  Le fichier ne contient ni date ni session : relancer donne exactement les mêmes octets (reprise contrôlée).

Matériel, durée, coût (L4, 1,94 unité/h, D25) — estimations, le script affiche les durées réelles :
  nnunet    : GPU L4. nnU-Net prédit chaque scanner par fenêtres de 50 % de recouvrement, chaque fenêtre 8 fois
              (image + 7 symétries, « miroir ») ; d'après une époque d'entraînement (114,5 s pour 250 pas de 2
              morceaux, D25), environ 1 à 2 min par pli et par scanner : A (5 plis) ≈ 30 à 60 min, N0 ≈ 6 à 12 min,
              post-traitement quelques minutes ; avec l'installation, prévoir une session de 1 h à 1 h 30,
              soit ≈ 2 à 3 unités.
  segresnet : GPU L4. Validation de 3 cas = 164 s (D38) → 5 cas ≈ 5 min, plus le chargement : ≈ 10-15 min,
              ≈ 30 min de session avec l'installation, ≈ 1 unité.
  controle  : pas de GPU. Mesures du contrôle (densités, morceaux) sur 20 masques et empreintes des modèles
              (≈ 1,7 Go lus dans Drive) : ≈ 20 à 45 min. En session processeur : pas d'unité GPU.
Reprise après coupure : relancer la MÊME cellule. Chaque étape terminée est notée dans la trace avec les
  empreintes de ses fichiers (vérifiées à la relance, puis sautée) ; une étape interrompue est refaite
  entièrement, après avoir RENOMMÉ (jamais effacé) son dossier incomplet en <dossier>_interrompu_<SESSION>.
  Une fois le manifeste écrit, « controle » ne recalcule rien : il réaffiche l'empreinte du manifeste et le
  commit inscrit dedans (depot_phase_B) à côté du commit courant. Si config/empreinte_manifeste_etape6.json
  existe dans le dépôt, son empreinte est comparée à celle du manifeste (écart = ARRÊT 17, comme
  verifier_manifeste en phase C) ; sinon, le bloc à recopier est réaffiché.
Codes de sortie :
  0 = sous-commande terminée ; 1 = garde-fou (dossier, fichiers en trop ou manquants, verrou, --sortie dans le
  dépôt) ; 3 = nnU-Net ou SegResNet s'est arrêté en erreur ou a été interrompu : RELANCER LA MÊME CELLULE ;
  5 = pas de GPU ; 10 à 18 = vérifications du § 9 ou garde (etape6_commun : 10 protocole, 11 seuils, 12 quatre
  valeurs, 13 découpage, 14 versions, 15 modèles, 16 TotalSegmentator, 17 manifeste (ou, à la relance de
  « controle », empreinte du manifeste ≠ référence du dépôt : ne rien effacer, en parler), 18 lecture refusée) ;
  19 = un fichier déjà produit a changé, ou le manifeste existe déjà dans le dépôt : NE RIEN EFFACER, en parler ;
  20 = R0 en échec (défaut technique d'une prédiction) : rien n'est écrit, ne rien corriger soi-même, en parler
  (A, N0, S0 : correction AVANT l'ouverture, consignée ; T : la correction de T relève d'une déviation,
  § 10.2 B.3, D43 : elle n'est PAS implémentée ici, un échec R0 sur T arrête la phase B) ;
  21 = incohérence d'une prédiction (post-traitement autre que D33, cas manquants, modèle non confirmé) :
  ne pas utiliser ; 22 = « controle » lancé avant la fin de « nnunet » ou « segresnet » ;
  24 = dépôt « modifié », ou commit différent de celui noté dans une trace de la phase B : rien n'est lu ni
  écrit. Que faire : remettre le dépôt au commit noté (après la section 2 : git checkout <commit>), ou bien
  refaire les sous-commandes déjà faites avec le nouveau commit ; le script n'écrase jamais leurs sorties :
  en parler d'abord (dossier phaseB à mettre de côté, consigné au journal).

Vocabulaire :
  - ensemble : moyenne des sorties des 5 modèles de la validation croisée (nnU-Net moyenne les logits,
    c'est-à-dire les scores avant la conversion en probabilités, c'est son réglage par défaut) ;
  - TTA / miroir : augmentation au moment de la prédiction ; on prédit aussi les images retournées et on moyenne ;
  - fenêtre glissante : le réseau ne voit qu'un bloc (« fenêtre ») à la fois ; on le fait glisser sur tout le
    scanner avec un recouvrement, et on combine les blocs avec un poids « gaussien » (plus fort au centre) ;
  - autocast : calcul en demi-précision (16 bits) sur le GPU, plus rapide, quasi sans effet sur le résultat ;
  - commit : identifiant (40 caractères) d'une version enregistrée du code dans git ; même commit = même code ;
  - R0 : cohérence technique du masque (même grille de voxels que le scanner, étiquettes 0 à 7) ;
  - manifeste : liste figée de tous les fichiers (prédictions, contrôles, modèles) avec leur empreinte.
"""
import argparse, glob, json, os, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import etape6_commun as e6                        # n'exécute rien à l'import

PHASE, SCRIPT = "B", "06c"
NOM_DS = "Dataset101_MMWHS"
pp = e6._charger("postraitement05i_06c", "05i_postraitement_nnunet.py")     # TRAINER, PLIS, NOM_CV, lire_postraitement
ctl = e6._charger("controle_plausibilite_06c", "controle_plausibilite.py")  # module figé du contrôle
DOSSIER_TRAINERS = os.path.join(e6.ICI, "trainers_nnunet")                    # comme 05c (nnUNet_extTrainer)
ETIQUETTES_D33 = [2, 4, 5, 6]                                                 # plus grand morceau : OG, OD, VD, aorte
DOSSIER_SEGRESNET = "entrainement_segresnet_1250"                             # D40, D41
CODE_COMMIT = 24                                                              # dépôt modifié ou commit différent
SOUS_COMMANDES_GPU = ("nnunet", "segresnet")                                  # celles qui écrivent une trace
PRECISION_AUTOCAST = "demi-précision automatique (torch.autocast, float16) sur GPU"
STATUT_ALERTES = {"A": "officielles (§ 7.1, comparaison (a))", "N0": "informatives (§ 7.1, P4)",
                  "S0": "informatives (§ 7.1, P4)",
                  "T": "à part, informatives (§ 7.1, Q7) : TOUTES les règles sont appliquées ici ; la phase C ne "
                       "retient que R0, R7 et R1, R2, R8 sur les structures 1 à 5 (§ 4 (c))"}
SERIALISATION = ("mesures et resultat convertis par en_json() de 06c : clés entières → chaînes ('1' à '7'), scalaires "
                 "numpy → nombres JSON, tuples → listes ; inverse : restaurer_mesures() / lire_controle() de 06c")


def nom_pred(cas):
    return e6.NOM_PREDICTION.format(cas=cas)


def nom_image(cas):
    return f"mmwhs_{cas}_0000.nii.gz"


# =================================================================================================
# Fonctions communes aux sous-commandes
# =================================================================================================
def lire_arguments(argv=None):
    p = argparse.ArgumentParser(description="Phase B de l'étape 6 : prédictions, contrôle, manifeste.")
    p.add_argument("sous_commande", choices=("nnunet", "segresnet", "controle"))
    p.add_argument("--raw", default=f"{e6.DRIVE}/nnunet_raw")
    p.add_argument("--preprocessed", default=f"{e6.DRIVE}/nnunet_preprocessed")
    p.add_argument("--results", default=f"{e6.DRIVE}/nnunet_results")
    p.add_argument("--travail", default=f"{e6.DRIVE}/auto3dseg")
    p.add_argument("--predictions-ts", default=e6.PREDICTIONS_TOTALSEG)
    p.add_argument("--sortie", default=e6.DOSSIERS_PHASES[PHASE])
    p.add_argument("--local", default="/content/etape6_images_test")
    p.add_argument("--processus", type=int, default=2)
    p.add_argument("--ignorer-verrou", action="store_true")
    a = p.parse_args(argv)
    for nom in ("raw", "preprocessed", "results", "travail", "predictions_ts", "sortie", "local"):
        setattr(a, nom, os.path.abspath(getattr(a, nom)))
    e6.verifier_hors_depot(a.sortie, "--sortie")
    e6.verifier_hors_depot(a.local, "--local")
    return a


def lister_images(raw):
    """Noms de imagesTs (aucun fichier ouvert) : exactement les 5 scanners de test. Renvoie {cas: chemin}."""
    dossier = os.path.join(raw, NOM_DS, "imagesTs")
    e6.verifier_chemin(dossier, PHASE)
    if not os.path.isdir(dossier):
        e6.arret(1, f"dossier des scanners de test introuvable : {dossier} (--raw).")
    attendus = {nom_image(c): c for c in e6.CAS_TEST_ATTENDUS}
    presents = set(os.listdir(dossier))
    manquants, en_trop = sorted(set(attendus) - presents), sorted(presents - set(attendus))
    if manquants or en_trop:
        e6.arret(1, f"contenu inattendu de {dossier} : manquants {manquants or 'aucun'}, en trop {en_trop or 'aucun'}.")
    return {c: e6.verifier_chemin(os.path.join(dossier, n), PHASE) for n, c in sorted(attendus.items(), key=lambda x: x[1])}


def releve(chemins):
    """{cas: {chemin, sha256}} (lecture binaire via la garde)."""
    return {c: {"chemin": os.path.abspath(p), "sha256": e6.sha256(p, PHASE)} for c, p in chemins.items()}


def copier_local(images, local):
    """Copie les 5 scanners dans un dossier local qui ne contient rien d'autre ; empreintes comparées.
    Renvoie {cas: chemin local}."""
    os.makedirs(local, exist_ok=True)
    autres = sorted(set(os.listdir(local)) - {os.path.basename(p) for p in images.values()})
    if autres:
        e6.arret(1, f"{local} contient d'autres fichiers ({autres[:5]}) : nnU-Net et SegResNet y prédiraient tout.")
    copies = {}
    for c, src in images.items():
        cible = e6.verifier_chemin(os.path.join(local, os.path.basename(src)), PHASE)
        if not os.path.isfile(cible) or e6.sha256(cible, PHASE) != e6.sha256(src, PHASE):
            shutil.copyfile(e6.verifier_chemin(src, PHASE), cible)
            if e6.sha256(cible, PHASE) != e6.sha256(src, PHASE):
                e6.arret(1, f"copie locale de {src} différente de l'original.")
        copies[c] = cible
    print(f"Copie locale des 5 scanners de test : {local} (empreintes identiques aux originaux)")
    return copies


def fichiers_predits(dossier, nom=nom_pred):
    """{cas: chemin} : exactement les 5 prédictions attendues dans le dossier (autres NIfTI = arrêt 21)."""
    e6.verifier_chemin(dossier, PHASE)
    nifti = {n for n in (os.listdir(dossier) if os.path.isdir(dossier) else []) if n.endswith((".nii", ".nii.gz"))}
    attendus = {nom(c) for c in e6.CAS_TEST_ATTENDUS}
    if nifti != attendus:
        e6.arret(21, f"{dossier} : fichiers NIfTI {sorted(nifti)}, attendu exactement {sorted(attendus)}.")
    return {c: e6.verifier_chemin(os.path.join(dossier, nom(c)), PHASE) for c in e6.CAS_TEST_ATTENDUS}


def r0_rapide(image, masque):
    """Contrôle immédiat après la prédiction (mêmes constantes que R0 du module figé : TOLERANCE_R0_MM,
    ETIQUETTES_AUTORISEES). Le R0 qui FAIT FOI est celui de mesurer(), refait par « controle »."""
    import numpy as np
    img, lab = e6.ouvrir_nifti(image, PHASE), e6.ouvrir_nifti(masque, PHASE)
    if img.shape[:3] != lab.shape[:3]:
        return [f"forme {lab.shape[:3]} ≠ image {img.shape[:3]}"]
    pb = []
    ecart = float(np.abs(img.affine - lab.affine).max())
    if ecart > ctl.TOLERANCE_R0_MM:
        pb.append(f"affine : écart max {ecart:.4f} mm > {ctl.TOLERANCE_R0_MM}")
    valeurs = np.unique(np.asanyarray(lab.dataobj))
    hors = [float(v) for v in valeurs if float(v) % 1 or int(v) not in ctl.ETIQUETTES_AUTORISEES]
    if hors:
        pb.append(f"valeurs hors des étiquettes 0 à 7 : {hors[:10]}")
    return pb


def r0_ou_arret(nom, predits, images):
    pb = [f"{nom} {c} : {x}" for c in e6.CAS_TEST_ATTENDUS for x in r0_rapide(images[c], predits[c])]
    if pb:
        e6.arret(20, "R0 en ÉCHEC (défaut technique) :\n  - " + "\n  - ".join(pb) + "\nRien n'est validé : ne rien "
                 "corriger soi-même, en parler (correction avant l'ouverture, consignée).")
    print(f"  R0 immédiat {nom} : 5 prédictions conformes (forme, affine à {ctl.TOLERANCE_R0_MM} mm, étiquettes 0 à 7)")


class Trace:
    """Trace d'une sous-commande (trace_<nom>.json) : étapes terminées et empreintes de leurs fichiers.
    Le commit du dépôt y est écrit dès l'ouverture (rubrique « depot »), avant toute étape."""
    def __init__(self, sortie, nom, depot):
        self.chemin = os.path.join(sortie, f"trace_{nom}.json")
        self.d = e6.lire_json(self.chemin, PHASE) if os.path.isfile(self.chemin) else {"etapes": {}, "sessions": []}
        self.d.setdefault("depot", depot)          # déjà présent = même commit (vérifié par verifier_commit)
        self.ecrire()

    def ecrire(self):
        e6.ecrire_json_atomique(self.chemin, self.d)


def etape(trace, nom, dossier, faire, refaire=False):
    """Étape avec reprise. Terminée (dans la trace) et fichiers inchangés → sautée ; fichier changé → arrêt 19 ;
    sinon le dossier incomplet est RENOMMÉ (jamais effacé) et l'étape refaite. faire() renvoie
    {"fichiers": {cas: chemin}, ...} ; les empreintes sont relevées ici. Renvoie True si l'étape a été faite."""
    fait = trace.d["etapes"].get(nom)
    if fait and fait.get("statut") == "terminé" and not refaire:
        changes = [c for c, f in fait["fichiers"].items()
                   if not os.path.isfile(f["chemin"]) or e6.sha256(f["chemin"], PHASE) != f["sha256"]]
        if changes:
            e6.arret(19, f"étape {nom} : fichiers modifiés ou absents depuis leur production ({changes}). Ne rien effacer, en parler.")
        print(f"Étape {nom} : déjà terminée ({fait['fin_utc']}), 5 empreintes inchangées : non refaite.")
        return False
    if os.path.exists(dossier):
        archive = f"{dossier}_interrompu_{e6.SESSION}"
        os.rename(dossier, archive)
        print(f"Étape {nom} : dossier incomplet d'une session précédente renommé en {archive} (rien n'est effacé).")
    trace.d["etapes"].pop(nom, None)
    t0 = time.time()
    print(f"\n=== Étape {nom} : début {e6.heure()} ===", flush=True)
    resultat = faire()
    resultat["fichiers"] = releve(resultat["fichiers"])
    resultat.update({"statut": "terminé", "duree_s": round(time.time() - t0, 1), "fin_utc": e6.maintenant()["utc"],
                     "session": e6.SESSION, "commit": trace.d["depot"]["commit"]})
    trace.d["etapes"][nom] = resultat
    trace.ecrire()
    print(f"=== Étape {nom} : terminée en {resultat['duree_s'] / 60:.1f} min ===", flush=True)
    return True


def etat_depot():
    """Commit du dépôt et modifications qui comptent : fichiers suivis modifiés (partout), fichiers non suivis
    dans scripts/ ou config/ (.gitignore respecté). Renvoie {"commit", "etat": "propre"|"modifié", "modifications"}.
    git illisible (pas un dépôt, git absent) = arrêt 24 : on ne peut pas garantir le même code."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=e6.DEPOT, capture_output=True, timeout=60)
    try:
        r, s = git("rev-parse", "HEAD"), git("status", "--porcelain", "-z", "--untracked-files=all")
    except (OSError, subprocess.SubprocessError) as err:
        e6.arret(CODE_COMMIT, f"impossible de lire l'état du dépôt avec git ({err}).")
    commit = r.stdout.decode("utf-8", "replace").strip()
    if r.returncode or s.returncode or len(commit) not in (40, 64):
        erreur = (r.stderr + s.stderr).decode("utf-8", "replace").strip()[:300]
        e6.arret(CODE_COMMIT, f"état du dépôt illisible ({erreur or commit!r}) : lancer le script depuis le dépôt "
                              "cloné à la section 2.")
    champs, modifs, i = s.stdout.decode("utf-8", "replace").split("\0"), [], 0
    while i < len(champs):                 # format -z : « XY chemin », séparés par un octet nul
        x, i = champs[i], i + 1
        if len(x) < 4:
            continue
        code, chemin = x[:2], x[3:]
        if code[0] in "RC":                # renommage ou copie : l'ancien nom suit dans le champ suivant
            i += 1
        if code == "??":
            if chemin.startswith(("scripts/", "config/")):
                modifs.append(f"fichier non suivi : {chemin}")
        elif code != "!!":
            modifs.append(f"fichier suivi modifié ({code.strip()}) : {chemin}")
    return {"commit": commit, "etat": "modifié" if modifs else "propre", "modifications": modifs}


def commits_notes(t):
    """Tous les commits écrits dans une trace (rubrique depot, champ commit, étapes, sessions)."""
    c = {(t.get("depot") or {}).get("commit"), t.get("commit")}
    c |= {(x or {}).get("commit") for x in t.get("sessions", [])}
    c |= {(x or {}).get("commit") for x in (t.get("etapes") or {}).values()}
    return c - {None}


def verifier_commit(a, depot):
    """Arrêt 24 si le dépôt est modifié, ou si une trace de la phase B existe avec un autre commit (ou sans commit).
    Appelée avant toute lecture d'image et toute écriture."""
    if depot["etat"] != "propre":
        e6.arret(CODE_COMMIT, f"dépôt « modifié » (commit {depot['commit']}) :\n  - " + "\n  - ".join(depot["modifications"])
                 + "\nLes cellules 28 a, b et c doivent tourner sur un commit exact, sans modification. Ne rien modifier "
                 "dans le dépôt depuis Colab : refaire la section 2 (nouvelle copie du dépôt), puis, si une trace de la "
                 "phase B existe déjà, git checkout <commit de la trace>. Rien n'a été lu ni écrit.")
    if a.sous_commande == "controle" and os.path.isfile(os.path.join(a.sortie, e6.NOM_MANIFESTE)):
        print("Commit : manifeste déjà écrit (il porte son propre commit) ; seule la propreté du dépôt est vérifiée.")
        return
    ecarts = []
    for nom in SOUS_COMMANDES_GPU:
        chemin = os.path.join(a.sortie, f"trace_{nom}.json")
        if os.path.isfile(chemin):
            notes = commits_notes(e6.lire_json(chemin, PHASE))
            if notes != {depot["commit"]}:
                ecarts.append(f"{chemin} : commit {', '.join(sorted(notes)) or 'non noté'}")
    if ecarts:
        e6.arret(CODE_COMMIT, f"commit différent entre les cellules de la phase B. Dépôt actuel : {depot['commit']} ;\n  - "
                 + "\n  - ".join(ecarts) + "\nQue faire : soit remettre le dépôt au commit noté (après la section 2 : "
                 "!git checkout <commit noté>) et relancer cette cellule ; soit refaire les sous-commandes déjà faites "
                 "avec le commit actuel : le script n'écrase jamais leurs sorties, il faut d'abord en parler (dossier "
                 "phaseB mis de côté, consigné au journal). Rien n'a été lu ni écrit.")
    deja = [n for n in SOUS_COMMANDES_GPU if os.path.isfile(os.path.join(a.sortie, f"trace_{n}.json"))]
    print(f"Commit : dépôt propre ; traces existantes au même commit : {', '.join(deja) or 'aucune'}")


def demarrer(a):
    """Affichage recopié, état du dépôt (commit), vérifications 1 à 4 et 7 (dépôt, puis 5 petits fichiers de
    TotalSegmentator)."""
    journal = e6.demarrer_sortie()
    print(f"Étape 6, phase B (06c {a.sous_commande}) — session {e6.SESSION} ; début : {e6.heure()}")
    a.depot = etat_depot()
    print(f"Commit du dépôt : {a.depot['commit']} ({a.depot['etat']})")
    verifier_commit(a, a.depot)
    verifs = e6.verifier_demarrage(PHASE)
    verifs["totalsegmentator"] = e6.verifier_empreintes_totalseg(a.predictions_ts, PHASE)
    return journal, verifs


def ouvrir(a, journal):
    """Après les vérifications seulement : dossier de sortie, copie de l'affichage, verrou."""
    os.makedirs(a.sortie, exist_ok=True)
    print(f"Copie de l'affichage : {e6.ouvrir_sortie(journal, a.sortie, f'{SCRIPT}_{a.sous_commande}')}")
    e6.poser_verrou(a.sortie, "verrou_06c", a.ignorer_verrou)


def gpu():
    import torch
    if not torch.cuda.is_available():
        e6.arret(5, "pas de GPU visible (Exécution → Modifier le type d'exécution → GPU L4).")
    nom = torch.cuda.get_device_name(0)
    if "L4" not in nom:
        print(f"ATTENTION : GPU {nom}, pas un L4 : les durées estimées ne valent que pour le L4 (résultat inchangé).")
    print(f"GPU : {nom} ; torch {torch.__version__}")
    return nom


def lire_options_cli(point_entree, argv):
    """Options RÉELLES qu'un outil en ligne de commande calcule pour notre ligne : on appelle son point
    d'entrée, on intercepte l'analyse des arguments (argparse) et on s'arrête AVANT tout calcul."""
    class _Capture(Exception):
        pass
    origine = argparse.ArgumentParser.parse_args

    def capturer(self, args=None, namespace=None):
        raise _Capture(origine(self, list(argv), namespace))
    argparse.ArgumentParser.parse_args = capturer
    try:
        point_entree()
    except SystemExit as err:                      # argparse refuse notre ligne : rien n'a été calculé
        e6.arret(1, f"options refusées par l'outil (code {err.code}) : {' '.join(argv)}")
    except _Capture as c:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(c.args[0]).items()}
    finally:
        argparse.ArgumentParser.parse_args = origine
    e6.arret(1, "options de la ligne de commande non relevées (point d'entrée inattendu).")


# =================================================================================================
# Sous-commande nnunet : A (ensemble 5 plis + post-traitement D33) et N0 (pli 0)
# =================================================================================================
def verifier_postraitement_d33(cv):
    """postprocessing.json de 05i = D33 exactement (plus grand morceau de 2, 4, 5, 6 ; rien d'autre)."""
    _, etapes = pp.lire_postraitement(cv)        # lecture du .json ; le .pkl n'est ouvert que par nnU-Net
    cibles = [x["etiquettes"] for x in etapes]
    if any(x["etape"] != "b" for x in etapes) or sorted(c[0] for c in cibles) != ETIQUETTES_D33:
        e6.arret(21, f"post-traitement de {cv} : {[x['cible'] for x in etapes]}, attendu D33 (plus grand morceau de "
                     f"{ETIQUETTES_D33} seulement).")
    print(f"Post-traitement de 05i conforme à D33 : plus grand morceau des étiquettes {ETIQUETTES_D33}")


def comparer_postraitement(brut, post):
    """Le post-traitement n'a fait QUE remettre à 0 des voxels des étiquettes D33. Renvoie (problèmes, retraits)."""
    import numpy as np
    b, p = e6.ouvrir_nifti(brut, PHASE), e6.ouvrir_nifti(post, PHASE)
    if b.shape != p.shape or float(np.abs(b.affine - p.affine).max()) > ctl.TOLERANCE_R0_MM:
        return ["forme ou affine différente de la sortie brute"], {}
    db, dp = np.asanyarray(b.dataobj), np.asanyarray(p.dataobj)
    change = db != dp
    pb = []
    if np.any(dp[change] != 0):
        pb.append("des voxels ont changé d'étiquette sans devenir du fond")
    retirees = sorted(int(x) for x in np.unique(db[change]))
    if not set(retirees) <= set(ETIQUETTES_D33):
        pb.append(f"voxels retirés pour les étiquettes {retirees}, attendu au plus {ETIQUETTES_D33}")
    return pb, {int(e): int(np.sum(db[change] == e)) for e in retirees}


def reglages_nnunet(dossier_modele, argv, plis, points):
    """P7 : options réelles de nnUNetv2_predict pour notre ligne, et ce que contient le modèle."""
    import inspect
    from nnunetv2.inference import predict_from_raw_data as prd
    options = lire_options_cli(prd.predict_entry_point, argv)
    conf = e6.lire_json(os.path.join(dossier_modele, "plans.json"))["configurations"]["3d_fullres"]
    axes = points[f"fold_{plis[0]}"]["inference_allowed_mirroring_axes"]   # nnU-Net lit ceux du 1er pli
    miroir = not options["disable_tta"] and axes is not None
    source = inspect.getsource(prd.nnUNetPredictor)
    return {"outil": f"nnUNetv2_predict (nnunetv2 {e6.versions_bibliotheques()['nnunetv2']})", "plis": plis,
            "options_ligne_de_commande": options, "point_de_controle": options["chk"],
            "miroir_tta": miroir, "axes_miroir": axes, "passages_par_fenetre": 2 ** len(axes) if miroir else 1,
            "fenetre_voxels": conf.get("patch_size"), "espacement_reseau_mm": conf.get("spacing"),
            "recouvrement_fenetres": 1 - float(options["step_size"]),
            "ponderation_fenetres": "gaussienne", "ponderation_source": "use_gaussian=True, fixé par nnUNetv2_predict",
            "precision": PRECISION_AUTOCAST if "torch.autocast" in source else "non relevée",
            "precision_source": "torch.autocast dans nnUNetPredictor (actif seulement sur GPU)",
            "calcul_sur_gpu": not options["not_on_device"],
            "ensemble": ("moyenne des logits des plis (sorties avant softmax), réglage par défaut de nnU-Net"
                         if len(plis) > 1 else "un seul modèle"),
            "torch_compile": os.environ.get("nnUNet_compile", "non défini (pas de compilation)"),
            "reechantillonnage": "vers l'espacement du réseau, puis retour à la géométrie de l'image (nnU-Net)",
            "points_de_controle": {k: v for k, v in points.items() if k in [f"fold_{p}" for p in plis]}}


def lire_points_nnunet(dossier_modele):
    """Contenu non-poids des 5 checkpoint_final.pth (nom du trainer, configuration, époque, axes de miroir)."""
    import torch
    r = {}
    for k in pp.PLIS:
        ck = torch.load(os.path.join(dossier_modele, f"fold_{k}", "checkpoint_final.pth"), map_location="cpu", weights_only=False)
        axes = ck.get("inference_allowed_mirroring_axes")
        r[f"fold_{k}"] = {"trainer_name": ck.get("trainer_name"), "configuration": (ck.get("init_args") or {}).get("configuration"),
                          "fold": (ck.get("init_args") or {}).get("fold"), "current_epoch": ck.get("current_epoch"),
                          "inference_allowed_mirroring_axes": None if axes is None else [int(x) for x in axes]}
        del ck
    pb = [f"{k} : {v}" for k, v in r.items() if v["trainer_name"] != pp.TRAINER or v["configuration"] != pp.CONFIGURATION
          or str(v["fold"]) != k.split("_")[1]]
    if pb:
        e6.arret(15, "points de contrôle nnU-Net inattendus :\n  - " + "\n  - ".join(pb))
    return r


def lancer_nnunet(commande, env, journal):
    t0 = time.time()
    code, _ = e6.prep.lancer(commande, env, journal)
    if code:
        e6.arret(3, f"{os.path.basename(commande[0])} s'est arrêté en erreur (code {code}) après "
                    f"{(time.time() - t0) / 60:.1f} min : RELANCER LA MÊME CELLULE (l'étape sera refaite).")
    return round(time.time() - t0, 1)


def sous_commande_nnunet(a, journal, verifs):
    versions = e6.verifier_versions(nnunet=True)
    modeles = {k: v for k, v in e6.chemins_modeles(a.results, a.travail).items() if k.startswith("nnunet_")}
    dossier_modele = pp.dossier_modele(a.results)
    cv = os.path.join(dossier_modele, pp.NOM_CV)
    modeles.update({"nnunet_plans": os.path.join(dossier_modele, "plans.json"),
                    "nnunet_dataset_json": os.path.join(dossier_modele, "dataset.json"),
                    "nnunet_trainer_py": os.path.join(DOSSIER_TRAINERS, f"{pp.TRAINER}.py")})
    releves_modeles = e6.verifier_modeles(modeles, PHASE)
    verifier_postraitement_d33(cv)
    images = lister_images(a.raw)
    nom_gpu = gpu()
    env = e6.prep.environnement_nnunet(a.raw, a.preprocessed, a.results, a.processus)
    env["nnUNet_extTrainer"] = DOSSIER_TRAINERS          # nnU-Net retrouve la classe du trainer par son nom
    os.environ.update({k: env[k] for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results", "nnUNet_extTrainer")})
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
    fichier = recursive_find_trainer_class_by_name(pp.TRAINER).__init__.__code__.co_filename
    if os.path.normcase(os.path.dirname(os.path.abspath(fichier))) != os.path.normcase(DOSSIER_TRAINERS):
        e6.arret(1, f"la classe {pp.TRAINER} trouvée ne vient pas de {DOSSIER_TRAINERS}.")
    exe_pred, exe_pp = shutil.which("nnUNetv2_predict"), shutil.which("nnUNetv2_apply_postprocessing")
    if not exe_pred or not exe_pp:
        e6.arret(1, "nnUNetv2_predict ou nnUNetv2_apply_postprocessing introuvable : installer nnU-Net (cellule ci-dessus).")
    ouvrir(a, journal)
    trace = Trace(a.sortie, "nnunet", a.depot)
    images_releve = releve(images)
    local = copier_local(images, a.local)
    points = lire_points_nnunet(dossier_modele)
    journal_nnunet = os.path.join(a.sortie, "journal_nnunet_06c.txt")
    dossiers = {"A_brut": os.path.join(a.sortie, "nnunet", "A_brut"),
                "A": os.path.join(a.sortie, "predictions", "A"), "N0": os.path.join(a.sortie, "predictions", "N0")}
    for d in dossiers.values():
        e6.verifier_chemin(d, PHASE)
        if " " in d:
            e6.arret(1, f"{d} contient un espace.")

    def args_predict(sortie, plis):          # réglages PAR DÉFAUT : seules les options obligatoires sont données
        return ["-i", a.local, "-o", sortie, "-d", str(pp.ID_DATASET), "-c", pp.CONFIGURATION, "-tr", pp.TRAINER,
                "-p", pp.PLANS, "-f", *map(str, plis)]
    reglages = {"A": reglages_nnunet(dossier_modele, args_predict(dossiers["A_brut"], pp.PLIS), list(pp.PLIS), points),
                "N0": reglages_nnunet(dossier_modele, args_predict(dossiers["N0"], [0]), [0], points)}

    def predire(nom, plis):
        def faire():
            commande = [exe_pred, *args_predict(dossiers[nom], plis)]
            duree = lancer_nnunet(commande, env, journal_nnunet)
            f = fichiers_predits(dossiers[nom])
            r0_ou_arret(nom, f, local)
            return {"fichiers": f, "commande": " ".join(commande), "duree_prediction_s": duree}
        return faire

    a_brut_refait = etape(trace, "A_brut", dossiers["A_brut"], predire("A_brut", pp.PLIS))

    def postraiter():
        commande = [exe_pp, "-i", dossiers["A_brut"], "-o", dossiers["A"], "-pp_pkl_file", modeles["nnunet_postprocessing"],
                    "-np", str(a.processus), "-plans_json", os.path.join(cv, "plans.json"),
                    "-dataset_json", os.path.join(cv, "dataset.json")]
        duree = lancer_nnunet(commande, env, journal_nnunet)
        f, brut = fichiers_predits(dossiers["A"]), fichiers_predits(dossiers["A_brut"])
        retraits, pb = {}, []
        for c in e6.CAS_TEST_ATTENDUS:
            p, retraits[c] = comparer_postraitement(brut[c], f[c])
            pb += [f"{c} : {x}" for x in p]
        if pb:
            e6.arret(21, "post-traitement non conforme à D33 :\n  - " + "\n  - ".join(pb))
        print("  Post-traitement : seuls des voxels des étiquettes D33 ont été remis à 0 ; voxels retirés par cas : "
              + " ; ".join(f"{c} {r or 'aucun'}" for c, r in retraits.items()))
        r0_ou_arret("A", f, local)
        return {"fichiers": f, "commande": " ".join(commande), "duree_postraitement_s": duree, "voxels_retires": retraits}
    etape(trace, "A", dossiers["A"], postraiter, refaire=a_brut_refait)
    etape(trace, "N0", dossiers["N0"], predire("N0", [0]))

    trace.d["sessions"].append({"session": e6.SESSION, "gpu": nom_gpu, "commit": a.depot["commit"], **e6.maintenant()})
    trace.d.update({"statut": "terminé", "versions": versions, "modeles": releves_modeles, "images_test": images_releve,
                    "reglages_inference": reglages, "verifications_demarrage": verifs, "commit": a.depot["commit"]})
    trace.ecrire()
    print(f"\nRéglages relevés (P7) : miroir {reglages['A']['miroir_tta']} (axes {reglages['A']['axes_miroir']}), fenêtre "
          f"{reglages['A']['fenetre_voxels']}, recouvrement {reglages['A']['recouvrement_fenetres']:.0%}, "
          f"{reglages['A']['precision']}")
    print(f"Résultat : sous-commande nnunet TERMINÉE (A et N0) ; trace : {trace.chemin} ; fin : {e6.heure()}")


# =================================================================================================
# Sous-commande segresnet : S0 (pli 0, model_final.pt), comme l'évaluation finale de 05m
# =================================================================================================
def reglages_segresnet(hp, segmenter_py, texte_journal, epoque):
    """P7 : ce que segmenter.py fait réellement en mode « validate » (fenêtre lue dans le journal)."""
    import re
    src = open(segmenter_py, encoding="utf-8").read()
    val_epoch = src[src.index("    def val_epoch("):src.index("    def logits2pred(")]
    roi = re.findall(e6.l05.MOTIFS["roi"], texte_journal)
    defaut = hp.get("sliding_inferrer") is None
    return {"outil": "segmenter.py du modèle segresnet (archive 21ed8e5, correctif 05m), mode « validate » comme "
                     "l'évaluation finale de 05m", "plis": [0], "point_de_controle": f"model_final.pt (époque {epoque})",
            "miroir_tta": "flip" in val_epoch, "axes_miroir": None, "passages_par_fenetre": 2 ** 3 if "flip" in val_epoch else 1,
            "fenetre_voxels": [int(x) for x in roi[-1].replace(",", " ").split()] if roi else None,
            "fenetre_voxels_source": "ligne « ROI size » du journal" if roi else "non relevée",
            "recouvrement_fenetres": 0.625 if defaut and "overlap=0.625" in src else "non relevé",
            "ponderation_fenetres": "gaussienne" if defaut and 'mode="gaussian"' in src else "non relevée",
            "fenetres_par_lot": 1 if defaut and "sw_batch_size=1" in src else "non relevé",
            "precision": PRECISION_AUTOCAST if hp.get("amp", True) else "pleine précision (amp: false)",
            "precision_source": "amp de hyper_parameters.yaml (torch.autocast dans val_epoch)",
            "calcul_sur_gpu": True, "ensemble": "un seul modèle",
            "reechantillonnage": {k: hp.get(k) for k in ("resample", "resample_resolution", "orientation_ras",
                                                         "crop_foreground", "normalize_mode", "intensity_bounds")},
            "retour_geometrie_origine": "Invertd (transformations inversées), comme 05m"}


def sous_commande_segresnet(a, journal, verifs):
    m05 = e6._charger("entrainer05m_06c", "05m_entrainer_segresnet.py")       # fonctions de l'évaluation finale
    versions = e6.verifier_versions(monai=True, archive_auto3dseg=os.path.join(a.travail, f"algorithm_templates_{e6.HASH_MODELES_AUTO3DSEG}.tar.gz"))
    dossier_pli = os.path.join(a.travail, DOSSIER_SEGRESNET, "pli_0")
    dossier_algo = os.path.join(dossier_pli, "segresnet_0")
    final = e6.chemins_modeles(a.results, a.travail)["segresnet_pli_0"]
    modeles = {"segresnet_pli_0": final,
               "segresnet_hyper_parameters": os.path.join(dossier_algo, "configs", "hyper_parameters.yaml"),
               "segresnet_segmenter_py": os.path.join(dossier_algo, "scripts", "segmenter.py")}
    releves_modeles = e6.verifier_modeles(modeles, PHASE)
    bilan = e6.lire_json(os.path.join(dossier_pli, m05.NOM_BILAN), PHASE) if os.path.isfile(os.path.join(dossier_pli, m05.NOM_BILAN)) else {}
    if not bilan.get("conforme") or bilan.get("sha256_model_final") != releves_modeles["segresnet_pli_0"]["sha256"]:
        e6.arret(15, f"{m05.NOM_BILAN} absent, non conforme, ou d'un autre model_final.pt : l'entraînement du pli 0 "
                     "(section 25) n'est pas terminé tel que décrit par D41.")
    if releves_modeles["segresnet_segmenter_py"]["sha256"] != m05.SHA256_SEGMENTER_CORRIGE:
        e6.arret(15, "segmenter.py du pli 0 n'est pas celui corrigé par 05m (SHA-256 différente de D39).")
    epoque = bilan["epoque_du_modele"]
    print(f"  6. SegResNet pli 0 terminé : bilan_05m.json conforme, model_final.pt de l'époque {epoque} (Dice hors pli "
          f"{bilan.get('dice_moyen'):.3f}, D41)")
    images = lister_images(a.raw)
    nom_gpu = gpu()
    from monai.apps.auto3dseg.utils import import_bundle_algo_history
    from monai.utils.enums import AlgoKeys
    algo = next((h[AlgoKeys.ALGO] for h in import_bundle_algo_history(dossier_pli, only_trained=False)
                 if h[AlgoKeys.ID] == "segresnet_0"), None)
    if algo is None:
        e6.arret(15, f"algorithme segresnet_0 introuvable dans {dossier_pli}.")
    ouvrir(a, journal)
    trace = Trace(a.sortie, "segresnet", a.depot)
    images_releve = releve(images)
    local = copier_local(images, a.local)
    travail_s0 = os.path.join(a.sortie, "segresnet")
    brut, sortie_s0 = os.path.join(travail_s0, "S0_brut"), os.path.join(a.sortie, "predictions", "S0")
    dossier_journal = os.path.join(travail_s0, "journal")
    for d in (a.local, travail_s0, sortie_s0, final):
        e6.verifier_chemin(d, PHASE)
        if " " in d:
            e6.arret(1, f"{d} contient un espace : MONAI découpe ses commandes aux espaces.")
    hp = e6.l05._lire_yaml(modeles["segresnet_hyper_parameters"])
    info = {}

    def predire():
        for d in (brut, dossier_journal):
            if os.path.exists(d):
                os.rename(d, f"{d}_interrompu_{e6.SESSION}")
            os.makedirs(d)
        liste = os.path.join(travail_s0, "liste_test.json")      # 5 scanners, AUCUN masque (aucune vérité)
        e6.ecrire_json_atomique(liste, {"testing": [{"image": local[c]} for c in e6.CAS_TEST_ATTENDUS]})
        params = {"validate#enabled": True, "validate#ckpt_name": final, "validate#save_mask": True,   # = 05m
                  "validate#invert": True, "validate#output_path": brut,                               # = 05m
                  "validation_key": "testing", "data_list_file_path": liste, "data_file_base_dir": a.local,
                  "log_output_file": os.path.join(dossier_journal, "training.log")}  # journal hors du dossier du pli
        print("Prédiction S0 : SegResNet en mode « validate » avec model_final.pt, comme l'évaluation finale de 05m. "
              "La ligne « Lancement de l'entraînement » ci-dessous lance cette prédiction : AUCUN entraînement.", flush=True)
        code, interrompu, _, suivi_g, duree = e6.l05.lancer_entrainement(algo, params, dossier_journal, travail_s0)
        if code or interrompu:
            e6.arret(3, f"SegResNet {'interrompu' if interrompu else 'en erreur'} (code {code}) : RELANCER LA MÊME CELLULE.")
        texte = open(os.path.join(dossier_journal, "training.log"), encoding="utf-8", errors="replace").read()
        ev = m05.lire_evaluation(texte)
        pb = []
        if ev["charge"] is None or os.path.basename(ev["charge"][0]) != "model_final.pt" or ev["charge"][1] != epoque:
            pb.append(f"le modèle n'a pas confirmé avoir chargé model_final.pt de l'époque {epoque} (lu : {ev['charge']})")
        if ev["nb_fichiers"] != len(e6.CAS_TEST_ATTENDUS):
            pb.append(f"le modèle a lu {ev['nb_fichiers']} scanners au lieu de 5")
        sorties = {e6.l05.prep05k._cas_du_chemin(p): p for p in glob.glob(os.path.join(brut, "**", "*.nii.gz"), recursive=True)}
        if set(sorties) != {f"mmwhs_{c}" for c in e6.CAS_TEST_ATTENDUS}:
            pb.append(f"prédictions brutes {sorted(sorties)} au lieu des 5 cas de test")
        if pb:
            e6.arret(21, "prédiction S0 non confirmée :\n  - " + "\n  - ".join(pb))
        os.makedirs(sortie_s0, exist_ok=True)
        import nibabel as nib
        import numpy as np
        for c in e6.CAS_TEST_ATTENDUS:          # contrôle et réécriture : fonctions de 05m, inchangées
            p, donnees, img = m05.controler_prediction(e6.verifier_chemin(local[c], PHASE),
                                                       e6.verifier_chemin(sorties[f"mmwhs_{c}"], PHASE))
            if donnees is None or p:
                e6.arret(20, f"R0 en ÉCHEC pour S0 {c} : {p}. Rien n'est validé : en parler.")
            cible = e6.verifier_chemin(os.path.join(sortie_s0, nom_pred(c)), PHASE)
            m05.ecrire_prediction(donnees, img, cible)
            relu = nib.load(cible)
            if relu.shape != img.shape[:3] or relu.get_data_dtype() != np.uint8 \
                    or not np.array_equal(np.asanyarray(relu.dataobj), donnees):
                e6.arret(20, f"S0 {c} : fichier réécrit non conforme.")
        f = fichiers_predits(sortie_s0)
        r0_ou_arret("S0", f, local)
        info["texte"] = texte
        return {"fichiers": f, "params_modele": {k: str(v) for k, v in params.items()}, "duree_prediction_s": round(duree, 1),
                "pic_memoire_gpu_mo": suivi_g.pic if suivi_g else None}
    fait = etape(trace, "S0", sortie_s0, predire)
    texte = info.get("texte") or open(os.path.join(dossier_journal, "training.log"), encoding="utf-8", errors="replace").read()
    reglages = {"S0": reglages_segresnet(hp, modeles["segresnet_segmenter_py"], texte, epoque)}
    if fait or trace.d.get("statut") != "terminé":
        trace.d["sessions"].append({"session": e6.SESSION, "gpu": nom_gpu, "commit": a.depot["commit"], **e6.maintenant()})
    trace.d.update({"statut": "terminé", "versions": versions, "modeles": releves_modeles, "images_test": images_releve,
                    "reglages_inference": reglages, "verifications_demarrage": verifs, "commit": a.depot["commit"],
                    "bilan_05m": {k: bilan.get(k) for k in ("epoque_du_modele", "sha256_model_final", "dice_moyen", "conforme")}})
    trace.ecrire()
    r = reglages["S0"]
    print(f"\nRéglages relevés (P7) : miroir {r['miroir_tta']}, fenêtre {r['fenetre_voxels']}, recouvrement "
          f"{r['recouvrement_fenetres']}, {r['precision']}")
    print(f"Résultat : sous-commande segresnet TERMINÉE (S0) ; trace : {trace.chemin} ; fin : {e6.heure()}")


# =================================================================================================
# Sous-commande controle : R0, contrôle de plausibilité, manifeste
# =================================================================================================
def en_json(x, chemin="racine"):
    """Conversion réversible vers des types JSON (voir l'en-tête). Arrêt si un type est inconnu."""
    import numpy as np
    if isinstance(x, dict):
        r = {}
        for k, v in x.items():
            if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, str, np.integer)):
                e6.arret(1, f"sérialisation : clé {k!r} ({type(k).__name__}) inattendue dans {chemin}.")
            cle = str(int(k)) if isinstance(k, (int, np.integer)) else k
            if cle in r:
                e6.arret(1, f"sérialisation : deux clés donnent {cle!r} dans {chemin}.")
            r[cle] = en_json(v, f"{chemin}.{cle}")
        return r
    if isinstance(x, (list, tuple)):
        return [en_json(v, chemin) for v in x]
    if isinstance(x, np.ndarray):
        return en_json(x.tolist(), chemin)
    if isinstance(x, np.generic):
        return x.item()
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    e6.arret(1, f"sérialisation : type {type(x).__name__} inattendu dans {chemin}.")


CLES_ENTIERES = ("voxels", "volumes_ml", "morceaux")      # dictionnaires de mesurer() indexés par étiquette


def restaurer_mesures(m):
    """Inverse de en_json() pour une sortie de mesurer() : clés '1'..'7' → entiers 1..7."""
    return {k: ({int(e): v for e, v in m[k].items()} if k in CLES_ENTIERES and isinstance(m[k], dict) else v)
            for k, v in m.items()}


def lire_controle(chemin, phase="C"):
    """Relit controle_<M>.json (pour 06d) et restaure les clés entières des mesures."""
    d = e6.lire_json(chemin, phase)
    for x in d["cas"].values():
        x["mesures"] = restaurer_mesures(x["mesures"])
    return d


def ecrire_sans_ecraser(chemin, contenu):
    """Écrit un JSON ; s'il existe, il doit être identique octet pour octet (sinon arrêt 19, rien n'est écrit)."""
    if os.path.isfile(chemin):
        if open(chemin, "rb").read() != e6._serialiser(contenu):
            e6.arret(19, f"{chemin} existe déjà avec un autre contenu : ne rien effacer, en parler.")
        print(f"  {os.path.basename(chemin)} existe déjà, identique : gardé tel quel.")
        return
    e6.ecrire_json_atomique(chemin, contenu)


def lire_trace(sortie, nom, etapes):
    chemin = os.path.join(sortie, f"trace_{nom}.json")
    t = e6.lire_json(chemin, PHASE) if os.path.isfile(chemin) else {}
    manque = [x for x in etapes if (t.get("etapes", {}).get(x) or {}).get("statut") != "terminé"]
    if t.get("statut") != "terminé" or manque:
        e6.arret(22, f"sous-commande « {nom} » non terminée ({chemin} ; étapes manquantes {manque or '—'}) : la lancer d'abord.")
    return t, chemin


def differences(r1, r2):
    """Réglages d'inférence qui diffèrent entre N0 et S0 (P7 : limite de (b), rien n'est aligné)."""
    noms = {"miroir_tta": "augmentation au moment de la prédiction (miroir)", "passages_par_fenetre": "passages par fenêtre",
            "fenetre_voxels": "taille de fenêtre (voxels)", "recouvrement_fenetres": "recouvrement des fenêtres",
            "ponderation_fenetres": "pondération des fenêtres", "precision": "précision du calcul"}
    return [{"reglage": n, "N0": r1.get(k), "S0": r2.get(k)} for k, n in noms.items() if r1.get(k) != r2.get(k)]


def sous_commande_controle(a, journal, verifs):
    chemin_manifeste = os.path.join(a.sortie, e6.NOM_MANIFESTE)
    if os.path.isfile(chemin_manifeste):                  # déjà figé : on réaffiche, on ne recalcule rien
        reafficher_manifeste(a, chemin_manifeste)
        return
    tn, chemin_tn = lire_trace(a.sortie, "nnunet", ("A_brut", "A", "N0"))
    ts, chemin_ts = lire_trace(a.sortie, "segresnet", ("S0",))
    pb = []                                               # point 5, relu dans les traces
    if tn["versions"].get("nnunetv2") != e6.VERSION_NNUNET:
        pb.append(f"A et N0 prédits avec nnU-Net {tn['versions'].get('nnunetv2')}")
    if ts["versions"].get("monai") != e6.VERSION_MONAI or ts["versions"].get("sha256_archive_auto3dseg") != e6.SHA256_ARCHIVE_AUTO3DSEG:
        pb.append(f"S0 prédit avec MONAI {ts['versions'].get('monai')} / archive {ts['versions'].get('sha256_archive_auto3dseg')}")
    if pb:
        e6.arret(14, "versions des prédictions non conformes (§ 9, point 5) :\n  - " + "\n  - ".join(pb))
    print(f"  5. versions des prédictions (traces) : nnU-Net {tn['versions']['nnunetv2']} (A, N0), MONAI "
          f"{ts['versions']['monai']} et archive {e6.HASH_MODELES_AUTO3DSEG} (S0)")
    modeles = {**{k: v["chemin"] for k, v in tn["modeles"].items()}, **{k: v["chemin"] for k, v in ts["modeles"].items()}}
    releves_modeles = e6.verifier_modeles(modeles, PHASE)
    for t in (tn, ts):
        changes = [k for k, v in t["modeles"].items() if releves_modeles[k]["sha256"] != v["sha256"]]
        if changes:
            e6.arret(19, f"modèles modifiés depuis la prédiction : {changes}. Ne rien effacer, en parler.")
    images = lister_images(a.raw)
    images_releve = releve(images)
    for nom, t in (("nnunet", tn), ("segresnet", ts)):
        if {c: x["sha256"] for c, x in t["images_test"].items()} != {c: x["sha256"] for c, x in images_releve.items()}:
            e6.arret(19, f"scanners de test différents de ceux prédits par « {nom} ».")
    # Même commit dans les deux traces et le dépôt courant : vérifié au démarrage (verifier_commit, arrêt 24).
    ouvrir(a, journal)

    # Prédictions : A, N0, S0 (phase B), T (étape 4 refaite selon D44, empreintes vérifiées au point 7)
    predictions = {"A": tn["etapes"]["A"]["fichiers"], "N0": tn["etapes"]["N0"]["fichiers"], "S0": ts["etapes"]["S0"]["fichiers"],
                   "T": verifs["totalsegmentator"]}
    for m in ("A", "N0", "S0"):
        dossier = os.path.join(a.sortie, "predictions", m)
        trouves = fichiers_predits(dossier)
        for c, p in trouves.items():
            if os.path.abspath(p) != predictions[m][c]["chemin"] or e6.sha256(p, PHASE) != predictions[m][c]["sha256"]:
                e6.arret(19, f"{m} {c} : fichier différent de celui produit ({p}). Ne rien effacer, en parler.")

    # R0 (fait foi) puis contrôle de plausibilité, un scanner à la fois
    with open(os.path.join(e6.DEPOT, e6.FICHIER_SEUILS), encoding="utf-8") as f:
        seuils = json.load(f)                             # empreinte et statut vérifiés au point 2
    mesures, defauts = {}, []
    for m in e6.MODELES_PREDICTION:
        for c in e6.CAS_TEST_ATTENDUS:
            image = e6.verifier_chemin(images[c], PHASE)
            masque = e6.verifier_chemin(predictions[m][c]["chemin"], PHASE)
            print(f"  Mesures {m} {c}...", flush=True)
            x = ctl.mesurer(c, image, masque)
            mesures[(m, c)] = x
            defauts += [f"{m} {c} : {p}" for p in x["r0"]] + ([] if x["mesurable"] else [f"{m} {c} : non mesurable"])
    if defauts:
        e6.arret(20, "R0 en ÉCHEC (défaut technique), rien n'est écrit :\n  - " + "\n  - ".join(defauts)
                 + "\nNe rien corriger soi-même, en parler d'abord. A, N0, S0 : correction du code puis nouvelle "
                 "prédiction (avant l'ouverture, consignée). T : ne rien corriger soi-même, en parler : la correction "
                 "de T relève d'une déviation (§ 10.2 B.3, D43) ; ce script ne la fait pas.")
    print(f"R0 : 20 prédictions conformes (forme, affine à {ctl.TOLERANCE_R0_MM} mm, étiquettes 0 à 7).")
    os.makedirs(os.path.join(a.sortie, "controle"), exist_ok=True)
    controle = {}
    print(f"\n{'modèle':<7}{'cas':<6}{'statut':<8}règles déclenchées")
    for m in e6.MODELES_PREDICTION:
        contenu = {"modele": m, "seuils_sha256": verifs["seuils_sha256"], "statut_alertes": STATUT_ALERTES[m],
                   "objet": "contrôle de plausibilité de la phase B (protocole § 7.1), avant toute lecture de labelsTs",
                   "regles_appliquees": list(ctl.TOUTES), "serialisation": SERIALISATION, "cas": {}}
        for c in e6.CAS_TEST_ATTENDUS:
            r = ctl.appliquer(mesures[(m, c)], seuils, ctl.TOUTES)
            contenu["cas"][c] = {"mesures": en_json(mesures[(m, c)]), "resultat": en_json(r), "resume": ctl.resume(r)}
            print(f"{m:<7}{c:<6}{r['statut']:<8}{ctl.resume(r)}")
        chemin = os.path.join(a.sortie, "controle", f"controle_{m}.json")
        e6.verifier_chemin(chemin, PHASE)
        ecrire_sans_ecraser(chemin, contenu)
        controle[m] = {"chemin": chemin}

    # Manifeste
    reglages = {**tn["reglages_inference"], **ts["reglages_inference"],
                "T": {"source": "TotalSegmentator 2.18.0, tâche heartchambers_highres, prédit à l'étape 4 (D19), puis "
                                "refait selon D44 (fichiers de l'étape 4 perdus) avec les mêmes version, tâche et "
                                "réglages, par 04c (GPU noté dans tracabilite_etape4) ; pas de nouvelle prédiction "
                                "dans l'étape 6", "tracabilite_etape4": e6.lire_json(os.path.join(a.predictions_ts, e6.NOM_TRACE_ETAPE4), PHASE)}}
    reglages["differences_N0_S0"] = differences(reglages["N0"], reglages["S0"])
    if reglages["differences_N0_S0"]:
        print("\nLIMITE de la comparaison (b) (P7, § 12) : réglages d'inférence différents entre N0 et S0, NON alignés :")
        for d in reglages["differences_N0_S0"]:
            print(f"  - {d['reglage']} : N0 {d['N0']} ; S0 {d['S0']}")
    versions = {"A_N0": tn["versions"], "S0": ts["versions"], "controle": e6.versions_bibliotheques(),
                "T": "TotalSegmentator 2.18.0 (étape 4 refaite selon D44, voir reglages_inference.T)"}
    extra = {"A_brut": tn["etapes"]["A_brut"]["fichiers"], "images_test": images_releve,
             "traces_06c": {n: {"chemin": c, "sha256": e6.sha256(c, PHASE)} for n, c in (("nnunet", chemin_tn), ("segresnet", chemin_ts))},
             "statut_alertes": STATUT_ALERTES, "format_controle": SERIALISATION,
             "verifications_demarrage": verifs, "tracabilite_06c": e6.tracabilite(PHASE, f"{SCRIPT}_controle"),
             "sessions_06c": {"nnunet": tn["sessions"], "segresnet": ts["sessions"]},
             "depot_phase_B": dict(a.depot, verification="même commit, dépôt propre : trace_nnunet, trace_segresnet "
                                                         "et session controle (arrêt 24 sinon)")}
    s = e6.ecrire_manifeste(chemin_manifeste, releves_modeles, predictions, controle, versions, reglages, extra)
    afficher_reference(chemin_manifeste, s)


def reafficher_manifeste(a, chemin):
    """Relance de « controle » une fois le manifeste écrit : rien n'est recalculé. Affiche le commit inscrit
    dans le manifeste à côté du commit courant ; si la référence du dépôt (config/empreinte_manifeste_etape6.json)
    existe, compare son empreinte à celle du manifeste (écart = arrêt 17, comme verifier_manifeste) ; sinon,
    réaffiche le bloc à recopier."""
    s = e6._sha256_brut(chemin)
    print(f"\nLe manifeste existe déjà (statut figé) : rien n'est recalculé.\n  SHA-256 du manifeste : {s}")
    try:
        m = e6.lire_json(chemin, PHASE)
    except (OSError, ValueError) as err:
        m = err
    if not isinstance(m, dict):
        e6.arret(e6.CODES["manifeste"], f"manifeste illisible ({chemin}) : {m}. Ne rien effacer, en parler.")
    inscrit = (m.get("depot_phase_B") or {}).get("commit") or m.get("commit") or "non noté"
    print(f"  Commit inscrit dans le manifeste (phase B) : {inscrit}\n  Commit courant du dépôt                    : "
          f"{a.depot['commit']}" + ("" if inscrit == a.depot["commit"] else
                                    "  (différent : normal si des commits ont suivi la phase B, ex. la référence)"))
    ref_chemin = os.path.join(e6.DEPOT, e6.REF_MANIFESTE)
    if not os.path.isfile(ref_chemin):
        afficher_reference(chemin, s)
        return
    try:
        with open(ref_chemin, encoding="utf-8") as f:
            attendu = json.load(f).get("manifeste_sha256")
    except (OSError, ValueError, AttributeError) as err:
        e6.arret(e6.CODES["manifeste"], f"{e6.REF_MANIFESTE} illisible ({err}).")
    if s != attendu:
        e6.arret(e6.CODES["manifeste"], f"SHA-256 du manifeste {s} ≠ référence du dépôt {e6.REF_MANIFESTE} ({attendu}). "
                 "Le manifeste ou la référence a changé : ne rien effacer ni recopier, en parler.")
    print(f"  Référence du dépôt ({e6.REF_MANIFESTE}) : empreinte IDENTIQUE au manifeste.")
    print(f"Résultat : phase B TERMINÉE (manifeste conforme à la référence) ; fin : {e6.heure()}")


def afficher_reference(chemin, sha):
    print(f"\n===== DÉBUT du contenu de {e6.REF_MANIFESTE} (à recopier tel quel) =====")
    print(json.dumps(e6.contenu_reference_manifeste(chemin, sha), ensure_ascii=False, indent=2))
    print(f"===== FIN du contenu de {e6.REF_MANIFESTE} =====")
    print(f"\nLigne proposée pour le journal (complément de D42) : manifeste de la phase B écrit ({chemin}), SHA-256 {sha} ; "
          "20 prédictions, 4 résultats du contrôle et les poids des modèles figés avant l'ouverture (§ 10.2).")
    print(f"Résultat : phase B TERMINÉE ; fin : {e6.heure()}")


def main(argv=None):
    a = lire_arguments(argv)
    journal, verifs = demarrer(a)
    {"nnunet": sous_commande_nnunet, "segresnet": sous_commande_segresnet,
     "controle": sous_commande_controle}[a.sous_commande](a, journal, verifs)


if __name__ == "__main__":
    main()
