"""
Étape 6, phase C — OUVERTURE DU LOT DE TEST, UNE SEULE FOIS (protocole § 10.1 R-O1, § 10.2 C, § 11).
C'est le SEUL script qui lit labelsTs (la vérité terrain des 5 cas de test : 1004, 1010, 1011, 1013, 1014).

Ce que fait le script, dans cet ordre (tout ce qui précède l'étape 7 ne lit AUCUNE vérité du test) :
  1. vérifications au démarrage du § 9 : points 1 à 4 (dépôt seulement) et point 5 (MONAI 1.6.1) ;
  2. dépôt au commit annoncé : --commit (noté au journal avant la phase C) doit être le début du commit
     actuel, et le dépôt doit être propre (aucun fichier modifié) ; --sortie doit être EXACTEMENT le dossier
     phaseC voisin du dossier du manifeste (<…>/etape6/phaseC si le manifeste est <…>/etape6/phaseB/…), et le
     manifeste doit être au chemin inscrit dans config/empreinte_manifeste_etape6.json : le registre de
     lecture unique est donc toujours au même endroit, quel que soit --sortie (code 22 sinon) ;
  3. MANIFESTE de la phase B vérifié AVANT TOUT AUTRE FICHIER DE DONNÉES (verifier_manifeste) : référence
     config/empreinte_manifeste_etape6.json absente, empreinte du manifeste différente, ou UN SEUL fichier
     listé (20 prédictions, 4 résultats du contrôle, modèles) modifié = arrêt (code 17) ;
  4. le code de calcul n'a pas changé depuis la phase B : « git diff --name-only <commit du manifeste> HEAD
     -- scripts/ » doit être vide (code 22 sinon) ;
     point 7 : les 5 prédictions de TotalSegmentator et tracabilite_etape4.json ont toujours l'empreinte de
     la phase 0 bis (config/empreintes_totalseg_etape4.json) ; point 6 : les empreintes des modèles sont
     contrôlées par le manifeste ;
  5. lit les 4 résultats du contrôle de plausibilité (contrat dans etape6_metriques.py) et vérifie, sur les
     EN-TÊTES seulement, que chaque prédiction a la géométrie de son scanner (imagesTs) ; une incohérence
     non signalée par R0 dans le contrôle figé = arrêt (code 20), avant l'ouverture ;
  6. REGISTRE D'OUVERTURE (phaseC/registre_ouverture.json) : si une évaluation complète existe (dossier
     phaseC/resultats/, OU une tentative « terminée » au registre, même si resultats/ a disparu), refus
     (code 21). Si une tentative s'est arrêtée sur un DÉFAUT du code ou des données (statut « arret_defaut »),
     toute relance est refusée, même avec --relance-apres-panne (code 22) : la suite est une déviation
     consignée au journal (§ 11) et une correction relue par QA. Si une tentative précédente est restée
     incomplète pour une cause extérieure (session coupée...), relance refusée sauf avec --relance-apres-panne
     "cause extérieure", et seulement au MÊME commit que la première tentative (§ 11 ; code 22). Le registre
     est écrit (tentative n°, heures, commit, cause, chemin de la copie de l'affichage) AVANT la première
     lecture de labelsTs ; puis la garde de labelsTs est ouverte ;
  7. lit les 5 masques de labelsTs (empreinte SHA-256 de chacun relevée), et calcule d'un coup les trois
     comparaisons (etape6_metriques : métriques § 5-6, agrégats § 5.3, classement des alertes et échecs
     silencieux § 7.3, verdict § 8.4, règle P6) ; densités de la vérité mesurées par le module figé si une
     alerte R6 ou R7 doit être classée ;
  8. écrit TOUTES les sorties en une fois (dossier .partiel puis renommage en phaseC/resultats/), avec la
     traçabilité, puis marque la tentative « terminée » dans le registre.

Entrées (chemins Drive par défaut) :
  --commit                OBLIGATOIRE : commit noté au journal pour la phase C (7 caractères ou plus)
  --raw                   nnUNet_raw (imagesTs et labelsTs de Dataset101_MMWHS)   défaut /content/drive/MyDrive/poc_ct/nnunet_raw
  --manifeste             manifeste de la phase B             défaut .../poc_ct/etape6/phaseB/manifeste_etape6.json
  --predictions-totalseg  prédictions de l'étape 4            défaut .../poc_ct/predictions/totalsegmentator
  --sortie                dossier de la phase C               défaut .../poc_ct/etape6/phaseC
  --relance-apres-panne   "CAUSE" : relance après une panne EXTÉRIEURE au code (§ 11), cause notée au registre
  --ignorer-verrou        comme 05c
Sorties (dans --sortie, Drive) : registre_ouverture.json ; resultats/ (metriques_par_cas.csv,
  dice_moyen_par_cas.csv, agregats.csv, alertes.csv, echecs_silencieux.csv, comparaison_b_p6.csv,
  resultats_etape6.json, rapport_etape6.md, tracabilite_06d.json) ; sortie_06d_<SESSION>.txt ;
  resultats.tentativeN.partiel/ si une tentative a été coupée pendant l'écriture, ou si elle s'est arrêtée
  sur un défaut (fichier partiel_non_classe.json : métriques et empreintes déjà calculées, marquées
  « PARTIEL, NON CLASSÉ ») ; conservé, jamais effacé ; resultats/ n'est alors JAMAIS créé.
Matériel et durée : pas de GPU (environnement processeur, MONAI 1.6.1 installé). Environ 10 à 20 min :
  5 cas × (7 + 7 + 7 + 5) = 130 structures à 2 à 5 s (HD95 et NSD), plus la lecture des fichiers sur Drive et,
  s'il le faut, les densités de la vérité (au plus 5 mesures).
Codes de sortie :
  0 = évaluation complète écrite ; 3 = interrompu APRÈS l'ouverture (registre à jour) : NE RIEN CHANGER au
  code, relancer avec --relance-apres-panne "cause" (au même commit) ou en parler s'il s'agit d'un défaut
  du code (déviation, § 11) ; 1 = garde-fou ; 10 à 14 = vérification au démarrage ; 16 = TotalSegmentator
  modifié ; 17 = manifeste ; 18 = lecture refusée par la garde ; 20 = contrôle ou géométrie incohérents
  (avant l'ouverture) ; 21 = évaluation complète déjà faite (resultats/ présent ou tentative « terminée ») :
  lecture unique, refus ; 22 = lancement ou relance refusés (option absente, commit différent, dépôt
  modifié, scripts/ modifié depuis la phase B, --sortie ou manifeste à un autre endroit, tentative
  précédente arrêtée sur un défaut) ; 23 = APRÈS l'ouverture, une alerte ne peut pas être classée pour un
  motif autre que « non classable » (définition D43 : règle inconnue, structure non mesurée, densités de la
  vérité non mesurées...) : défaut du code ou des données. Code 23, ou tout autre arrêt codé APRÈS
  l'ouverture pendant le calcul (ex. 18, garde) : registre « arret_defaut », aucun résultat classé (partiels
  conservés à part) ; ce n'est PAS une panne extérieure : aucune relance possible, déviation consignée au
  journal (§ 11), correction relue par QA.

Vocabulaire :
  - lecture unique (R-O1) : labelsTs n'est lu que par UNE exécution de CE script ; une relance n'est permise
    qu'après une panne extérieure au code (session Colab coupée, Drive indisponible, mémoire) ;
  - registre d'ouverture : petit fichier qui trace chaque tentative AVANT de lire la vérité ;
  - commit : identifiant de la version exacte du code ; « propre » = aucun fichier modifié depuis ce commit.
"""
import argparse, json, os, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import etape6_commun as e6                        # n'exécute rien à l'import
import etape6_metriques as em

PHASE, SCRIPT = "C", "06d"
DS = "Dataset101_MMWHS"
ETIQUETTES = {"A": em.TOUTES_ETIQUETTES, "N0": em.TOUTES_ETIQUETTES, "S0": em.TOUTES_ETIQUETTES, "T": em.ETIQUETTES_C}
REMARQUES = {"1014": "seul cas à 0,45 mm (D20)"}


def lire_arguments():
    p = argparse.ArgumentParser(description="Phase C : ouverture du lot de test (une seule fois).")
    p.add_argument("--commit", required=True)
    p.add_argument("--raw", default=f"{e6.DRIVE}/nnunet_raw")
    p.add_argument("--manifeste", default=os.path.join(e6.DOSSIERS_PHASES["B"], e6.NOM_MANIFESTE))
    p.add_argument("--predictions-totalseg", default=e6.PREDICTIONS_TOTALSEG)
    p.add_argument("--sortie", default=e6.DOSSIERS_PHASES[PHASE])
    p.add_argument("--relance-apres-panne", default=None, metavar="CAUSE")
    p.add_argument("--ignorer-verrou", action="store_true")
    return p.parse_args()


def verifier_commit(attendu):
    actuel = e6.commit_git()
    if len(attendu) < 7 or not actuel.split("+")[0].startswith(attendu) or "+" in actuel:
        e6.arret(22, f"dépôt au commit {actuel}, attendu {attendu} (noté au journal) et sans fichier modifié. "
                     "Section 2 du notebook : récupérer exactement ce commit, puis relancer.")
    return actuel


def verifier_scripts_inchanges(commit_b):
    """Le code de calcul (scripts/) est identique à celui de la phase B (commit noté dans le manifeste)."""
    import subprocess
    c = str(commit_b or "").split("+")[0]
    if not c:
        e6.arret(22, "commit de la phase B absent du manifeste : impossible de vérifier que scripts/ n'a pas changé.")
    try:
        r = subprocess.run(["git", "diff", "--name-only", c, "HEAD", "--", "scripts/"], cwd=e6.DEPOT,
                           capture_output=True, text=True, timeout=30)
    except Exception as err:                      # git absent, délai dépassé...
        e6.arret(22, f"impossible de comparer scripts/ au commit de la phase B ({c}) : {err}")
    if r.returncode != 0:
        e6.arret(22, f"commit de la phase B {c!r} (manifeste) introuvable dans le dépôt : {r.stderr.strip()}")
    if r.stdout.strip():
        e6.arret(22, f"scripts/ a changé depuis la phase B (commit {c[:12]}) : {r.stdout.split()}. Le code de "
                     "calcul doit être celui de la phase B ; en parler (déviation, § 11). Rien n'est lu.")
    print(f"Code de calcul (scripts/) identique au commit de la phase B ({c[:12]}).")


def verifier_emplacements(manifeste, sortie):
    """Lecture unique indépendante de --sortie : le registre est TOUJOURS dans <dossier etape6>/phaseC, où
    <dossier etape6> est le parent du dossier phaseB du manifeste ; et le manifeste est au chemin inscrit dans
    le dépôt (config/empreinte_manifeste_etape6.json). Sinon arrêt 22 ; aucune donnée lue. Renvoie phaseC."""
    norm = lambda x: os.path.normcase(os.path.realpath(os.path.abspath(x)))
    manifeste = os.path.abspath(manifeste)
    attendu = os.path.join(os.path.dirname(os.path.dirname(manifeste)), "phaseC")
    if os.path.basename(os.path.dirname(manifeste)) != "phaseB":
        e6.arret(22, f"le manifeste doit être dans un dossier phaseB (…/etape6/phaseB/{e6.NOM_MANIFESTE}) : {manifeste}.")
    if norm(sortie) != norm(attendu):
        e6.arret(22, f"--sortie {sortie} ≠ {attendu} (dossier phaseC voisin du manifeste) : le registre de lecture "
                     "unique est toujours à cet endroit. Retirer --sortie ou donner ce dossier. Rien n'est lu.")
    ref = os.path.join(e6.DEPOT, e6.REF_MANIFESTE)
    if os.path.isfile(ref):                       # absent : verifier_manifeste arrête ensuite (code 17)
        with open(ref, encoding="utf-8") as f:
            inscrit = json.load(f).get("manifeste_chemin_drive")
        if not inscrit or norm(inscrit) != norm(manifeste):   # chemin absent de la référence = arrêt aussi
            e6.arret(22, f"--manifeste {manifeste} : chemin absent de {e6.REF_MANIFESTE} ou différent ({inscrit}). Rien n'est lu.")
    return attendu


ARRET_DEFAUT = "arret_defaut"
TEXTE_SUITE_DEFAUT = "suite = déviation consignée au journal (§ 11), correction relue par QA"


def preparer_registre(chemin, resultats, commit, cause, copie_sortie=None):
    """Tentative n° suivante, ou refus. Ne lit aucune donnée du test."""
    if os.path.isdir(resultats):
        e6.arret(21, f"une évaluation COMPLÈTE existe déjà ({resultats}) : lecture unique (R-O1), refus. Rien n'est lu.")
    reg = e6.lire_json(chemin, PHASE) if os.path.isfile(chemin) else {"objet": "registre d'ouverture du lot de test "
                                                                     "(protocole § 10.1 R-O1, § 11)", "tentatives": []}
    t = reg["tentatives"]
    finies = [x.get("numero") for x in t if x.get("statut") == "terminée"]
    if finies:
        e6.arret(21, f"tentative n° {finies[0]} « terminée » au registre ({chemin}) : lecture unique (R-O1), refus, "
                     f"même si {resultats} a disparu. Rien n'est lu.")
    defauts = [x.get("numero") for x in t if x.get("statut") == ARRET_DEFAUT          # (ancienne écriture d'un
               or str(x.get("erreur", "")).startswith("ArretClassement")]          #  arrêt 23 : même refus)
    if defauts:
        e6.arret(22, f"tentative n° {defauts[0]} arrêtée sur un DÉFAUT du code ou des données (statut "
                     f"« {ARRET_DEFAUT} » ou arrêt 23 noté) : relance refusée, même avec --relance-apres-panne ; {TEXTE_SUITE_DEFAUT}. "
                     "Rien n'est lu.")
    if t and not cause:
        e6.arret(22, f"tentative précédente INCOMPLÈTE ({len(t)} au registre, dernière : {t[-1].get('statut')}). Relance "
                     "permise seulement après une panne EXTÉRIEURE au code (§ 11), avec --relance-apres-panne \"cause\". "
                     "Un défaut du code = déviation : en parler avant toute relance. Rien n'est lu.")
    if t and t[0]["commit"] != commit:
        e6.arret(22, f"relance refusée : commit {commit} ≠ commit de la première tentative {t[0]['commit']} (§ 11).")
    if not t and cause:
        e6.arret(22, "--relance-apres-panne donné alors qu'aucune tentative n'existe : retirer l'option.")
    t.append({"numero": len(t) + 1, **{f"heure_{k}": v for k, v in e6.maintenant().items()}, "commit": commit,
              "session": e6.SESSION, "cause_relance": cause, "copie_affichage": copie_sortie,
              "statut": "commencée : lecture de labelsTs autorisée"})
    e6.ecrire_json_atomique(chemin, reg)
    return reg


def maj_registre(chemin, reg, **champs):
    reg["tentatives"][-1].update(champs, **{f"fin_{k}": v for k, v in e6.maintenant().items()})
    e6.ecrire_json_atomique(chemin, reg)


def main():
    journal = e6.demarrer_sortie()
    t0 = time.time()
    print(f"Étape 6, phase C (06d) — OUVERTURE DU LOT DE TEST — session {e6.SESSION} ; début : {e6.heure()}")
    a = lire_arguments()
    commit = verifier_commit(a.commit)
    print(f"Commit du dépôt : {commit} (conforme à --commit {a.commit})")
    sortie = verifier_emplacements(a.manifeste, e6.verifier_hors_depot(a.sortie, "--sortie"))   # registre fixe
    verifs = e6.verifier_demarrage(PHASE)
    versions = e6.verifier_versions(monai=True)
    seuils = e6.lire_json(os.path.join(e6.DEPOT, e6.FICHIER_SEUILS))
    cas = list(e6.CAS_TEST_ATTENDUS)

    # 3. Manifeste AVANT tout autre fichier de données.
    m = e6.verifier_manifeste(a.manifeste)
    sha_manifeste = e6.sha256(a.manifeste, PHASE)              # pour la traçabilité (déjà comparée au dépôt)
    verifier_scripts_inchanges(m.get("commit"))                # même code de calcul qu'en phase B
    # 4. Point 7 (TotalSegmentator) ; les modèles sont contrôlés par le manifeste (point 6).
    t_orig = e6.verifier_empreintes_totalseg(a.predictions_totalseg, PHASE)
    t_corriges = [c for c in cas if os.path.abspath(m["predictions"]["T"][c]["chemin"]) != t_orig[c]["chemin"]]
    if t_corriges:
        print(f"  T corrigé en phase B pour {t_corriges} : fichiers du manifeste utilisés, originaux inchangés (point 7).")
    print(f"  6. empreintes des {len(m['modeles'])} fichiers de modèles identiques au manifeste.")

    # 5. Contrôles figés et géométrie (en-têtes) : avant l'ouverture.
    controles, pb = {}, []
    for mod in e6.MODELES_PREDICTION:
        try:
            lu = em.lire_controle(e6.lire_json(m["controle"][mod]["chemin"], PHASE), mod, cas, seuils, e6.SHA256_SEUILS)
        except ValueError as err:
            e6.arret(20, str(err))
        controles[mod] = {c: x["resultat"] for c, x in lu.items()}
        for c in cas:
            pred = m["predictions"][mod][c]["chemin"]
            if os.path.basename(str(lu[c]["mesures"].get("masque", ""))) != os.path.basename(pred):
                pb.append(f"{mod} {c} : contrôle mesuré sur {lu[c]['mesures'].get('masque')} ≠ prédiction du manifeste {pred}")
            ok, motif = em.geometrie(e6.ouvrir_nifti(pred, PHASE), e6.ouvrir_nifti(image(a, c), PHASE))
            r0 = any(x["regle"] == "R0" for x in controles[mod][c]["regles_declenchees"])
            if not ok and not r0:
                pb.append(f"{mod} {c} : géométrie différente du scanner ({motif}) sans alerte R0 dans le contrôle figé")
    if pb:
        e6.arret(20, "incohérences avant l'ouverture (rien n'est lu dans labelsTs) :\n  - " + "\n  - ".join(pb))
    print("Contrôles de plausibilité figés relus (R0 à R8, 4 modèles) ; géométries des 20 prédictions vérifiées.")

    # 6. Registre d'ouverture, écrit AVANT la première lecture de labelsTs.
    os.makedirs(sortie, exist_ok=True)
    copie = e6.ouvrir_sortie(journal, sortie, SCRIPT)
    print(f"Copie de l'affichage : {copie}")
    e6.poser_verrou(sortie, "verrou_06d", a.ignorer_verrou)
    chemin_reg, resultats = os.path.join(sortie, e6.NOM_REGISTRE), os.path.join(sortie, "resultats")
    reg = preparer_registre(chemin_reg, resultats, commit, a.relance_apres_panne, copie)
    n = reg["tentatives"][-1]["numero"]
    print(f"Registre d'ouverture écrit : tentative n° {n} ({e6.heure()}). OUVERTURE DU LOT DE TEST.")
    e6.autoriser_ouverture_test(chemin_reg)

    # 7. Lecture unique de labelsTs et calcul d'un coup.
    etat = {}                                     # rempli au fil du calcul : gardé (partiel) en cas d'arrêt
    try:
        res, empreintes, anomalies = calculer(a, m, cas, controles, seuils, etat)
        trace = {**e6.tracabilite(PHASE, SCRIPT), "tentative": n, "registre": reg, "manifeste_sha256": sha_manifeste,
                 "manifeste": a.manifeste, "empreintes_labelsTs": empreintes, "verifications_demarrage": verifs,
                 "versions_verifiees": versions, "totalseg_corriges_en_phase_B": t_corriges, "anomalies": anomalies,
                 "duree_s": time.time() - t0, "definition_hd95": em.DEFINITION_HD95}
        res["anomalies"] = anomalies
        noms = em.publier(resultats, res, "Étape 6 : évaluation sur le lot de test (lecture unique)", "",
                          {"tracabilite_06d.json": trace}, tmp=f"{resultats}.tentative{n}.partiel")
    except BaseException as err:                  # panne : registre à jour, résultats partiels conservés
        if isinstance(err, SystemExit) and err.code == 0:
            raise
        # Défaut du code ou des données (D43) : ArretClassement (23), DefautCalcul (incohérence de calcul)
        # ou arrêt codé (sys.exit non nul, ex. 18).
        if isinstance(err, (em.ArretClassement, em.DefautCalcul)) or (isinstance(err, SystemExit) and err.code not in (None, 0)):
            code = getattr(err, "code", None) if isinstance(getattr(err, "code", None), int) else (23 if isinstance(err, em.DefautCalcul) else 1)
            partiel = garder_partiel(resultats, n, etat, code, err)
            maj_registre(chemin_reg, reg, statut=ARRET_DEFAUT,
                         erreur=f"arrêt {code} : défaut du code ou des données (déviation § 11) — "
                                f"{type(err).__name__} : {err}",
                         suite=TEXTE_SUITE_DEFAUT, resultats_partiels=partiel, trace=traceback.format_exc()[-3000:])
            print(f"\nARRÊT (code {code}) APRÈS L'OUVERTURE : {err}\nRegistre mis à jour (tentative {n}, statut "
                  f"« {ARRET_DEFAUT} ») ; aucun résultat classé. Partiels non classés : {partiel}.\nCe n'est PAS "
                  f"une panne extérieure : AUCUNE relance possible ; {TEXTE_SUITE_DEFAUT}.")
            sys.exit(code)
        maj_registre(chemin_reg, reg, statut="interrompue", erreur=f"{type(err).__name__}: {err}",
                     trace=traceback.format_exc()[-3000:])
        print(f"\nINTERROMPU APRÈS L'OUVERTURE ({type(err).__name__}: {err}). Registre mis à jour (tentative {n}).\n"
              "NE RIEN CHANGER AU CODE. Panne extérieure (session, Drive, mémoire) : relancer avec "
              "--relance-apres-panne \"cause\". Défaut du code : déviation (§ 11), en parler.")
        traceback.print_exc()
        sys.exit(3)
    sha_res = e6.sha256(os.path.join(resultats, "resultats_etape6.json"), PHASE)
    maj_registre(chemin_reg, reg, statut="terminée", resultats=resultats, sha256_resultats_json=sha_res)

    # Résumé affiché.
    v, ea = res["verdict"], res["comparaisons"]["a"]["A"]["estimation_fausses_alertes"]
    print(f"\nÉcrit en une fois dans {resultats} : {', '.join(noms)}")
    print(f"Empreintes des 5 masques de labelsTs : " + " ; ".join(f"{c} {s}" for c, s in empreintes.items()))
    for x in anomalies:
        print(f"ANOMALIE : {x}")
    print("\n" + "=" * 100)
    for x in v["C1"]["structures"]:
        print(f"  C1 {x['structure']:<12} médiane du Dice {x['mediane'] if x['mediane'] is not None else '—'} "
              f"(n = {x['n']}) ≥ {x['seuil']} : {'oui' if x['rempli'] else 'NON'}")
    print(f"  C2 aucun échec silencieux : {'oui' if v['C2']['rempli'] else 'NON'}")
    print(f"  Fausses alertes (a), « erreur probable » : {ea['erreur_probable']['texte']} ; hors domaine : {ea['hors_domaine']['texte']}")
    print(f"VERDICT (pipeline A, § 8.4) : {v['verdict'].upper()}" + ("" if not v["raisons"] else " — " + " ; ".join(v["raisons"])))
    print(f"SHA-256 de resultats_etape6.json : {sha_res}")
    print(f"Durée : {(time.time() - t0) / 60:.1f} min ; fin : {e6.heure()}\nRésultat : évaluation COMPLÈTE (tentative {n}).")


def garder_partiel(resultats, n, etat, code, err):
    """Arrêt sur défaut après l'ouverture : écrit ce qui est déjà calculé (métriques, morceaux égarés, densités,
    empreintes des masques de labelsTs) dans <resultats>.tentativeN.partiel/partiel_non_classe.json, marqué
    « PARTIEL, NON CLASSÉ ». Ne crée JAMAIS resultats/. Renvoie le chemin écrit (None si l'écriture échoue)."""
    try:
        dossier = f"{resultats}.tentative{n}.partiel"
        os.makedirs(dossier, exist_ok=True)
        chemin = os.path.join(dossier, "partiel_non_classe.json")
        if os.path.exists(chemin):                # jamais écrasé
            chemin = os.path.join(dossier, f"partiel_non_classe_{e6.SESSION}.json")
        e6.ecrire_json_atomique(chemin, em._sans_nan({
            "statut": "PARTIEL, NON CLASSÉ : ne pas utiliser comme résultat de l'étape 6",
            "raison": f"arrêt {code} après l'ouverture : défaut du code ou des données (déviation § 11) — {err}",
            "tentative": n, **{f"heure_{k}": v for k, v in e6.maintenant().items()}, **etat}))
        return chemin
    except Exception as e:                        # l'écriture des partiels ne doit pas masquer l'arrêt
        print(f"ANOMALIE : résultats partiels non écrits ({type(e).__name__}: {e})")
        return None


def image(a, c):
    return os.path.join(a.raw, DS, "imagesTs", f"mmwhs_{c}_0000.nii.gz")


def calculer(a, m, cas, controles, seuils, etat=None):
    """Lit labelsTs (un cas à la fois) et calcule tout. Renvoie (résultats, empreintes de labelsTs, anomalies).
    etat (dict) reçoit au fil du calcul ces mêmes objets : ils sont gardés comme partiels si un arrêt survient."""
    import numpy as np
    met, egares, vdens, empreintes, anomalies = {k: {} for k in ETIQUETTES}, {k: {} for k in ETIQUETTES}, {}, {}, []
    if etat is not None:
        etat.update({"empreintes_labelsTs": empreintes, "metriques": met, "egares": egares,
                     "verites_densites": vdens, "anomalies": anomalies})
    for c in cas:
        verite = os.path.join(a.raw, DS, "labelsTs", f"mmwhs_{c}.nii.gz")
        empreintes[c] = e6.sha256(verite, PHASE)
        print(f"  {c} : vérité lue (SHA-256 {empreintes[c][:12]}…), Dice, HD95, NSD des 4 modèles...", flush=True)
        img, ref_img = e6.ouvrir_nifti(image(a, c), PHASE), e6.ouvrir_nifti(verite, PHASE)
        ok, motif = em.geometrie(img, ref_img)
        if not ok:
            anomalies.append(f"{c} : vérité de géométrie différente du scanner ({motif}) ; volumes avec l'espacement du scanner")
        ref = np.asanyarray(ref_img.dataobj)
        if not np.array_equal(ref, np.round(ref)) or not set(np.unique(ref).astype(int).tolist()) <= set(range(8)):
            anomalies.append(f"{c} : valeurs de la vérité hors des étiquettes entières 0 à 7")
        ref = ref.astype(np.uint8)
        esp, forme = tuple(float(x) for x in img.header.get_zooms()[:3]), tuple(img.shape[:3])
        for mod, etiq in ETIQUETTES.items():
            pimg = e6.ouvrir_nifti(m["predictions"][mod][c]["chemin"], PHASE)
            gok, gmotif = em.geometrie(pimg, ref_img)
            if not gok:
                anomalies.append(f"{mod} {c} : prédiction de géométrie différente de la vérité ({gmotif}) : D1 (§ 6)")
            pred = np.asanyarray(pimg.dataobj).astype(np.uint8) if gok else np.zeros_like(ref)
            met[mod][c], egares[mod][c] = em.mesurer_prediction(pred, ref, esp, forme, etiq, controles[mod][c],
                                                                mod == "T", gok, gmotif)
            anomalies += [f"{mod} {c} {em.NOMS[k]} : {x['anomalie']}" for k, x in met[mod][c].items() if x["anomalie"]]
            del pred
        if any(em.besoins_classement(em.restreindre_T(controles[mod][c]) if mod == "T" else controles[mod][c])[1]
               for mod in ETIQUETTES):
            e6.verifier_chemin(image(a, c), PHASE), e6.verifier_chemin(verite, PHASE)   # le module ouvre lui-même
            vdens[c] = em.normaliser_mesures(em.cp.mesurer(c, image(a, c), verite))
        del ref
    entrees = {"cas": cas, "metriques": met, "controles": controles, "egares": egares, "verites_densites": vdens,
               "seuils": seuils, "remarques_cas": REMARQUES}
    return em.evaluer(entrees), empreintes, anomalies


if __name__ == "__main__":
    main()
