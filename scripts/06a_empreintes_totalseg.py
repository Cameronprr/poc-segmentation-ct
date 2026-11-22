"""
Étape 6, phase 0 bis — Empreintes des 5 prédictions de TotalSegmentator (protocole § 10.2, D42 ; dossier D44).

Pourquoi : à l'étape 4, on a relevé l'empreinte des SCANNERS, pas celle des PRÉDICTIONS (protocole § 12). On la
relève maintenant, avant toute autre opération de l'étape 6. Ensuite, les phases B et C vérifient que ces
fichiers n'ont pas changé (protocole § 9, point 7).
D44 : les prédictions de l'étape 4 ont été perdues (Drive plein ; seul mmwhs_1004 a survécu). Elles sont
refaites par 04c (mêmes version, tâche et réglages ; section 26 bis) dans predictions/totalsegmentator_d44,
avec un nouveau tracabilite_etape4.json (même nom, contenu marqué D44). C'est CE dossier qui est relevé ici.

Ce que fait le script :
  1. vérifications au démarrage, points 1 à 4 du § 9 (protocole figé, seuils figés, quatre valeurs du fichier
     de seuils identiques au code, découpage et 5 cas de test). Elles ne lisent que des fichiers du dépôt ;
     en cas d'échec, rien n'est lu ni écrit sur Drive ;
  2. contrôle du dossier des prédictions : exactement mmwhs_1004.nii.gz ... mmwhs_1014.nii.gz (les 5 cas de
     test du découpage) et tracabilite_etape4.json. Un autre fichier NIfTI = arrêt ; un autre fichier non
     NIfTI = simple avertissement ;
  3. pour chacun de ces 6 fichiers : SHA-256 calculée sur les OCTETS du fichier (lecture binaire par blocs,
     sans décompresser ni décoder l'image), taille, date de modification (UTC et Paris) ;
  4. écrit <sortie>/empreintes_totalseg.json, puis affiche le bloc à recopier dans le dépôt
     (config/empreintes_totalseg_etape4.json) et une ligne pour le journal (complément de D42).
  Il ne lit JAMAIS imagesTs ni labelsTs, ni aucune vérité : toutes les lectures passent par la garde de
  etape6_commun (phase « 0bis »), qui refuserait ces chemins.

Entrées :
  --predictions  dossier des prédictions refaites selon D44 (défaut : /content/drive/MyDrive/poc_ct/predictions/totalsegmentator_d44)
  --sortie       dossier de la phase 0 bis (défaut : /content/drive/MyDrive/poc_ct/etape6/phase0bis)
  --ignorer-verrou  comme 05c (seulement si tu es SÛR qu'aucune autre session ne tourne)
Sorties (dans --sortie, Drive) :
  empreintes_totalseg.json ; sortie_06a_<SESSION>.txt (copie de l'affichage, heures UTC et Paris) ;
  verrou_06a (retiré à la fin).
  Si empreintes_totalseg.json existe déjà : mêmes valeurs → il est gardé tel quel (rien n'est réécrit) ;
  autres valeurs → ARRÊT (code 19), jamais d'écrasement.
Matériel et durée : pas de GPU (une session processeur suffit) ; moins d'une minute (6 petits fichiers).
Codes de sortie :
  0 = empreintes relevées (ou déjà relevées, identiques) ;
  1 = garde-fou : dossier introuvable, fichiers manquants ou en trop, --sortie dans le dépôt, verrou ;
  10 à 13 = vérification au démarrage en échec (10 protocole, 11 seuils, 12 quatre valeurs, 13 découpage) :
      rien n'est lu ni écrit sur Drive ;
  18 = lecture refusée par la garde (chemin de vérité ou d'image du test) : rien n'est lu ;
  19 = empreintes_totalseg.json existe déjà avec d'autres valeurs : NE RIEN EFFACER, envoyer la sortie.

Vocabulaire :
  - SHA-256 (empreinte) : 64 caractères calculés sur tous les octets du fichier ; un seul octet changé
    donne une empreinte différente ;
  - date de modification : date à laquelle Drive a enregistré le fichier pour la dernière fois ; depuis D44,
    elle doit dater de la nouvelle prédiction (04c, section 26 bis), et non plus de l'étape 4 (seule preuve
    d'intégrité avant cette phase, protocole § 12).
"""
import argparse, datetime, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import etape6_commun as e6                        # n'exécute rien à l'import

PHASE, SCRIPT = "0bis", "06a"


def lire_arguments():
    p = argparse.ArgumentParser(description="Phase 0 bis : empreintes des prédictions de TotalSegmentator.")
    p.add_argument("--predictions", default=e6.PREDICTIONS_TOTALSEG)
    p.add_argument("--sortie", default=e6.DOSSIERS_PHASES[PHASE])
    p.add_argument("--ignorer-verrou", action="store_true")
    return p.parse_args()


def date_fichier(horodatage):
    t = datetime.datetime.fromtimestamp(horodatage, datetime.timezone.utc)
    return t.isoformat(timespec="seconds"), t.astimezone(e6.suivi05c.PARIS).isoformat(timespec="seconds")


def relever(chemin):
    """Empreinte (binaire, via la garde), taille et date de modification d'un fichier."""
    s = e6.sha256(chemin, PHASE)
    st = os.stat(chemin)
    utc, paris = date_fichier(st.st_mtime)
    return {"nom": os.path.basename(chemin), "sha256": s, "taille_octets": st.st_size,
            "date_modification_utc": utc, "date_modification_paris": paris}


def controler_dossier(dossier, cas):
    """Liste les NOMS du dossier (aucun fichier ouvert) : 5 prédictions + traçabilité, rien d'autre en NIfTI."""
    if not os.path.isdir(dossier):
        e6.arret(1, f"dossier des prédictions introuvable : {dossier} (--predictions).")
    e6.verifier_chemin(dossier, PHASE)            # refuse un dossier imagesTs / labelsTs
    attendus = {e6.NOM_PREDICTION.format(cas=c) for c in cas} | {e6.NOM_TRACE_ETAPE4}
    presents = set(os.listdir(dossier))
    manquants = sorted(attendus - presents)
    nifti_en_trop = sorted(n for n in presents - attendus if re.search(r"\.nii(\.gz)?$", n, re.I))
    autres = sorted(presents - attendus - set(nifti_en_trop))
    if manquants or nifti_en_trop:
        e6.arret(1, f"contenu inattendu de {dossier} : manquants {manquants or 'aucun'} ; fichiers NIfTI en trop "
                    f"{nifti_en_trop or 'aucun'}. Attendu exactement {sorted(attendus)}.")
    if autres:
        print(f"ATTENTION : autres fichiers présents, ignorés (non lus) : {autres}")
    print(f"Dossier conforme : les 5 prédictions ({', '.join(cas)}) et {e6.NOM_TRACE_ETAPE4} sont présents.")


def valeurs_comparees(contenu):
    """Ce qui doit être identique entre deux relevés : chemin, et pour chaque fichier nom, empreinte, taille, date."""
    cles = ("nom", "sha256", "taille_octets", "date_modification_utc")
    f = {c: {k: v.get(k) for k in cles} for c, v in contenu["fichiers"].items()}
    t = {k: contenu["tracabilite_etape4"].get(k) for k in cles}
    return {"dossier_predictions": contenu["dossier_predictions"], "fichiers": f, "tracabilite_etape4": t}


def main():
    journal = e6.demarrer_sortie()                # affichage gardé en mémoire : rien n'est écrit avant l'étape 3
    print(f"Étape 6, phase 0 bis (06a) — session {e6.SESSION} ; début : {e6.heure()}")
    print(f"Commit du dépôt : {e6.commit_git()}")
    a = lire_arguments()
    predictions, sortie = os.path.abspath(a.predictions), e6.verifier_hors_depot(a.sortie, "--sortie")

    # 1. Vérifications au démarrage (dépôt seulement)
    verifs = e6.verifier_demarrage(PHASE)
    cas = verifs["cas_test"]

    # 2. Contrôle du dossier (noms seulement)
    controler_dossier(predictions, cas)

    # 3. Sortie recopiée sur Drive et verrou, seulement maintenant
    os.makedirs(sortie, exist_ok=True)
    print(f"Copie de l'affichage : {e6.ouvrir_sortie(journal, sortie, SCRIPT)}")
    e6.poser_verrou(sortie, "verrou_06a", a.ignorer_verrou)

    # 4. Empreintes (lecture binaire, sans décoder)
    fichiers = {c: relever(os.path.join(predictions, e6.NOM_PREDICTION.format(cas=c))) for c in cas}
    trace = relever(os.path.join(predictions, e6.NOM_TRACE_ETAPE4))
    print(f"\n{'cas':<6}{'fichier':<20}{'SHA-256':<66}{'octets':>10}  modifié le (UTC)")
    for c, f in fichiers.items():
        print(f"{c:<6}{f['nom']:<20}{f['sha256']:<66}{f['taille_octets']:>10}  {f['date_modification_utc']}")
    print(f"{'':<6}{trace['nom']:<20}{trace['sha256']:<66}{trace['taille_octets']:>10}  {trace['date_modification_utc']}")

    contenu = {"objet": "empreintes SHA-256 des 5 prédictions de TotalSegmentator, refaites selon D44 (protocole "
                        "§ 10.2, phase 0 bis ; complément de D42 et D44)",
               "algorithme": "SHA-256 des octets du fichier .nii.gz (lecture binaire, sans décompression)",
               "dossier_predictions": predictions, "fichiers": fichiers, "tracabilite_etape4": trace,
               "verifications_demarrage": verifs, "tracabilite": e6.tracabilite(PHASE, SCRIPT)}
    chemin = os.path.join(sortie, e6.NOM_EMPREINTES_0BIS)

    # 5. Écriture sans écrasement
    if os.path.exists(chemin):
        ancien = e6.lire_json(chemin, PHASE)
        if valeurs_comparees(ancien) != valeurs_comparees(contenu):
            e6.arret(19, f"{chemin} existe déjà avec d'AUTRES valeurs (empreinte, taille, date ou dossier). "
                         "Ne rien effacer : envoyer toute la sortie dans la conversation.\n"
                         f"  existant : {json.dumps(valeurs_comparees(ancien), ensure_ascii=False)}\n"
                         f"  relevé   : {json.dumps(valeurs_comparees(contenu), ensure_ascii=False)}")
        print(f"\n{chemin} existe déjà avec les MÊMES valeurs : gardé tel quel (relevé du "
              f"{ancien['tracabilite']['heure_utc']}).")
        contenu = ancien
    else:
        e6.ecrire_json_atomique(chemin, contenu)
        print(f"\nÉcrit : {chemin}")

    # 6. Bloc à recopier dans le dépôt et ligne pour le journal
    reference = {"objet": "référence des empreintes des prédictions de TotalSegmentator (étape 4, refaites selon "
                          "D44), relevées en phase 0 bis ; vérifiées au démarrage des phases B et C (protocole § 9, point 7)",
                 "algorithme": contenu["algorithme"], "dossier_predictions": contenu["dossier_predictions"],
                 "date_releve_utc": contenu["tracabilite"]["heure_utc"], "commit_releve": contenu["tracabilite"]["commit"],
                 "fichiers": {c: {k: f[k] for k in ("nom", "sha256", "taille_octets", "date_modification_utc")}
                              for c, f in contenu["fichiers"].items()},
                 "tracabilite_etape4": {k: contenu["tracabilite_etape4"][k]
                                        for k in ("nom", "sha256", "taille_octets", "date_modification_utc")}}
    print(f"\n===== DÉBUT du contenu de {e6.REF_EMPREINTES_TOTALSEG} (à recopier tel quel) =====")
    print(json.dumps(reference, ensure_ascii=False, indent=2))
    print(f"===== FIN du contenu de {e6.REF_EMPREINTES_TOTALSEG} =====")
    print("\nLigne proposée pour le journal (complément de D42 et D44) : empreintes SHA-256 des 5 prédictions de "
          "TotalSegmentator refaites selon D44 (phase 0 bis, " + contenu["tracabilite"]["heure_utc"][:10] + ") : "
          + " ; ".join(f"{c} {f['sha256']} (modifié le {f['date_modification_utc']})" for c, f in contenu["fichiers"].items()))
    print(f"\nRésultat : phase 0 bis TERMINÉE ; fin : {e6.heure()}")


if __name__ == "__main__":
    main()
