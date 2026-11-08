"""
Étape 3b — Regrouper les cas par patient probable et proposer un découpage PAR PATIENT (décision D20).

Pourquoi : l'examen visuel (section 8 du notebook) a montré que deux paires de cas (1005/1012 et
1020/1001) sont très probablement le même patient scanné à deux phases du battement cardiaque.
Un même patient présent à la fois en test et en entraînement fausserait le verdict final.

Ce que fait le script :
  1. pour les 20 cas du découpage actuel, lit UNIQUEMENT l'en-tête des scanners avec nibabel
     (forme, espacement, affine, orientation, sform/qform) : aucun voxel n'est chargé, aucun masque lu ;
  2. regroupe en « patients probables » les cas qui ont la même forme, le même espacement et la même
     affine (à une tolérance près), et affiche tous les groupes, y compris ceux d'un seul cas.
     Un cas dont l'origine n'est pas fiable (sform_code = qform_code = 0) n'est JAMAIS regroupé ;
  3. affiche pour information les « paires proches » (même plan de coupe, même position x, y, mais
     nombre de coupes ou hauteur différents) : possibles patients communs NON regroupés, à examiner ;
  4. propose un nouveau découpage : le lot de test ne contient QUE des patients à un seul examen
     (groupe d'un seul cas, origine fiable), tirés au sort avec une graine fixe : 5 cas = 5 patients
     indépendants. Le tirage est STRATIFIÉ par épaisseur de coupe : 1 patient à 0,45 mm, 4 à 0,625 mm.
     Les groupes de plusieurs cas (même patient) vont tous en entraînement, entiers ;
  5. affiche la proposition, les cas qui changent de lot, et l'écrit en JSON dans --sortie (Drive).
  Le découpage figé (splits/mmwhs_split.json) est seulement LU, jamais modifié : c'est l'utilisateur
  qui recopiera la proposition dans le dépôt après validation.

  Attention : la géométrie seule ne prouve rien. Elle s'ajoute à l'examen visuel (section 8) ;
  c'est l'ensemble des deux qui justifie de traiter deux cas comme un même patient.

Entrées :
  --donnees            dossier où MM-WHS a été téléchargé (dans Google Drive), cherché récursivement
  --sortie             dossier de Google Drive où écrire la proposition (jamais dans le dépôt)
  --split              découpage actuel (défaut : ../splits/mmwhs_split.json à côté du script)
  --graine             graine du tirage des 5 patients de test (défaut : 42, comme le découpage d'origine)
  --tolerance-origine  écart maximal entre deux affines (donc entre deux origines), en mm (défaut : 0.01)
Sortie (uniquement dans --sortie) :
  mmwhs_split_par_patient_PROPOSITION.json : même format que splits/mmwhs_split.json, plus les clés
  "groupes_patients" (groupes d'au moins 2 cas) et "methode" (comment la proposition a été faite).
Codes de sortie : 0 = proposition écrite (lire les avertissements en fin de sortie) ;
  1 = arrêt (fichier manquant ou ambigu, découpage incohérent, --sortie dans le dépôt, épaisseur de
  coupe imprévue ou mal lisible (3e axe ni S ni I), ou pas assez de patients à un seul examen (origine fiable) dans une strate pour
  remplir son quota) : rien n'a été écrit.

Vocabulaire :
  - en-tête (header) : le début du fichier NIfTI, qui décrit le volume (taille, espacement, position)
    sans contenir les voxels. Le lire est presque instantané ;
  - espacement : taille d'un voxel en mm le long de chaque axe ;
  - affine : matrice 4x4 qui relie chaque voxel à sa position dans le corps (en mm). Sa dernière
    colonne est l'origine : la position du premier voxel du volume ;
  - sform_code / qform_code : deux codes de l'en-tête qui disent si l'affine a été réellement
    enregistrée par le scanner. S'ils valent tous les deux 0, l'en-tête ne contient PAS de position :
    nibabel fabrique alors une affine par défaut, centrée sur le volume, et deux volumes de même taille
    ont forcément « la même origine ». Cette origine n'est pas fiable et ne prouve rien ;
  - orientation : sens des axes, en 3 lettres (ex. « RAS » = vers la droite, l'avant, le haut) ;
  - même forme, même espacement et même affine au centième de mm suggèrent fortement le même examen
    reconstruit à une autre phase : deux examens indépendants ont presque toujours des origines
    différentes. Mais c'est un indice, pas une preuve ;
  - phase cardiaque : instant du battement (cœur contracté ou relâché). Un scanner cardiaque peut
    produire plusieurs volumes du même patient, un par phase ;
  - fuite de patient : le même patient dans deux lots. Le modèle aurait « vu » ce patient pendant
    l'entraînement : son score sur le test serait trop optimiste ;
  - graine : nombre qui fixe le tirage au sort. Même graine = même mélange à chaque exécution ;
  - patients indépendants : dans le test, chaque patient ne doit compter qu'une fois. Deux examens du
    même patient en test feraient compter ce patient double dans le verdict (et ne seraient pas deux
    preuves indépendantes) : c'est pourquoi le test ne prend que des patients à un seul examen ;
  - épaisseur de coupe : ici, l'ESPACEMENT ENTRE COUPES (3e valeur de l'espacement, pixdim[3]), c.-à-d.
    la distance entre deux coupes successives. Ce n'est pas forcément l'épaisseur de reconstruction
    du scanner (SliceThickness en DICOM), qui diffère si les coupes se chevauchent. Il
    dépend du scanner et du protocole ;
  - tirage stratifié : on tire au sort séparément dans chaque sous-groupe (« strate », ici chaque
    épaisseur de coupe), avec un nombre de places fixé pour chacune. Le test contient ainsi les deux
    types d'examens, au lieu de dépendre du hasard.
"""
import argparse, glob, json, os, random, sys
import nibabel as nib
import numpy as np

TOLERANCE_ESPACEMENT = 1e-3   # mm : deux espacements plus proches que cela sont considérés égaux
TOLERANCE_PAIRE_PROCHE = 1.0  # mm : écart maximal en x et y pour signaler une « paire proche »
# Places en test par épaisseur de coupe (mm) : 1 patient à 0,45 mm et 4 à 0,625 mm (décision D20).
# Le jeu contient deux familles d'examens : 0,45 mm (sform/qform 1/1, probablement un autre scanner)
# et 0,625 mm. Sans stratification, le tirage avait mis en test 5 cas à 0,625 mm et aucun à 0,45 mm.
QUOTAS_TEST = {0.45: 1, 0.625: 4}
TAILLE_TEST = sum(QUOTAS_TEST.values())   # 5 patients (= 5 cas) en test, comme le découpage d'origine
NOM_FICHIER = "mmwhs_split_par_patient_PROPOSITION.json"

# --- Arguments --------------------------------------------------------------------------------
depot = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
p = argparse.ArgumentParser()
p.add_argument("--donnees", required=True)
p.add_argument("--sortie", required=True)
p.add_argument("--split", default=os.path.join(depot, "splits", "mmwhs_split.json"))
p.add_argument("--graine", type=int, default=42)
p.add_argument("--tolerance-origine", type=float, default=0.01)
a = p.parse_args()

sortie = os.path.abspath(a.sortie)
if os.path.commonpath([sortie, depot]) == depot:
    sys.exit("ARRÊT : --sortie est dans le dépôt. Les données doivent rester dans Google Drive.")

with open(a.split, encoding="utf-8") as f:          # lecture seule du découpage actuel
    split = json.load(f)
ancien_ts, ancien_tr = split["test_sous_cle"], split["entrainement_validation_croisee"]
tous = sorted(ancien_ts + ancien_tr)
if len(tous) != 20 or len(set(tous)) != 20:
    sys.exit(f"ARRÊT : découpage actuel incohérent ({len(tous)} entrées, {len(set(tous))} cas distincts, "
             "attendu 20).")

# --- 1. Lecture des en-têtes ------------------------------------------------------------------
entetes, erreurs = {}, []
for c in tous:
    trouves = sorted(glob.glob(os.path.join(a.donnees, "**", f"{c}_image.nii.gz"), recursive=True))
    if len(trouves) != 1:   # 0 = manquant, plus de 1 = ambigu (ex. archive décompressée deux fois)
        erreurs.append(f"{c} : {len(trouves)} fichier(s) image trouvé(s) au lieu de 1 {trouves}")
        continue
    img = nib.load(trouves[0])   # nib.load ne lit que l'en-tête ; les voxels ne sont pas chargés
    sform, qform = int(img.header["sform_code"]), int(img.header["qform_code"])
    entetes[c] = {"forme": tuple(int(n) for n in img.shape),
                  "espacement": np.array(img.header.get_zooms()[:3], dtype=float),
                  "affine": np.array(img.affine, dtype=float),
                  "origine": np.array(img.affine[:3, 3], dtype=float),
                  "orientation": "".join(nib.aff2axcodes(img.affine)),
                  "sform": sform, "qform": qform,
                  "fiable": not (sform == 0 and qform == 0)}   # 0 et 0 = affine fabriquée par nibabel
if erreurs:
    print("ARRÊT : fichiers manquants ou ambigus, rien n'a été écrit :")
    for e in erreurs:
        print("  -", e)
    sys.exit(1)
non_fiables = [c for c in tous if not entetes[c]["fiable"]]

# --- 2. Regroupement en patients probables ----------------------------------------------------
def meme_patient(x, y):
    """Vrai si deux cas à origine fiable ont la même forme, le même espacement et la même affine
    (donc la même origine et la même orientation), aux tolérances près."""
    return (x["fiable"] and y["fiable"]
            and x["forme"] == y["forme"]
            and np.allclose(x["espacement"], y["espacement"], rtol=0, atol=TOLERANCE_ESPACEMENT)
            and np.allclose(x["affine"], y["affine"], rtol=0, atol=a.tolerance_origine))

# Regroupement « de proche en proche » : si A ressemble à B et B à C, alors A, B et C forment un
# seul groupe. Chaque cas part de son propre groupe ; on fusionne les groupes des paires semblables.
groupe_de = {c: {c} for c in tous}
for i, x in enumerate(tous):
    for y in tous[i + 1:]:
        if meme_patient(entetes[x], entetes[y]) and groupe_de[x] is not groupe_de[y]:
            fusion = groupe_de[x] | groupe_de[y]
            for c in fusion:
                groupe_de[c] = fusion
groupes = sorted({tuple(sorted(g)) for g in groupe_de.values()})   # ordre fixe, trié par 1er cas

def lot_actuel(c):
    return "TEST" if c in ancien_ts else "entr."

print(f"Groupes de patients probables (forme identique, espacement à {TOLERANCE_ESPACEMENT} mm près, "
      f"affine à {a.tolerance_origine} mm près) : {len(groupes)} groupes pour {len(tous)} cas\n")
print(f"{'groupe':<8}{'cas (lot actuel)':<24}{'forme':<17}{'espacement (mm)':<24}{'orient.':<8}"
      f"{'sform/qform':<13}origine (mm)")
for n, g in enumerate(groupes, 1):
    for c in g:   # une ligne par cas ; le numéro de groupe est répété pour les cas regroupés
        e = entetes[c]
        origine = str(tuple(round(float(v), 2) for v in e["origine"]))
        if not e["fiable"]:
            origine += "  origine NON FIABLE"
        print(f"{n:<8}{c + ' (' + lot_actuel(c) + ')':<24}{str(e['forme']):<17}"
              f"{str(tuple(round(float(v), 3) for v in e['espacement'])):<24}{e['orientation']:<8}"
              f"{str(e['sform']) + '/' + str(e['qform']):<13}{origine}")
for g in groupes:
    if len(g) > 2:
        print(f"ATTENTION : groupe de {len(g)} cas {list(g)} : plus de 2 phases pour un patient, "
              "ou regroupement en chaîne à vérifier à l'œil.")
for g in groupes:
    if len(g) > 1 and len({lot_actuel(c) for c in g}) > 1:
        print(f"FUITE dans le découpage actuel : le groupe {list(g)} est à cheval sur test et entraînement.")

# --- 3. Paires proches, NON regroupées (pour information) ---------------------------------------
# Un même patient peut avoir été reconstruit avec un nombre de coupes ou une hauteur différents :
# le critère strict ci-dessus ne le verrait pas. On liste donc les paires qui partagent le plan de
# coupe (même nombre de voxels et même espacement en x, y) et la position x, y (à 1 mm près).
# Les cas à origine non fiable sont exclus : leur position x, y n'a pas de sens.
paires_proches = []
for i, x in enumerate(tous):
    for y in tous[i + 1:]:
        ex, ey = entetes[x], entetes[y]
        if (ex["fiable"] and ey["fiable"] and groupe_de[x] is not groupe_de[y]
                and ex["forme"][:2] == ey["forme"][:2]
                and np.allclose(ex["espacement"][:2], ey["espacement"][:2], rtol=0, atol=TOLERANCE_ESPACEMENT)
                and np.allclose(ex["origine"][:2], ey["origine"][:2], rtol=0, atol=TOLERANCE_PAIRE_PROCHE)):
            paires_proches.append((x, y))
print(f"\nPaires proches NON regroupées (même plan de coupe, x et y à {TOLERANCE_PAIRE_PROCHE} mm près, "
      "mais nombre de coupes ou hauteur z différents) :")
if not paires_proches:
    print("  aucune")
for x, y in paires_proches:
    print(f"  {x} {entetes[x]['forme']} z={entetes[x]['origine'][2]:.2f} mm  ~  "
          f"{y} {entetes[y]['forme']} z={entetes[y]['origine'][2]:.2f} mm")

# --- 4. Nouveau découpage par patient ---------------------------------------------------------
# Règle D20 : le test ne contient que des patients à UN SEUL examen, pour que chaque patient de test
# compte une seule fois dans le verdict. Candidats = groupes d'un seul cas dont l'origine est fiable
# (un cas à origine non fiable pourrait être un patient commun non détecté : il reste en entraînement).
# Les groupes de plusieurs cas (même patient, plusieurs phases) vont tous, entiers, en entraînement.
candidats = sorted(g[0] for g in groupes if len(g) == 1 and entetes[g[0]]["fiable"])

# Tirage STRATIFIÉ par épaisseur de coupe (espacement le long du 3e axe) : chaque strate reçoit son
# quota de places en test (QUOTAS_TEST), pour que le test représente les deux types d'examens.
def strate_de(c):
    """Épaisseur de coupe du cas c si elle correspond à une strate connue, sinon None."""
    ep = float(entetes[c]["espacement"][2])
    return next((s for s in QUOTAS_TEST if abs(ep - s) <= TOLERANCE_ESPACEMENT), None)

# zooms[2] n'est l'espacement entre coupes que si le 3e axe du volume va des pieds à la tête (code
# d'orientation S ou I). Sinon ce serait un espacement dans le plan de coupe : on s'arrête.
axe_inattendu = [(c, entetes[c]["orientation"]) for c in candidats if entetes[c]["orientation"][2] not in "SI"]
if axe_inattendu:
    print(f"\nARRÊT : 3e axe qui n'est pas l'axe des coupes (orientation) pour {axe_inattendu} : "
          "l'épaisseur de coupe serait mal lue. Rien n'a été écrit.")
    sys.exit(1)
# Même tolérance que pour comparer deux espacements (TOLERANCE_ESPACEMENT).
hors_strate = [c for c in candidats if strate_de(c) is None]
if hors_strate:   # une épaisseur imprévue : les quotas ne représenteraient plus le jeu, on s'arrête
    print(f"\nARRÊT : épaisseur de coupe hors des strates {sorted(QUOTAS_TEST)} mm pour "
          f"{[(c, round(float(entetes[c]['espacement'][2]), 3)) for c in hors_strate]}. Rien n'a été écrit.")
    sys.exit(1)
tirage = random.Random(a.graine)            # un seul générateur, strates parcourues dans un ordre fixe
test = []
print("\nTirage stratifié par épaisseur de coupe (patients à un seul examen, origine fiable) :")
for s in sorted(QUOTAS_TEST):
    strate = sorted(c for c in candidats if strate_de(c) == s)   # trié : le tirage ne dépend que de la graine
    print(f"  {s} mm : {len(strate)} candidat(s), {QUOTAS_TEST[s]} place(s) en test")
    if len(strate) < QUOTAS_TEST[s]:
        print(f"\nARRÊT : seulement {len(strate)} patient(s) à un seul examen à {s} mm pour "
              f"{QUOTAS_TEST[s]} place(s) en test. Rien n'a été écrit : les groupes sont à revoir ensemble.")
        sys.exit(1)
    tirage.shuffle(strate)                   # mélange reproductible
    test += strate[:QUOTAS_TEST[s]]          # les premiers après mélange
nouveau_ts = sorted(test)
nouveau_tr = sorted(c for c in tous if c not in test)

# Vérifications : aucun cas perdu ou en double, 5 cas en test, quotas respectés, uniquement des patients
# à un seul examen, et aucun groupe coupé (tous les groupes de plusieurs cas sont entièrement en entraînement).
assert sorted(nouveau_ts + nouveau_tr) == tous and len(nouveau_ts) == TAILLE_TEST
assert all(sum(strate_de(c) == s for c in nouveau_ts) == q for s, q in QUOTAS_TEST.items())
assert all(len(groupe_de[c]) == 1 for c in nouveau_ts)
assert all(c in nouveau_tr for g in groupes if len(g) > 1 for c in g)

# --- 5. Affichage de la proposition et des changements ------------------------------------------
print(f"\nProposition (graine {a.graine}) :")
print(f"  test sous clé ({len(nouveau_ts)} cas) : "
      f"{[f'{c} ({strate_de(c)} mm)' for c in nouveau_ts]}")
print(f"  entraînement / validation croisée ({len(nouveau_tr)} cas) : {nouveau_tr}")

sortent = sorted(set(ancien_ts) - set(nouveau_ts))
entrent = sorted(set(nouveau_ts) - set(ancien_ts))
print("\nChangements par rapport au découpage actuel :")
if not sortent and not entrent:
    print("  aucun : le lot de test est inchangé.")
for c in sortent:
    print(f"  {c} : test → entraînement")
for c in entrent:
    print(f"  {c} : entraînement → test")

a_cheval = [(x, y) for x, y in paires_proches if (x in nouveau_ts) != (y in nouveau_ts)]
for x, y in a_cheval:
    print(f"ATTENTION : la paire proche {x} / {y} est à cheval sur test et entraînement dans la "
          "PROPOSITION : à examiner à l'œil avant de valider.")

# --- 6. Écriture de la proposition -------------------------------------------------------------
methode = (f"Décision D20 : découpage par patient. Deux cas appartiennent au même patient probable s'ils ont "
           f"la même forme, le même espacement (tolérance {TOLERANCE_ESPACEMENT} mm) et la même affine "
           f"(tolérance {a.tolerance_origine} mm), d'après l'en-tête NIfTI des scanners, et si leur origine "
           f"est fiable (sform_code ou qform_code non nul) ; regroupement de proche en proche, à confirmer par "
           f"l'examen visuel. Test = {TAILLE_TEST} patients à un seul examen (origine fiable), tirage stratifié "
           f"par épaisseur de coupe ({', '.join(f'{v} à {s} mm' for s, v in sorted(QUOTAS_TEST.items()))}) "
           f"avec random.Random({a.graine}) ; tous les groupes de plusieurs cas sont en entraînement "
           f"(script scripts/03b_regrouper_patients.py).")
proposition = {
    "jeu": split["jeu"],
    "graine": a.graine,
    "test_sous_cle": nouveau_ts,
    "entrainement_validation_croisee": nouveau_tr,
    "regle": "Le lot de test n'est utilisé qu'une seule fois, pour le verdict final. Il ne contient que "
             "des patients à un seul examen ; les cas d'un même patient restent ensemble, en entraînement "
             "(et dans la même part de validation croisée). Ne jamais modifier ce fichier.",
    "groupes_patients": [list(g) for g in groupes if len(g) >= 2],
    "methode": methode,
}
os.makedirs(sortie, exist_ok=True)
chemin = os.path.join(sortie, NOM_FICHIER)
with open(chemin, "w", encoding="utf-8") as f:   # remplace une éventuelle proposition précédente
    json.dump(proposition, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"\nProposition écrite : {chemin}")
print("Contenu du fichier :")
print(json.dumps(proposition, indent=2, ensure_ascii=False))
print("\nRien n'a été modifié dans le dépôt. Après validation, cette proposition remplacera "
      "splits/mmwhs_split.json (à recopier à la main).")

# Avertissement placé tout à la fin pour qu'il ne passe pas inaperçu.
if non_fiables:
    print("\n" + "!" * 100)
    print(f"ATTENTION : ORIGINE NON FIABLE pour {len(non_fiables)} cas (sform_code = qform_code = 0) :")
    print("  " + ", ".join(non_fiables))
    print("  Leur en-tête ne contient pas de position : ces cas n'ont été regroupés avec AUCUN autre, et ils")
    print("  ont été exclus du lot de test par prudence (ils restent en entraînement).")
    print("  Le regroupement de ces cas n'est pas démontré par la géométrie : à décider avec l'examen visuel.")
    print("!" * 100)
