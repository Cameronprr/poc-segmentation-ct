"""
Variante d'entraînement nnU-Net v2 de l'étape 5 (décision D25) : 250 époques, point de reprise toutes les 10.

Ce que fait ce fichier :
  définit la classe nnUNetTrainer_250epochs_reprise10, sous-classe de nnUNetTrainer (le trainer par
  défaut de nnU-Net). Seules trois choses changent ; tout le reste est le réglage par défaut :
    - num_epochs = 250 au lieu de 1000 (D25 : compromis entre durée sur Colab et qualité) ;
    - save_every = 10 au lieu de 50 : checkpoint_latest.pth est écrit toutes les 10 époques, pour ne
      perdre au plus que 9 époques (environ 17 min sur L4) si la session Colab est coupée ;
    - save_checkpoint écrit d'abord dans un fichier temporaire, puis le renomme. nnU-Net 2.8.1 écrit
      directement par-dessus l'ancien fichier (torch.save) : une coupure pendant l'écriture dans Drive
      laisserait un checkpoint_latest.pth tronqué, le seul point de reprise. Avec le renommage, l'ancien
      fichier reste intact tant que le nouveau n'est pas complet.

Comment nnU-Net trouve cette classe :
  nnUNetv2_train cherche le nom donné après -tr d'abord dans son propre paquet, puis dans les dossiers
  listés dans la variable d'environnement nnUNet_extTrainer (nnunetv2 2.8.1, utilities/find_objects.py).
  Le script 05c fixe nnUNet_extTrainer = ce dossier (scripts/trainers_nnunet) : rien n'est écrit dans
  le paquet installé. Ce dossier ne doit contenir QUE des fichiers de trainers : nnU-Net importe chacun.
  ATTENTION : la même variable sera nécessaire pour prédire avec ce modèle (étape 6), car nnU-Net
  retrouve la classe par son nom, noté dans le checkpoint.

Le nom du trainer fait partie du dossier de résultats :
  nnUNet_results/Dataset101_MMWHS/nnUNetTrainer_250epochs_reprise10__nnUNetPlans__3d_fullres/fold_X/
  Renommer la classe reviendrait à repartir de zéro dans un autre dossier.

Vocabulaire :
  - sous-classe : une classe qui reprend tout le code d'une autre (« hérite ») et n'en change que quelques
    morceaux ;
  - renommage atomique (os.replace) : sur un même disque, le fichier change de nom en une seule
    opération : on voit soit l'ancien fichier, soit le nouveau complet, jamais un fichier à moitié écrit.
    Limite : Google Drive recopie ensuite le fichier vers ses serveurs, en différé ; si la machine Colab
    disparaît avant la fin de cette recopie, Drive garde la version précédente (qui reste lisible).
"""
import os

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_250epochs_reprise10(nnUNetTrainer):
    # Même signature que les variantes officielles (nnUNetTrainer_Xepochs.py) : nnU-Net s'en sert pour
    # recréer le trainer à partir du checkpoint.
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.save_every = 10

    def save_checkpoint(self, filename: str) -> None:
        temporaire = filename + ".ecriture_en_cours"
        if os.path.isfile(temporaire):
            os.remove(temporaire)                    # reste d'une écriture interrompue : jamais renommé
        super().save_checkpoint(temporaire)          # écrit seulement si rang 0 et points de reprise actifs
        if os.path.isfile(temporaire):
            os.replace(temporaire, filename)         # l'ancien fichier n'est remplacé qu'une fois le nouveau complet
