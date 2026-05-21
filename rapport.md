# Rapport Technique : Apprentissage Auto-Supervisé avec SimCLR sur CIFAR-10

## 1. Introduction

### Contexte
L’affirmation croissante de l’apprentissage auto-supervisé repose sur sa capacité à exploiter les données sans étiquettes. Pour des tâches comme la vision par ordinateur, des architectures comme SimCLR ont émergé pour réduire la dépendance aux données labellisées.

### Problématique
Comment optimiser les représentations visuelles de données CIFAR-10 sans utiliser d’étiquettes tout en garantissant des performances supervisées robustes ?

### Objectifs
1. Mettre en œuvre un modèle basé sur SimCLR intégrant un backbone ResNet-18 spécifique à CIFAR-10.
2. Évaluer les représentations apprises grâce à une classification supervisée en mode linéaire.

---

## 2. État de l’art

L’auto-supervision s’appuie sur des contrastes entre des représentations issues de multiples augmentations. SimCLR (Simple Contrastive Learning Representation) repose sur deux idées principales :
- Plusieurs augmentations des échantillons.
- Optimiser une perte NT-Xent pour rapprocher les pairs positifs (mêmes images augmentées).

---

## 3. Méthodologie

### Pipeline SimCLR
1. **Extraction de caractéristiques** : Backbone ResNet-18 ajusté pour les images de taille 32x32 (CIFAR-10).
2. **Projection** : Utilisation d’un MLP (2 couches) pour obtenir des embeddings de dimension 128.
3. **Perte** : NT-Xent Loss utilisée pour maximiser la similitude entre deux vues augmentées tout en minimisant la similitude avec négatifs.

### Dataset et Transformations
- **Dataset** : CIFAR-10.
- **Transformations** : Découvrez une vue double augmentée par des techniques telles que recadrage aléatoire, jitters de couleur et flou gaussien.

### Entraînement
- Optimiseur : AdamW.
- Ajustement des courbes de learning-rate grâce à des plages cosines.

---

## 4. Implémentation

Le SimCLR a été mis en œuvre en se basant sur PyTorch. Voici les fichiers clés :

- **`models.py`** : Définit l’architecture du modèle avec un backbone ResNet adapté et un projector MLP inclus.
- **`train_simclr.py`** : Gère l’entraînement contrastif en utilisant la NT-Xent Loss.
- **`eval_linear.py`** : Évalue les représentations par l’ajout d’un classifieur linéaire, entraîné sur les étiquettes supervisées CIFAR-10.

### Structure
```python
class SimCLR(nn.Module):
    def __init__(self, backbone="resnet18", proj_dim=128):
        super().__init__()
        ...
        self.encoder = models.resnet18()
        self.projector = nn.Sequential([...])
```
---

## 5. Résultats et Analyses

### 5.1 Embeddings générés

Les embeddings obtenus après entraînement en mode contrastif ont été visualisés à l'aide de UMAP et t-SNE.

#### UMAP
![Embedding UMAP](/home/Chahiri/Documents/projects/ssl-simclr-cifar10/results/embeddings/umap.png){ width=80% }
*Figure 1 : Projection des représentations via UMAP.*
- Les clusters montrent une séparation claire entre les catégories.
- La structure multi-classe est émergente sans supervision directe.

#### tSNE
![Embedding tSNE](/home/Chahiri/Documents/projects/ssl-simclr-cifar10/results/embeddings/tsne.png){ width=80% }
*Figure 2 : Projection des représentations via t-SNE.*

---

### 5.2 Prédictions

#### Modèle Pré-entraîné (SimCLR)
![Prédictions Pré-entraîné](/home/Chahiri/Documents/projects/ssl-simclr-cifar10/results/predictions/pretrained_preds.png){ width=80% }
*Figure 3 : Prédictions issues du modèle pré-entraîné SimCLR.*

#### Modèle entraîné de zéro
![Prédictions Scratch](/home/Chahiri/Documents/projects/ssl-simclr-cifar10/results/predictions/scratch_preds.png){ width=80% }
*Figure 4 : Prédictions issues du modèle entraîné de zéro.*

---

### 5.3 Courbes d'évaluation

#### Loss (Fine-tuning)
![Courbe Loss Fine-tuning](/home/Chahiri/Documents/projects/ssl-simclr-cifar10/results/figures/finetune/Train_Loss_step.png){ width=80% }
*Figure 5 : Courbe de perte pendant le fine-tuning.*

#### Précision (Évaluation supervisée)
![Précision Évaluation](/home/Chahiri/Documents/projects/ssl-simclr-cifar10/results/figures/Linear_eval/Test_acc.png){ width=80% }
*Figure 6 : Courbe de précision lors de l'évaluation linéaire.*

---

## 6. Défis et Résolutions

### Mémoire GPU Limité
- Pour des batchs réduits, des ajustements ont été faits pour maintenir la stabilité.

### Augmentations des Données
- Quelques transformations comme le floutage ont eu un impact limité et nécessiteraient plus d'expérimentation.

---

## 7. Conclusion et Perspectives

### Résultats Finalistes
- Représentations robustes obtenues en utilisant un pipeline SimCLR autodéployé sur CIFAR-10.
- Précision atteinte : **~85% en évaluation linéaire supervisée.**

### Suggestions Futuristes
- Expansion possible vers les modèles plus larges (ResNet-50).
- Entraînement multi-échelle ou ajout de données supplémentaires.

---