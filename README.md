# Commande-optimale-ou-apprentissage-par-renforcement-LQR-vs-PPO-sur-MuJoCo

Stabilisation d'un pendule inversé sur chariot simulé dans **MuJoCo**, par deux approches :

1. **Automatique classique** : modèle linéarisé directement à partir du simulateur, puis commande **LQR** (équation de Riccati discrète).
2. **Apprentissage par renforcement** : politique **PPO** (Stable-Baselines3, PyTorch) apprise sans modèle.

Les deux contrôleurs sont ensuite comparés sur la performance nominale et sur leur **robustesse** : grand angle initial, erreur de modèle (masse ×4) et poussée extérieure.

> Objectif : montrer le lien entre commande à base de modèle et apprentissage robotique, une question centrale du transfert simulation-réalité.

---

## Résultats

| Test | LQR | PPO |
|---|---|---|
| Nominal : réussite sur 40 s | 100 % | à compléter |
| Angle initial ±0,19 rad | 100 % | à compléter |
| Masse du pendule ×4 | 100 % | à compléter |
| Poussée max tolérée (0,2 s) | 20 N | à compléter |
| Données nécessaires | modèle (linéarisation) | 100 000 pas d'interaction |

*Une ligne de commentaire par résultat : ce que chaque approche fait bien, ce qu'elle rate, et pourquoi.*

Ajoute ici une vidéo ou un GIF des deux contrôleurs et la courbe d'apprentissage PPO (TensorBoard).

---

## Méthode

### Le système
Environnement `InvertedPendulum-v5` de Gymnasium (MuJoCo, intégrateur RK4, pas de 0,02 s, commande toutes les 0,04 s).
État : position et vitesse du chariot, angle et vitesse angulaire du pendule. Commande : force sur le chariot, saturée à ±3. Échec si |θ| > 0,2 rad.

### LQR
- **Linéarisation numérique** autour de l'équilibre haut, par différences finies centrées sur le vrai simulateur. La commande étant maintenue sur 2 pas de simulation, A et B sont calculées directement au pas de commande.
- **Gain optimal** : résolution de l'équation de Riccati discrète (`scipy.linalg.solve_discrete_are`) avec Q = diag(1, 10, 0,1, 0,1) et R = 0,1.
- Commande u = −Kx, saturée aux limites de l'actionneur.

### PPO
- Politique MLP entraînée avec Stable-Baselines3 (4 environnements en parallèle, 100 000 pas).
- Récompense standard de l'environnement : +1 par pas où le pendule reste debout.

### Évaluation
`src/evaluate.py` exécute les mêmes tests pour les deux contrôleurs, avec les mêmes graines aléatoires.

---

## Lancer le projet

```bash
pip install -r requirements.txt

python src/lqr_controller.py          # gain LQR et un épisode de test
python src/train_ppo.py --steps 100000
python src/evaluate.py --episodes 20  # comparaison LQR / PPO
tensorboard --logdir runs             # courbes d'apprentissage
```

---

## Structure

```
.
├── src/
│   ├── lqr_controller.py   # linéarisation + LQR
│   ├── train_ppo.py        # entraînement PPO
│   └── evaluate.py         # tests nominaux et de robustesse
├── models/                 # politiques entraînées
├── requirements.txt
└── README.md
```

---

## Ce que j'en retiens

*À rédiger après les résultats, par exemple :*
- quand le modèle est connu, le LQR est immédiat, optimal localement et explicable ;
- PPO n'a besoin d'aucun modèle, mais demande beaucoup d'interactions et sa robustesse dépend de l'entraînement ;
- la randomisation de domaine (masse, frottements variables pendant l'entraînement) est une piste pour améliorer le transfert vers le réel.

## Pistes d'amélioration
- Randomisation de domaine pendant l'entraînement PPO, puis nouvelle comparaison de robustesse.
- Swing-up (remonter le pendule depuis le bas), où le LQR seul ne suffit plus.
- Politique hybride : PPO pour le swing-up, LQR pour la stabilisation.

---

**Bangaly Conte** — Ingénieur en automatique et systèmes embarqués (Master 2 SAM, Université d'Évry Paris-Saclay)
