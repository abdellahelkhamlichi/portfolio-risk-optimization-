# Portfolio Risk & Optimization

Projet de portfolio professionnel en gestion d'actifs : mesurer le risque réel d'un portefeuille, comparer trois méthodes de VaR, calculer le CVaR, explorer une frontière efficiente et tester des scénarios de crise.

## Résultats

- Données de marché réelles téléchargées directement depuis `yfinance`.
- Notebook unique couvrant collecte, EDA, VaR historique/paramétrique/Monte Carlo, CVaR, Markowitz, stress testing et génération du rapport PDF.
- Dashboard Streamlit permettant de composer un portefeuille et de consulter instantanément VaR, CVaR, allocation optimale et frontière efficiente.
- Artefacts légers : CSV de rendements et JSON de métriques/optimisation. Aucun pickle volumineux.

## Structure

```text
portfolio-risk-optimization/
├── data/raw/                         # vide : aucun fichier de marché manuel
├── data/processed/                   # artefacts générés par le notebook
├── notebook/portfolio_risk_project.ipynb
├── dashboard/app.py
├── reports/figures/                  # graphiques générés
├── reports/final_report.pdf          # rapport généré
├── requirements.txt
└── README.md
```

## Installation

Python 3.10+ est recommandé. Python 3.14 peut nécessiter des versions récentes des paquets scientifiques.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Exécution reproductible

1. Ouvrir `notebook/portfolio_risk_project.ipynb` dans VS Code ou Jupyter.
2. Exécuter les cellules dans l'ordre avec une connexion Internet active.
3. Vérifier la présence de `data/processed/portfolio_returns.csv`, des deux JSON et de `reports/final_report.pdf`.
4. Lancer le dashboard :

```bash
streamlit run dashboard/app.py
```

Le dashboard construit ses chemins depuis `Path(__file__).resolve().parents[1]`, donc il ne dépend pas du répertoire courant.

## Méthodologie et hypothèses

Le portefeuille de référence est équipondéré entre `SPY`, `QQQ`, `TLT`, `GLD` et `IEF`. L'historique couvre les dix dernières années disponibles. Les rendements sont quotidiens, annualisés sur 252 séances, et les pertes sont définies comme `-rendement`.

La VaR historique est un percentile empirique ; la VaR paramétrique suppose une loi normale ; Monte Carlo simule 20 000 rendements multivariés avec la moyenne et la covariance historiques. Le taux sans risque de 2 % sert uniquement au ratio de Sharpe. L'optimisation est long-only, sans levier, avec somme des poids égale à 1.

Ces modèles sont des outils pédagogiques d'aide à la décision : ils ne constituent pas un conseil en investissement. Les limites principales sont la dépendance à l'historique, la non-normalité des rendements, l'instabilité des paramètres d'optimisation et l'absence de coûts de transaction.

## Contrôles de cohérence

Le notebook vérifie un historique minimal, la positivité de la VaR selon la convention de perte, la somme des poids optimisés et l'existence de résultats d'optimisation. Pour un usage de production, ajouter une validation walk-forward, des coûts, des limites réglementaires et un suivi de dérive des modèles.
