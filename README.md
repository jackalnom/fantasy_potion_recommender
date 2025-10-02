# Fantasy Potion Recommender

A machine learning demonstration project that generates synthetic fantasy RPG data and builds binary classifiers and recommender systems to predict adventurer enjoyment of potions.

## Overview

This package contains three main components:

1. **Data Generation** (`create_fantasy_data.py`) - Simulates adventurers, potions, and their interactions
2. **Binary Classifier** (`binary_classifier.py`) - Predicts whether an adventurer will like a potion (enjoyment > 0.5)
3. **Recommender System** (`recommender.py`) - Generates top-K potion recommendations for adventurers

---

## 1. Data Generation

The data generator creates a synthetic dataset of fantasy adventurers trying different colored potions.

### Adventurers

Adventurers are generated with three classes: **Fighter**, **Wizard**, and **Paladin**, with levels ranging from 1-20.

Each adventurer has damage stats calculated using **dice rolls**:

#### Fighter Damage
- **Physical**: `1d8+3` (level 1-5), `2d8+6` (6-10), `3d8+9` (11-16), `4d8+12` (17-20)
- **Magic**: `0` (1-5), `1d4` (6-10), `2d4` (11-16), `3d4` (17-20)
- Fighters deal primarily physical damage with some magic at higher levels

#### Wizard Damage
- **Physical**: `1d4` (all levels)
- **Magic**: `1d12` (1-4), `2d12` (5-10), `3d12` (11-16), `4d12` (17-20)
- Wizards deal primarily magic damage with minimal physical

#### Paladin Damage
- **Physical**: `1d8+3` (1-4), `2d8+6` (5-16), `3d8+9` (17-20)
- **Magic**: `1d6` (1-4), `2d6` (5-10), `3d6` (11-16), `4d6` (17-20)
- Paladins are balanced between physical and magic damage

### Potions

100 potions are generated with RGB color values that sum to 100. Each potion's color determines which classes will enjoy it.

### Preference Model

Adventurer enjoyment is based on potion color preferences:

- **Fighters** prefer **red** potions:
  - Ideal potion: 100 red, 0 green, 0 blue
- **Wizards** prefer **blue** potions:
  - Ideal potion: 0 red, 0 green, 100 blue
- **Paladins** prefer **balanced red+blue** potions:
  - Ideal potion: 50 red, 0 green, 50 blue

### Output Files

- `interactions.csv` - Adventurer-potion interactions with enjoyment scores

---

## 2. Binary Classifier

`binary_classifier.py` trains two models to predict whether an adventurer will **like** a potion (enjoyment > 0.5).

### Features Used
- `adv_id` - Adventurer ID
- `potion_id` - Potion ID
- `avg_phys` - Adventurer's average physical damage
- `avg_magic` - Adventurer's average magic damage
- `red`, `green`, `blue` - Potion RGB color values

### Models

1. **K-Nearest Neighbors (KNN)** - Uses 5 neighbors with standardized features
2. **Random Forest**

### Evaluation

The script splits the data 80/20 (train/test) and outputs:

- **Console metrics**: Accuracy, F1 score, ROC AUC, confusion matrix, and classification report
- **ROC curve** (`roc_curve.html`) - Interactive Plotly plot comparing KNN vs RandomForest true/false positive rates
- **PR curve** (`pr_curve.html`) - Interactive Plotly precision-recall curve comparison

### Usage

```bash
uv run binary_classifier.py
```

---

## 3. Recommender System

`recommender.py` builds collaborative filtering recommenders that predict enjoyment scores and generate top-K recommendations.

### Train/Test Split Strategy

Instead of random splitting, the recommender uses a **multi-positive holdout** approach:

1. For each adventurer with at least 1 liked potion, hold out up to 3 liked potions for testing
2. All other interactions go to training
3. Only adventurers with held-out positives are evaluated

This ensures:
- Every evaluated user has ground truth (held-out liked potions)
- Models must predict on **unseen potions** (not in training set)
- Realistic cold-start scenario

### Models

1. **KNN Regressor** - Predicts enjoyment scores using K=5 nearest neighbors with standardized features
2. **Random Forest Regressor** - Predicts enjoyment using 300 trees
3. **Random Baseline** - Random shuffling of candidates (control)

### Recommendation Process

For each adventurer:
1. Build candidate set: all potions **not seen during training**
2. Create feature vectors pairing adventurer stats with each candidate potion's RGB
3. Predict enjoyment score for each candidate
4. Rank candidates by predicted score (highest first)
5. Return top-K recommendations

### Evaluation Metrics

- **Precision@K**: Of the top K recommendations, what fraction are in the held-out liked set?
- **Recall@K**: Of all held-out liked potions, what fraction appear in top K?
- **MAP@K** (Mean Average Precision): Rewards ranking liked items higher in the list

Metrics are averaged across all evaluated users.

### Output

The script prints:
- Overall metrics for each model (KNN, RandomForest, Random baseline)
- Sample recommendations for first 5 evaluated adventurers showing:
  - Adventurer class/level
  - Held-out positive potions (ground truth)
  - Top-K recommendations from each model with RGB colors
  - Hit count (how many recommendations were correct)
  - Candidate set size

### Usage

```bash
uv run recommender.py
```

### Configuration

Edit constants at the top of `recommender.py`:
- `KNN_K` - Number of neighbors for KNN (default: 5)
- `TOP_K` - Number of recommendations to generate (default: 3)
- `HOLDOUT_POS_PER_USER` - Maximum positives to hold out per user (default: 3)
- `SAMPLE_N` - Number of sample users to print detailed recommendations (default: 5)

---

## Installation

```bash
# Install dependencies using uv
uv sync
```

## Quick Start

```bash
# 1. Generate data
uv run create_fantasy_data.py

# 2. Train binary classifier
uv run binary_classifier.py

# 3. Build recommender system
uv run recommender.py
```

## Dependencies

- pandas
- numpy
- scikit-learn
- plotly (for binary classifier visualizations)
