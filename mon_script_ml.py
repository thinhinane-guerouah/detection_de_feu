# --- BIBLIOTHÈQUES POUR LA MANIPULATION ET VISUALISATION DES DONNÉES ---
import pandas as pd              # Pour manipuler les données sous forme de tableaux (DataFrames)
import matplotlib.pyplot as plt  # Pour créer des graphiques de base (courbes, barres, etc.)
import seaborn as sns           # Pour des visualisations statistiques plus avancées et esthétiques

# --- BIBLIOTHÈQUES POUR LE MACHINE LEARNING (SKLEARN) ---

# 1. Modèles de Régression (Prédire une valeur numérique continue)
from sklearn.linear_model import Ridge 

# 2. Modèles de Classification (Prédire une catégorie : Feu / Pas de feu)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 3. Outils de Prétraitement et de Découpage
from sklearn.preprocessing import StandardScaler, PolynomialFeatures  # PolynomialFeatures pour la régression non-linéaire
from sklearn.model_selection import train_test_split, GridSearchCV # Pour séparer les données en entraînement et test + CV
from sklearn.pipeline import Pipeline

# 4. Métriques d'Évaluation (Mesurer la précision des modèles)
from sklearn.metrics import r2_score, accuracy_score, classification_report
# 5.Visualisation de l'Arbre 

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# On charge le fichier CSV qui se trouve dans le même dossier
# pd.read_csv : Fonction de Pandas qui lit un fichier CSV (Comma Separated Values) et le convertit en un DataFrame.
# Un DataFrame est une structure de données tabulaire en 2 dimensions avec des axes étiquetés (lignes et colonnes).
df = pd.read_csv("Algerian_forest_fires_dataset.csv")

# Petit test pour voir si ça marche
# df.head() : Affiche les 5 premières lignes du DataFrame. Utile pour vérifier rapidement que les données sont bien chargées.
print(df.head())

# Ajout de la feature 'Region' avant le nettoyage
# .loc : Accesseur pour accéder à un groupe de lignes et de colonnes par étiquettes ou un tableau booléen.
df.loc[:122, 'Region'] = 0  # 0 pour Bejaia
df.loc[122:, 'Region'] = 1  # 1 pour Sidi-Bel Abbes
# .astype(int) : Convertit le type de données de la colonne en entier (integer).
df['Region'] = df['Region'].astype(int)

# Ensuite, tu peux faire ton drop comme prévu
# .drop : Supprime des lignes ou des colonnes spécifiées par étiquette ou index.
# .reset_index : Réinitialise l'index du DataFrame (par défaut il ajoute une nouvelle colonne "index", drop=True évite cela).
df = df.drop(index=[122, 123, 124], errors='ignore').reset_index(drop=True)

# 2. Nettoyage des espaces invisibles dans les noms de colonnes
# .str.strip() : Méthode vectorisée pour supprimer les espaces blancs au début et à la fin de chaque chaîne de caractères.
df.columns = df.columns.str.strip()

# 3. Vérification des colonnes
print("Colonnes disponibles :", df.columns)

# .tail() : Affiche les 5 dernières lignes du DataFrame.
print(df.tail())


# 3. Conversion de TOUTES les colonnes numériques
# On liste toutes les colonnes qui doivent être des chiffres
numeric_cols = ['day', 'month', 'year', 'Temperature', 'RH', 'Ws', 
                   'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']

for col in numeric_cols:
    # 'coerce' va transformer les erreurs (texte) en NaN (vide)
    # pd.to_numeric : Convertit l'argument en type numérique (float ou int). Si errors='coerce', remplace les valeurs invalides par NaN.
    df[col] = pd.to_numeric(df[col], errors='coerce')

# On supprime les lignes qui ont échoué à la conversion (les restes de texte)
# .dropna() : Supprime les lignes (axis=0) contenant des valeurs manquantes (NaN).
df = df.dropna().reset_index(drop=True)

# 4. Correction de l'Encodage (Classification)
# On nettoie d'abord les espaces
df['Classes'] = df['Classes'].astype(str).str.strip()

# La logique corrigée : Si ça contient "not", c'est 0. Sinon c'est 1.
# .apply : Applique une fonction le long d'un axe du DataFrame ou sur une Series. Ici, une fonction lambda.
df['Classes_Bin'] = df['Classes'].apply(lambda x: 0 if 'not' in x else 1)

# --- VÉRIFICATION ---
print("--- Types des colonnes (Doit être float ou int) ---")
# .info() : Affiche un résumé concis du DataFrame, y compris le type d'index, les dtypes des colonnes, les valeurs non nulles et l'utilisation de la mémoire.
print(df.info()) 
print("\n--- Vérification Encodage (not fire doit être 0) ---")
print(df[['Classes', 'Classes_Bin']].head())
print("\n--- ÉTAPE 2 : Visualisation ---")

# 1. Nettoyage spécifique de la colonne 'Classes' (pour la visualisation et l'étape 3)
# On enlève les espaces invisibles et on met tout en minuscules
# .str.lower() : Convertit toutes les chaînes de caractères en minuscules.
df['Classes'] = df['Classes'].astype(str).str.strip().str.lower()

# 2. Matrice de Corrélation
# On exclut la colonne 'Classes' car elle est encore en texte pour l'instant
plt.figure(figsize=(10, 8))
# .corr() : Calcule la corrélation par paire de colonnes (par défaut Pearson).
# sns.heatmap : Trace des données rectangulaires sous forme de matrice colorée.
sns.heatmap(df.drop(columns=['Classes']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de Corrélation")
plt.show()

# --- NOUVEAU : Visualisation pour le choix du modèle de Régression ---
print("\n--- Visualisation des relations pour la Régression ---")

# On définit les variables qu'on veut analyser
features_vis = ['Temperature', 'RH', 'Ws', 'Rain']
target_vis = 'FWI'

# 1. Nuages de points (Scatter plots) : Feature vs Target
# Cela aide à voir si la relation est Linéaire (Ligne droite) ou Non-Linéaire (Courbe)
plt.figure(figsize=(20, 5))
for i, col in enumerate(features_vis):
    plt.subplot(1, 4, i+1)
    sns.scatterplot(x=df[col], y=df[target_vis], alpha=0.6, color='teal')
    plt.title(f"Relation : {col} vs FWI")
    plt.xlabel(col)
    plt.ylabel("FWI (Target)")
    plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# 2. Pairplot : Voir les relations entre TOUTES les variables d'entrée
# Utile pour détecter la multicollinéarité (si deux variables d'entrée sont trop corrélées)
print("Génération du Pairplot...")
sns.pairplot(df[features_vis + [target_vis]], diag_kind='kde')
plt.suptitle("Relations croisées (Features + Target)", y=1.02)
plt.show()


# 2. Définition des Features (X) et des Cibles (y)
# Pour simplifier, on utilise la météo de base pour prédire
features_list = ['Temperature', 'RH', 'Ws', 'Rain'] 

X = df[features_list]
# y : La variable cible (Target)
y_regression_target = df['FWI']           # Cible pour l'étape 4 (Nombre continu)
y_classification_target = df['Classes_Bin'] # Cible pour l'étape 5 (0 ou 1)

# Split commun (régression + classification) AVANT la standardisation pour éviter le data leakage
X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
    X,
    y_regression_target,
    y_classification_target,
    test_size=0.2,
    random_state=42,
    stratify=y_classification_target
)

# 3. Standardisation (fit uniquement sur le train pour éviter le data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#ÉTAPE 4 : Régression (Prédire l'indice FWI)

# Modèle 3 : Régression Polynomial + Ridge avec GridSearchCV (validation croisée)
param_grid = {
    'poly__degree': [2, 3, 4],
    'ridge__alpha': [0.01, 0.1, 1, 5, 10, 20, 50]
}

pipe = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('ridge', Ridge())
])

grid_cv = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_cv.fit(X_train_scaled, y_train_reg)
best_model = grid_cv.best_estimator_
best_r2_cv = grid_cv.best_score_

# Évaluation sur le jeu de test tenu à part
y_pred_poly = best_model.predict(X_test_scaled)
r2_test = r2_score(y_test_reg, y_pred_poly)

print(f"Meilleur modèle (CV) : degree={best_model.named_steps['poly'].degree}, alpha={best_model.named_steps['ridge'].alpha}")
print(f"R² moyen CV : {best_r2_cv:.4f} | R² test hold-out : {r2_test:.4f}")

# Visualisation des prédictions polynomiales vs vraies valeurs (meilleur modèle)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_poly, color='green', alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel("Vraies Valeurs (FWI)")
plt.ylabel("Prédictions (FWI) - Poly")
plt.title("Régression Polynomial (GridSearchCV) : Vraies vs Prédictions")
plt.show()

# On conserve ces références pour la sauvegarde
poly_features = best_model.named_steps['poly']
ridge_poly_model = best_model.named_steps['ridge']

# ÉTAPE 5 : Classification (Feu vs Pas feu) 

print("--- Test des hyperparamètres KNN ---")
best_knn_acc = float("-inf")
best_k = None
for k in range(1, 16, 2):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_scaled, y_train_cls)
    acc = accuracy_score(y_test_cls, knn_model.predict(X_test_scaled))
    print(f"Précision KNN (K={k}) : {acc:.2%}")
    if acc > best_knn_acc:
        best_knn_acc = acc
        best_k = k

print(f"Meilleur KNN : K={best_k} avec précision={best_knn_acc:.2%}")

# --- KNN FINAL ET VISUALISATION ---
final_knn_model = KNeighborsClassifier(n_neighbors=best_k)
final_knn_model.fit(X_train_scaled, y_train_cls)

# CORRECTION ICI : On sauvegarde la prédiction dans une variable
# .predict : Prédit les étiquettes de classe pour les données fournies.
y_pred_knn = final_knn_model.predict(X_test_scaled)

print("\nRapport détaillé (K=5) :")
# classification_report : Construit un rapport texte montrant les principales métriques de classification (précision, rappel, f1-score).
print(classification_report(y_test_cls, y_pred_knn))

# Matrice de Confusion
# confusion_matrix : Calcule la matrice de confusion pour évaluer la précision d'une classification.
cm = confusion_matrix(y_test_cls, y_pred_knn)
# ConfusionMatrixDisplay : Visualisation de la matrice de confusion.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Pas de Feu', 'Feu'])

plt.figure(figsize=(6, 6))
# On utilise plot() directement, il gère la figure
disp.plot(cmap='Blues', values_format='d')
plt.title("Matrice de Confusion - KNN")
plt.show()

# --- ARBRE DE DÉCISION ET VISUALISATION ---
# DecisionTreeClassifier : Un classificateur par arbre de décision.
# max_depth=3 limite la profondeur de l'arbre pour éviter le surapprentissage et améliorer la lisibilité.
decision_tree_model = DecisionTreeClassifier(max_depth=3, random_state=42) # Profondeur limitée pour la lisibilité
decision_tree_model.fit(X_train_scaled, y_train_cls)
y_pred_dt = decision_tree_model.predict(X_test_scaled)

print("\n--- Arbre de Décision ---")
print(f"Précision Arbre de Décision : {accuracy_score(y_test_cls, y_pred_dt):.2%}")
print(classification_report(y_test_cls, y_pred_dt))

# Visualisation de l'arbre
plt.figure(figsize=(20, 10))
# plot_tree : Trace un arbre de décision.
plot_tree(decision_tree_model, 
          feature_names=X.columns, 
          class_names=['Pas de Feu', 'Feu'], 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Visualisation de l'Arbre de Décision")
plt.show()

# Visualisation Bonus : Importance des Features
# Feature importances : L'importance des caractéristiques (plus le score est élevé, plus la feature est importante pour la prédiction).
feature_importance = pd.Series(decision_tree_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar', color='orange')
plt.title("Importance des variables (Arbre de Décision)")
plt.ylabel("Importance")
plt.show()

import joblib

print("--- ÉTAPE 8 : Sauvegarde pour le déploiement ---")

# On sauvegarde :
# 1. Le Scaler (pour mettre les entrées utilisateur à la bonne échelle)
# 2. Le modèle de Classification (KNN)
# 3. Le modèle de Régression (Linear ou Ridge)

joblib.dump(scaler, 'mon_scaler.pkl')
joblib.dump(final_knn_model, 'mon_modele_classification.pkl')
joblib.dump(poly_features, 'mon_poly_features.pkl')
joblib.dump(ridge_poly_model, 'mon_modele_regression.pkl') # Modèle polynomial pour la régression

print("Modèles et Scaler sauvegardés avec succès ! (fichiers .pkl créés)")
