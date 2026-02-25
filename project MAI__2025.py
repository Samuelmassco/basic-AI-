# Dataset: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, make_scorer
)

def main():
    # 1. Data loading and exploration
    columns = [
        "id", "diagnosis",
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]

    # Get the path to data.csv relative to this script
    script_dir = Path(__file__).parent
    csv_path = script_dir / "data.csv"

    # Automatic detection of the separator (comma or semicolon); we take advantage of the fact that it's a CSV file.
    with open(str(csv_path), "r", encoding="utf-8") as f:
        first_line = f.readline()
        sep = ',' if first_line.count(',') >= first_line.count(';') else ';'

    # Use pandas to load the CSV file (we can then use all pandas functions to explore the dataset)
    df = pd.read_csv(str(csv_path), header=None, names=columns, sep=sep)

    # Display basic information about the dataset
    print(df.info())
    print(df.head())

    # Count the number of healthy and sick patients
    counts = df['diagnosis'].value_counts()
    print(f" Healthy patients (B): {counts.get('B', 0)}")
    print(f" Sick patients (M): {counts.get('M', 0)}")


    # 2. Correlation analysis: which attributes influence the diagnosis the most

    df['diagnosis_encoded'] = df['diagnosis'].map({'B': 0, 'M': 1})#else we would have a probleme doing computation cause there would be some string value
    #we associate the 1 if sick 0 if healthy
    correlations = df.drop(columns=['id', 'diagnosis']).corr()['diagnosis_encoded'].drop('diagnosis_encoded')
    correlations_sorted = correlations.abs().sort_values(ascending=False)

    # Display the results
    print("Top 10 attributes most correlated with diagnosis (absolute values):\n")
    # Attributes with high absolute correlation are considered more discriminative,
    # as they tend to vary significantly between benign and malignant tumors,
    # and are therefore potentially good indicators for diagnosis.
    print(correlations_sorted.head(10))

    top_features = correlations_sorted.head(10)
    plt.figure(figsize=(10, 6))
    top_features.plot(kind='barh', color='mediumseagreen')
    plt.title("Top 10 attributes most correlated with diagnosis (M vs B)")
    plt.xlabel("Correlation coefficient (absolute)")
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



    # 3. PCA (Principal Component Analysis)

    # Select features (excluding 'id' and 'diagnosis')
    features = df.drop(columns=['id', 'diagnosis', 'diagnosis_encoded'])

    # Standardize the data (important for PCA)
    scaler_pca = StandardScaler()
    scaled_features = scaler_pca.fit_transform(features)

    # Apply PCA (fit only — we just need the explained variance ratios)
    pca = PCA()
    pca.fit(scaled_features)

    # Explained variance by each principal component
    explained_variance_ratio = pca.explained_variance_ratio_
    print("\n Variance explained by each principal component:\n", explained_variance_ratio)

    # Visualization of explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio.cumsum(), marker='o', linestyle='--')
    plt.title('Cumulative explained variance by principal components')
    plt.xlabel('Number of principal components')
    plt.ylabel('Cumulative explained variance')
    plt.grid(True)
    plt.show()

    # Determine number of components to keep (e.g., those explaining at least 95% of variance)
    cumulative_variance = explained_variance_ratio.cumsum()
    n_components = next(i for i, v in enumerate(cumulative_variance) if v >= 0.95) + 1
    print(f"\n Number of principal components needed to explain 95% of the variance: {n_components}")

    # Apply PCA with the selected number of components
    pca_final = PCA(n_components=n_components)
    principal_components_final = pca_final.fit_transform(scaled_features)

    print("\n Data dimensions after applying PCA:", principal_components_final.shape)

    # Create a new DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components_final,
                          columns=[f'principal_component_{i + 1}' for i in range(n_components)])

    # Add the diagnosis column for future analysis
    pca_df['diagnosis'] = df['diagnosis']

    print("\n Preview of the DataFrame with principal components:\n", pca_df.head())

    # 4. PCA (3D Visualization)
    # Each point represents an observation in the dataset, positioned based on its three main components.
    # The color of each point shows whether the tumor was benign (green) or malignant (red).
    # This helps visualize whether the two diagnosis classes are separable in this reduced-dimension space.

    # Reuse scaled_features from section 3
    # Apply PCA to obtain 3 components
    n_components_3d = 3
    pca_3d = PCA(n_components=n_components_3d)
    principal_components_3d = pca_3d.fit_transform(scaled_features)

    print(f"\n Data dimensions after PCA (3 components): {principal_components_3d.shape}")

    # Create a DataFrame for the 3 principal components
    pca_df_3d = pd.DataFrame(data=principal_components_3d,
                             columns=['principal_component_1', 'principal_component_2', 'principal_component_3'])

    # Add diagnosis column for coloring
    pca_df_3d['diagnosis'] = df['diagnosis']

    print("\n Preview of the DataFrame with 3 principal components:\n", pca_df_3d.head())

    # 3D Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of points colored by diagnosis
    scatter = ax.scatter(pca_df_3d['principal_component_1'],
                         pca_df_3d['principal_component_2'],
                         pca_df_3d['principal_component_3'],
                         c=pca_df_3d['diagnosis'].map({'B': 'green', 'M': 'red'}))

    # Add labels and title
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D PCA Visualization of Breast Cancer Data')

    # Create a legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Benign'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Malignant')
    ]
    ax.legend(handles=legend_elements, title="Diagnosis")

    plt.show()


    # 5. FDA (Linear Discriminant Analysis)

    # Prepare data for LDA
    X = df.drop(columns=['id', 'diagnosis', 'diagnosis_encoded'])
    y = df['diagnosis_encoded']

    # Split the data into training and test sets (consistent split for all models)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize the data (recommended for LDA)
    scaler_lda = StandardScaler()
    X_train_scaled = scaler_lda.fit_transform(X_train)
    X_test_scaled = scaler_lda.transform(X_test)

    # Apply LDA
    lda = LinearDiscriminantAnalysis(n_components=1)  # Reduce to 1 component for visualization (optional)
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)

    print("\n Training data dimensions after LDA:", X_train_lda.shape)
    print(" Test data dimensions after LDA:", X_test_lda.shape)

    # Analyze class separability in the LDA space (optional)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train_lda[y_train == 0], [0] * (y_train == 0).sum(), label='Benign', color='green', alpha=0.7)
    plt.scatter(X_train_lda[y_train == 1], [0] * (y_train == 1).sum(), label='Malignant', color='red', alpha=0.7)
    plt.xlabel('Linear Discriminant Component 1')
    plt.yticks([])
    plt.title('Class separation in LDA space (Training data)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Simple evaluation of LDA performance (as a classifier)
    logistic_regression = LogisticRegression(max_iter=1000)
    logistic_regression.fit(X_train_lda, y_train)
    y_pred_lda = logistic_regression.predict(X_test_lda)
    accuracy_lda = accuracy_score(y_test, y_pred_lda)
    print(f"\n Accuracy of logistic regression model on LDA components: {accuracy_lda:.4f}")

    # Display coefficients of the linear discriminant function
    print("\nCoefficients of the linear discriminant function:\n", lda.coef_)


    # 6. SVM (Support Vector Machine)

    # Reusing the same train/test split as for LDA (already done above)

    # Standardization
    scaler_svm = StandardScaler()
    X_train_scaled = scaler_svm.fit_transform(X_train)
    X_test_scaled = scaler_svm.transform(X_test)

    # Training the SVM model
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_svm = svm_model.predict(X_test_scaled)

    # Evaluation
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f" SVM accuracy: {acc_svm:.4f}")
    print("\n Classification report:\n", classification_report(y_test, y_pred_svm))
    print(" Confusion matrix:\n", confusion_matrix(y_test, y_pred_svm))

    # 2D visualization with PCA to show the decision boundary
    # Note: fit PCA only on training data to avoid data leakage
    pca_vis = PCA(n_components=2)
    X_train_vis = pca_vis.fit_transform(X_train_scaled)
    X_vis = pca_vis.transform(scaler_svm.transform(X))

    # Retrain SVM for visualization on the projected training data
    svm_model_vis = SVC(kernel='linear')
    svm_model_vis.fit(X_train_vis, y_train)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap='bwr', alpha=0.6, edgecolors='k')
    plt.title('SVM - Visualization of data projected in 2D (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.legend(*scatter.legend_elements(), title="Diagnosis", loc="best")
    plt.tight_layout()
    plt.show()

    # 7. MLP (Multi-Layer Perceptron)

    # Reusing the same train/test split as previous models
    target_names = ['Benign', 'Malignant']

    # Standardization
    scaler_mlp = StandardScaler()
    X_train_mlp_scaled = scaler_mlp.fit_transform(X_train)
    X_test_mlp_scaled = scaler_mlp.transform(X_test)

    # PCA for dimensionality reduction (helps prevent overfitting)
    pca_mlp = PCA(n_components=0.95)  # Keep 95% of variance
    X_train_mlp_pca = pca_mlp.fit_transform(X_train_mlp_scaled)
    X_test_mlp_pca = pca_mlp.transform(X_test_mlp_scaled)

    # Parameter grid for hyperparameter tuning
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [300],
        'early_stopping': [True],
        'validation_fraction': [0.15],
        'n_iter_no_change': [10]
    }

    # Initialize the MLP model
    mlp = MLPClassifier(solver='adam', random_state=42)

    # Scoring metric: macro F1-score
    f1_scorer = make_scorer(f1_score, average='macro')

    # Grid search with cross-validation
    grid = GridSearchCV(mlp, param_grid, cv=5, scoring=f1_scorer, verbose=2, n_jobs=-1)
    grid.fit(X_train_mlp_pca, y_train)

    # Best hyperparameters
    print("Best hyperparameters:", grid.best_params_)

    # Evaluation on the test set
    best_mlp = grid.best_estimator_
    y_pred_mlp = best_mlp.predict(X_test_mlp_pca)

    print("\n Final evaluation of the MLP on the test set:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_mlp):.4f}")
    print(f"F1-score (macro): {f1_score(y_test, y_pred_mlp, average='macro'):.4f}")
    print(f"Recall (macro): {recall_score(y_test, y_pred_mlp, average='macro'):.4f}")
    print(f"Precision (macro): {precision_score(y_test, y_pred_mlp, average='macro'):.4f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred_mlp))
    print("\nClassification report:\n", classification_report(y_test, y_pred_mlp, target_names=target_names))







if __name__ == "__main__":
    main()

