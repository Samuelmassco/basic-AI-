# link to the data set  https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic



# Importation des librairies essentielles
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import nécessaire pour la 3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, make_scorer
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score, precision_score, recall_score

def main():
    # 1 we analyse the data set
    columns = [
        "id", "diagnosis",
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]

    # Automatic detection of the separator (comma or semicolon); we take advantage of the fact that it's a CSV file.
    with open("data.csv", "r", encoding="utf-8") as f:
        first_line = f.readline()
        sep = ',' if first_line.count(',') >= first_line.count(';') else ';'

    # Use pandas to load the CSV file (we can then use all pandas functions to explore the dataset)
    df = pd.read_csv("data.csv", header=None, names=columns, sep=sep)

    # Display basic information about the dataset
    print(df.info())
    print(df.head())

    # Count the number of healthy and sick patients
    counts = df['diagnosis'].value_counts()
    print(f" Healthy patients (B): {counts.get('B', 0)}")
    print(f" Sick patients (M): {counts.get('M', 0)}")


    # 3 Let us now analyze which attributes have the greatest influence on the diagnosis

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



    # 4 PCA

    # Select features (excluding 'id' and 'diagnosis')
    features = df.drop(columns=['id', 'diagnosis', 'diagnosis_encoded'])

    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply PCA
    pca = PCA()
    principal_components = pca.fit_transform(scaled_features)

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

    # 4 PCA (3D Visualization)
    # Each point represents an observation in the dataset, positioned based on its three main components.
    # The color of each point shows whether the tumor was benign (green) or malignant (red).
    # This helps visualize whether the two diagnosis classes are separable in this reduced-dimension space.

    # Select features (excluding 'id' and 'diagnosis')
    features = df.drop(columns=['id', 'diagnosis', 'diagnosis_encoded'])

    # Standardize the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

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
    legend1 = ax.legend(*scatter.legend_elements(), title="Diagnosis")
    ax.add_artist(legend1)

    plt.show()


    # 5 FDA (Linear Discriminant Analysis)

    # Prepare data for LDA
    X = df.drop(columns=['id', 'diagnosis', 'diagnosis_encoded'])
    y = df['diagnosis_encoded']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
    plt.scatter(X_train_lda[y_train == 0], [0] * sum(y_train == 0), label='Benign', color='green', alpha=0.7)
    plt.scatter(X_train_lda[y_train == 1], [0] * sum(y_train == 1), label='Malignant', color='red', alpha=0.7)
    plt.xlabel('Linear Discriminant Component 1')
    plt.yticks([])
    plt.title('Class separation in LDA space (Training data)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Simple evaluation of LDA performance (as a classifier)
    from sklearn.linear_model import LogisticRegression

    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train_lda, y_train)
    y_pred_lda = logistic_regression.predict(X_test_lda)
    accuracy_lda = accuracy_score(y_test, y_pred_lda)
    print(f"\n Accuracy of logistic regression model on LDA components: {accuracy_lda:.4f}")

    # Display coefficients of the linear discriminant function
    print("\nCoefficients of the linear discriminant function:\n", lda.coef_)


    #6LDA
    # 7 SVM (Support Vector Machine)

    # Reusing the same data as for LDA
    X = df.drop(columns=['id', 'diagnosis', 'diagnosis_encoded'])
    y = df['diagnosis_encoded']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
    pca_vis = PCA(n_components=2)
    X_vis = pca_vis.fit_transform(scaler_svm.fit_transform(X))

    # Retrain SVM for visualization on the whole dataset
    svm_model_vis = SVC(kernel='linear')
    svm_model_vis.fit(X_vis, y)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap='bwr', alpha=0.6, edgecolors='k')
    plt.title('SVM - Visualization of data projected in 2D (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.legend(*scatter.legend_elements(), title="Diagnosis", loc="best")
    plt.tight_layout()
    plt.show()

    # MLP (Multi-Layer Perceptron)

    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

    # Split the dataset into training+validation and test sets
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardization
    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval)
    X_test_scaled = scaler.transform(X_test)

    # PCA for dimensionality reduction (helps prevent overfitting)
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_trainval_pca = pca.fit_transform(X_trainval_scaled)
    X_test_pca = pca.transform(X_test_scaled)

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
    grid.fit(X_trainval_pca, y_trainval)

    # Best hyperparameters
    print("Best hyperparameters:", grid.best_params_)

    # Evaluation on the test set
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_pca)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"F1-score: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='macro'):.4f}")

    # Final evaluation on the test set
    best_mlp = grid.best_estimator_
    y_pred_test = best_mlp.predict(X_test_pca)

    print("\n🧪 Final evaluation of the MLP on the test set:")
    print("Accuracy:", accuracy_score(y_test, y_pred_test))
    print("F1-score (macro):", f1_score(y_test, y_pred_test, average='macro'))
    print("Recall (macro):", recall_score(y_test, y_pred_test, average='macro'))
    print("Precision (macro):", precision_score(y_test, y_pred_test, average='macro'))
    print("\n📊 Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))
    print("\n🧾 Classification report:\n", classification_report(y_test, y_pred_test, target_names=target_names))







if __name__ == "__main__":
    main()

