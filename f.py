# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # For saving and loading the model

# --- Configuration ---
# Path to your dataset file. Replace with the actual path to your NSL-KDD or similar dataset.
# Make sure the dataset is in a CSV format and contains 'label' or 'attack' column.
DATASET_PATH = 'KDDTrain+.txt' # Example: For NSL-KDD, rename it appropriately
# Define column names if your dataset doesn't have a header.
# This list is typical for NSL-KDD. Adjust if using a different dataset.
FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack', 'level' # 'attack' will be our target variable
]

# --- 1. Data Acquisition and Preprocessing ---
def load_and_preprocess_data(file_path):
    """
    Loads the dataset, handles categorical features, and scales numerical features.

    Args:
        file_path (str): The path to the dataset file.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Features (X)
            - pd.Series: Target labels (y)
            - list: List of scaled feature names
            - sklearn.preprocessing.StandardScaler: The fitted scaler object
            - dict: Dictionary of fitted LabelEncoders for categorical columns
    """
    print(f"Loading data from {file_path}...")
    try:
        # Load the dataset. If it has no header, use 'names'.
        df = pd.read_csv(file_path, names=FEATURE_NAMES, index_col=False)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}. Please check the path.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        exit()

    # Drop the 'level' column if it exists, as 'attack' is sufficient for binary classification
    if 'level' in df.columns:
        df = df.drop('level', axis=1)

    # Separate features (X) and target (y)
    X = df.drop('attack', axis=1)
    y = df['attack']

    # --- Preprocessing ---
    # Convert 'attack' column to binary: 'normal' is 0, anything else is 1 (malicious)
    y = y.apply(lambda x: 0 if x == 'normal' else 1)

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns

    # Apply Label Encoding to categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"Encoded categorical column: {col}")

    # Apply StandardScaler to numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    print("Scaled numerical features.")

    return X, y, X.columns.tolist(), scaler, label_encoders

# --- 2. Model Training ---
def train_model(X_train, y_train):
    """
    Trains a RandomForestClassifier model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        sklearn.ensemble.RandomForestClassifier: The trained model.
    """
    print("Training the RandomForestClassifier model...")
    # Initialize the RandomForestClassifier. You can experiment with different parameters.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

# --- 3. Evaluation ---
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model and prints performance metrics.

    Args:
        model (sklearn.base.BaseEstimator): The trained machine learning model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): True labels for the test set.
    """
    print("Evaluating the model...")
    y_pred = model.predict(X_test)

    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    # target_names will be ['Normal', 'Attack'] as per our binary conversion
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    print("\nConfusion Matrix:")
    # Rows are true labels, columns are predicted labels
    # [[True Normal, False Attack (Type I error)]
    #  [False Normal (Type II error), True Attack]]
    print(confusion_matrix(y_test, y_pred))
    print("--------------------------------")

# --- Main execution flow ---
if __name__ == "__main__":
    # Load and preprocess data
    X, y, feature_names, scaler, label_encoders = load_and_preprocess_data(DATASET_PATH)

    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # Train the model
    ids_model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(ids_model, X_test, y_test)

    # --- Saving and Loading the model (Example) ---
    MODEL_FILENAME = 'intrusion_detection_model.joblib'
    SCALER_FILENAME = 'feature_scaler.joblib'
    ENCODERS_FILENAME = 'label_encoders.joblib'
    FEATURE_NAMES_FILENAME = 'feature_names.joblib'

    print(f"\nSaving the trained model to {MODEL_FILENAME}...")
    joblib.dump(ids_model, MODEL_FILENAME)
    print(f"Saving the scaler to {SCALER_FILENAME}...")
    joblib.dump(scaler, SCALER_FILENAME)
    print(f"Saving the label encoders to {ENCODERS_FILENAME}...")
    joblib.dump(label_encoders, ENCODERS_FILENAME)
    print(f"Saving feature names to {FEATURE_NAMES_FILENAME}...")
    joblib.dump(feature_names, FEATURE_NAMES_FILENAME)
    print("Model and preprocessing tools saved successfully.")

    # Example of how to load the model and make a prediction on new data
    # In a real scenario, 'new_data_point' would come from live network traffic.
    print("\n--- Example: Loading model and making a prediction ---")
    print("Loading model and preprocessing tools...")
    loaded_model = joblib.load(MODEL_FILENAME)
    loaded_scaler = joblib.load(SCALER_FILENAME)
    loaded_encoders = joblib.load(ENCODERS_FILENAME)
    loaded_feature_names = joblib.load(FEATURE_NAMES_FILENAME)
    print("Model and tools loaded.")

    # Create a dummy new data point for demonstration
    # This data point must have the same features as the training data
    # Let's create a hypothetical 'normal' connection data point.
    # IMPORTANT: Ensure the order of features matches 'feature_names'
    # And categorical values are in their original string form before encoding.
    dummy_data = {
        'duration': 0, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF',
        'src_bytes': 200, 'dst_bytes': 1000, 'land': 0, 'wrong_fragment': 0,
        'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 1,
        'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
        'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
        'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
        'count': 10, 'srv_count': 10, 'serror_rate': 0.0, 'srv_serror_rate': 0.0,
        'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0,
        'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0, 'dst_host_count': 255,
        'dst_host_srv_count': 255, 'dst_host_same_srv_rate': 1.0,
        'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 0.0,
        'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0
    }
    # Convert to DataFrame, ensuring column order matches
    new_data_point_df = pd.DataFrame([dummy_data], columns=loaded_feature_names)

    # Preprocess the new data point using the loaded encoders and scaler
    # Handle categorical columns first
    for col, encoder in loaded_encoders.items():
        if col in new_data_point_df.columns:
            # Handle unknown categories: if a category is new, it might raise an error
            # A common approach is to ignore it or map it to a default.
            # For simplicity, we'll assume known categories for this example.
            try:
                new_data_point_df[col] = encoder.transform(new_data_point_df[col])
            except ValueError as e:
                print(f"Warning: Category in column '{col}' not seen during training. Error: {e}")
                # You might want to assign a default value or drop this column
                # For now, let's just use 0 for unknown
                new_data_point_df[col] = 0

    # Handle numerical columns
    numerical_cols = new_data_point_df.select_dtypes(include=['number']).columns
    if not numerical_cols.empty:
        new_data_point_df[numerical_cols] = loaded_scaler.transform(new_data_point_df[numerical_cols])

    # Make prediction
    prediction = loaded_model.predict(new_data_point_df)
    prediction_proba = loaded_model.predict_proba(new_data_point_df)

    if prediction[0] == 0:
        print(f"Prediction for new data point: Normal (Confidence: {prediction_proba[0][0]:.2f})")
    else:
        print(f"Prediction for new data point: Attack Detected! (Confidence: {prediction_proba[0][1]:.2f})")

    print("\nTo run this code:")
    print("1. Save it as a Python file (e.g., `ids_project.py`).")
    print("2. Download an IDS dataset (e.g., NSL-KDD 'KDDTrain+.txt') and place it in the same directory.")
    print("3. Ensure you have the required libraries installed: `pip install pandas scikit-learn joblib`")
    print("4. Run from your terminal: `python ids_project.py`")


