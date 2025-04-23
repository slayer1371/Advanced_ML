# !pip install category_encoders

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Load the training dataset
df = pd.read_csv('/content/UNSW_NB15_training-set.csv')

# Display the first 5 rows
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Print the column names and their data types
print(df.info())

#   --- Feature Engineering ---

#   1. Handle 'proto' column (REVERTED to original PCA)
ohe_proto = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
proto_encoded = ohe_proto.fit_transform(df[['proto']])

pca_proto = PCA(n_components=0.95)
proto_pca = pca_proto.fit_transform(proto_encoded)
proto_pca_df = pd.DataFrame(proto_pca, columns=[f'proto_pca_{i}' for i in range(proto_pca.shape[1])])
df = pd.concat([df.reset_index(drop=True), proto_pca_df], axis=1)
df.drop(columns=['proto'], inplace=True)

from category_encoders import TargetEncoder
from sklearn.preprocessing import RobustScaler
import numpy as np

# 2. Handle 'service' feature with target encoding
target_encoder_service = TargetEncoder()
df['service_encoded'] = target_encoder_service.fit_transform(df['service'], df['label'])
df.drop(columns=['service'], inplace=True)

# 3. Handle 'state' feature with target encoding
target_encoder_state = TargetEncoder()
df['state_encoded'] = target_encoder_state.fit_transform(df['state'], df['label'])
df.drop(columns=['state'], inplace=True)

# 4. Scale numerical features with RobustScaler
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
numerical_cols.remove('label')
numerical_cols.remove('id')

# Apply log transformation to highly skewed columns
highly_skewed_cols = [
    'trans_depth', 'response_body_len', 'sbytes', 'sloss', 'dloss', 'proto_pca_15', 'spkts', 
    'dbytes', 'dpkts', 'dinpkt', 'djit', 'proto_pca_7', 'proto_pca_57', 'ct_flw_http_mthd', 
    'proto_pca_6', 'proto_pca_12', 'proto_pca_17', 'proto_pca_64', 'sjit', 'proto_pca_33', 
    'proto_pca_70', 'proto_pca_27', 'proto_pca_53', 'proto_pca_49', 'proto_pca_55', 'proto_pca_29', 
    'proto_pca_14', 'proto_pca_13', 'ct_ftp_cmd', 'is_ftp_login', 'proto_pca_23', 'proto_pca_28', 
    'proto_pca_10', 'proto_pca_39', 'proto_pca_35', 'sload', 'proto_pca_30', 'proto_pca_5', 
    'proto_pca_8', 'sinpkt', 'proto_pca_9', 'is_sm_ips_ports', 'proto_pca_24', 'dur', 'proto_pca_25', 
    'proto_pca_37', 'proto_pca_22', 'synack', 'proto_pca_60', 'ackdat', 'proto_pca_26', 'proto_pca_44', 
    'dload', 'proto_pca_42', 'proto_pca_62', 'proto_pca_45', 'proto_pca_18', 'proto_pca_48', 'tcprtt', 
    'smean', 'proto_pca_38', 'proto_pca_68', 'proto_pca_40', 'proto_pca_66', 'rate', 'proto_pca_50', 
    'dmean', 'proto_pca_43', 'proto_pca_1', 'ct_src_dport_ltm', 'proto_pca_59', 'ct_dst_ltm', 
    'proto_pca_47', 'proto_pca_51', 'ct_src_ltm', 'proto_pca_54', 'ct_dst_sport_ltm', 'proto_pca_65', 
    'proto_pca_2', 'ct_dst_src_ltm', 'ct_srv_dst', 'ct_srv_src', 'proto_pca_46', 'proto_pca_67', 
    'proto_pca_69', 'proto_pca_58', 'proto_pca_34', 'proto_pca_71', 'proto_pca_56', 'stcpb', 'dtcpb'
]

# Ensure only valid columns are processed (in case some are missing)
highly_skewed_cols = [col for col in highly_skewed_cols if col in df.columns]
df[highly_skewed_cols] = df[highly_skewed_cols].apply(
    lambda x: np.log1p(x) if x.min() >= 0 else np.log(x - x.min() + 1)
)

# Scale numerical features
scaler = RobustScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 5. Drop unnecessary columns
df = df.drop(columns=['id', 'attack_cat'])

from imblearn.over_sampling import ADASYN
import pandas as pd
import numpy as np

# Separate features (X) and target (y)
X = df.drop(columns=['label'])
y = df['label']

# Split data into training and validation sets before resampling
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Use normal data only for training (label == 0)
X_train_normal = X_train[y_train == 0]
y_train_normal = y_train[y_train == 0]

# Apply ADASYN to validation set for balanced evaluation
adasyn = ADASYN(random_state=42)
X_val_resampled, y_val_resampled = adasyn.fit_resample(X_val, y_val)

# Convert to DataFrames
X_train_normal_df = pd.DataFrame(X_train_normal, columns=X.columns)
y_train_normal_df = pd.DataFrame(y_train_normal, columns=['label'])
X_val_resampled_df = pd.DataFrame(X_val_resampled, columns=X.columns)
y_val_resampled_df = pd.DataFrame(y_val_resampled, columns=['label'])

# Concatenate for downstream use
df_train_normal = pd.concat([X_train_normal_df, y_train_normal_df], axis=1)
df_val_resampled = pd.concat([X_val_resampled_df, y_val_resampled_df], axis=1)

# Display shapes and distributions
print("\nOriginal data shape:", X.shape, y.shape)
print("Training (normal) data shape:", X_train_normal.shape, y_train_normal.shape)
print("Resampled validation data shape:", X_val_resampled.shape, y_val_resampled.shape)

print("\nDistribution of 'label' in resampled validation set:")
print(y_val_resampled.value_counts().to_markdown(numalign="left", stralign="left"))

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def create_autoencoder(input_dim):
    # Define input layer
    inputs = Input(shape=(input_dim,))
    
    # Encoder
    x = Dense(150, activation='relu')(inputs)
    x = Dropout(0.2)(x)  # Add dropout to prevent overfitting
    x = Dense(75, activation='relu')(x)
    x = Dropout(0.2)(x)
    encoded = Dense(32, activation='relu')(x)  # Latent space
    
    # Decoder
    x = Dense(75, activation='relu')(encoded)
    x = Dense(150, activation='relu')(x)
    decoded = Dense(input_dim, activation=None)(x)  # Linear activation for standardized features
    
    # Autoencoder model
    autoencoder = Model(inputs=inputs, outputs=decoded, name='autoencoder')
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mae')
    
    return autoencoder

# Create and train autoencoder
input_dim = X_train_normal.shape[1]
autoencoder = create_autoencoder(input_dim)

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # Increased patience
model_checkpoint = ModelCheckpoint('autoencoder_best.h5', monitor='val_loss', save_best_only=True)

history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=100,
    batch_size=256,
    validation_data=(X_val_resampled, X_val_resampled),
    callbacks=[early_stopping, model_checkpoint]
)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

from sklearn.metrics import precision_recall_fscore_support, roc_curve, roc_auc_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

# 1. Reconstruct the validation data
X_val_reconstructed = autoencoder.predict(X_val_resampled.astype(np.float32))

# 2. Calculate the reconstruction error (MAE, consistent with training loss)
mae_val = np.mean(np.abs(X_val_resampled.astype(np.float32) - X_val_reconstructed), axis=1)

# 3. Dynamic threshold selection based on F1-score
thresholds = np.linspace(mae_val.min(), mae_val.max(), 100)
f1_scores = [f1_score(y_val_resampled, mae_val > t) for t in thresholds]

# Also compute 5th and 10th percentile thresholds for comparison
threshold_5th = np.percentile(mae_val, 25)
threshold_10th = np.percentile(mae_val, 30)

print(f"Anomaly Threshold (Optimal F1): {optimal_threshold:.4f}")
print(f"Anomaly Threshold (5th percentile): {threshold_5th:.4f}")
print(f"Anomaly Threshold (10th percentile): {threshold_10th:.4f}")

# 4. Classify the validation data
y_pred_optimal = (mae_val > optimal_threshold).astype(int)
y_pred_5th = (mae_val > threshold_5th).astype(int)
y_pred_10th = (mae_val > threshold_10th).astype(int)

# 5. Calculate evaluation metrics
precision_opt, recall_opt, f1_opt, _ = precision_recall_fscore_support(y_val_resampled, y_pred_optimal, average='binary')
precision_5th, recall_5th, f1_5th, _ = precision_recall_fscore_support(y_val_resampled, y_pred_5th, average='binary')
precision_10th, recall_10th, f1_10th, _ = precision_recall_fscore_support(y_val_resampled, y_pred_10th, average='binary')

print("\nOptimal F1 Threshold:")
print(f"  Precision: {precision_opt:.4f}")
print(f"  Recall:    {recall_opt:.4f}")
print(f"  F1-score:  {f1_opt:.4f}")

print("\n25th Percentile Threshold:")
print(f"  Precision: {precision_5th:.4f}")
print(f"  Recall:    {recall_5th:.4f}")
print(f"  F1-score:  {f1_5th:.4f}")

print("\n30th Percentile Threshold:")
print(f"  Precision: {precision_10th:.4f}")
print(f"  Recall:    {recall_10th:.4f}")
print(f"  F1-score:  {f1_10th:.4f}")

# 6. ROC Curve and AUC
fpr, tpr, roc_thresholds = roc_curve(y_val_resampled, mae_val)
auc = roc_auc_score(y_val_resampled, mae_val)
print(f"\nAUC: {auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR) / Recall')
plt.title('ROC Curve')
plt.legend()
plt.show()