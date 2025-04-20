# %% [markdown]
"""
# Flight Delay Prediction System

This notebook combines data cleaning, preprocessing, and multiple modeling approaches to predict flight delays.
"""
# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras import layers, models

# %% [markdown]
"""
## Data Cleaning Module
"""
# %%
def clean_data():
    def insert_colon(s):
        mid = len(s) // 2
        return s[:mid] + ':' + s[mid:]

    df = pd.read_csv("flight_delays_train.csv")
    
    # Clean columns
    df['Month'] = df['Month'].map(lambda x: x.lstrip("c-"))
    df['DayofMonth'] = df['DayofMonth'].map(lambda x: x.lstrip("c-"))
    df['DayOfWeek'] = df['DayOfWeek'].map(lambda x: x.lstrip("c-"))
    
    # Convert target to binary
    df["dep_delayed_15min"] = df["dep_delayed_15min"].replace("N", 0)
    df["dep_delayed_15min"] = df["dep_delayed_15min"].replace("Y", 1)
    
    # Format departure time
    df['DepTime'] = df['DepTime'].astype(str)
    df['DepTime'] = df['DepTime'].apply(insert_colon)
    
    # Create date column
    df['Month/Day'] = df['Month'].astype(str) + '/' + df['DayofMonth'].astype(str)
    df['Month/Day'] = pd.to_datetime(df['Month/Day'] + '/2024', format='%m/%d/%Y')
    df = df.drop(['Month', 'DayofMonth'], axis=1)
    
    df.to_csv("cleaned.csv", index=False)
    return df

# %% [markdown]
"""
## Data Preprocessing Module
"""
# %%
def preprocess_data(train_file="flight_delays_train.csv", test_file="flight_delays_test.csv"):
    # Load datasets
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # Clean columns
    for df in [df_train, df_test]:
        df['Month'] = df['Month'].map(lambda x: x.lstrip("c-"))
        df['DayofMonth'] = df['DayofMonth'].map(lambda x: x.lstrip("c-"))
        df['DayOfWeek'] = df['DayOfWeek'].map(lambda x: x.lstrip("c-"))

    # Convert target to binary in training data
    if 'dep_delayed_15min' in df_train.columns:
        df_train["dep_delayed_15min"] = df_train["dep_delayed_15min"].replace("N", 0)
        df_train["dep_delayed_15min"] = df_train["dep_delayed_15min"].replace("Y", 1)

    # Process departure time
    for df in [df_train, df_test]:
        df['DepTime'] = df['DepTime'].astype(str).str.zfill(4)
        df['DepHour'] = df['DepTime'].str[:2].astype(int)
        df['DepMinute'] = df['DepTime'].str[2:].astype(int)

    # Combine for consistent one-hot encoding
    combined_df = pd.concat([df_train, df_test], keys=['train', 'test'])

    # One-hot encoding
    combined_df_hour_one_hot = pd.get_dummies(combined_df['DepHour'], prefix='Hour')
    combined_df_minute_one_hot = pd.get_dummies(combined_df['DepMinute'], prefix='Minute')
    combined_df_encoded = pd.get_dummies(combined_df, columns=["Origin", "Dest", "UniqueCarrier"])
    
    # Convert boolean to int
    combined_df_encoded = combined_df_encoded.replace(False, 0)
    combined_df_encoded = combined_df_encoded.replace(True, 1)
    
    # Combine all features
    combined_df_encoded = pd.concat([combined_df_encoded, combined_df_hour_one_hot, combined_df_minute_one_hot], axis=1)
    combined_df_encoded.drop(columns=['DepTime', 'DepHour', 'DepMinute'], inplace=True)

    # Split back
    df_train_encoded = combined_df_encoded.xs('train')
    df_test_encoded = combined_df_encoded.xs('test')

    # Save processed data
    df_train_encoded.to_csv("df_encoded_train.csv", index=False)
    df_test_encoded.to_csv("df_encoded_test.csv", index=False)
    
    return df_train_encoded, df_test_encoded

# %% [markdown]
"""
## CNN Model
"""
# %%
def train_cnn():
    # Load and preprocess
    df = pd.read_csv("flight_delays_train.csv")
    
    # Encode categorical features
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Extract features and labels
    X = df.drop(columns=['dep_delayed_15min'])
    y = df['dep_delayed_15min']

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    X_reshaped = np.array(X_reshaped)
    y = np.array(y)

    # Define CNN model
    def create_simple_cnn(input_shape):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 1), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 1)))
        model.add(layers.Conv2D(64, (3, 1), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    # Train model
    input_shape = (X_reshaped.shape[1], X_reshaped.shape[2], 1)
    model = create_simple_cnn(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_reshaped, y, epochs=10, batch_size=8, validation_split=0.2)

    # Save model
    model.save('simple_cnn_model.h5')
    return model

# %% [markdown]
"""
## Decision Tree Model
"""
# %%
def train_decision_tree():
    # Load data
    df_train = pd.read_csv("df_encoded_train.csv")
    X_train = df_train.drop("dep_delayed_15min", axis=1)
    Y_train = df_train["dep_delayed_15min"]

    # Load test data
    df_test = pd.read_csv("df_encoded_test.csv")

    # Align columns
    missing_cols = set(X_train.columns) - set(df_test.columns)
    for c in missing_cols:
        df_test[c] = 0
    df_test = df_test[X_train.columns]

    # Train model
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)

    # Predict and save
    results = model.predict(df_test)
    df_test["dep_delayed_15min"] = results
    df_test[["dep_delayed_15min"]].to_csv("results_decision_tree.csv", index=False)
    
    return model

# %% [markdown]
"""
## Logistic Regression Model
"""
# %%
def train_logistic_regression():
    df = pd.read_csv("df_encoded_train.csv")
    X = df.drop("dep_delayed_15min", axis=1)
    Y = df["dep_delayed_15min"]

    dftest = pd.read_csv("df_encoded_test.csv")

    model = LogisticRegression()
    model.fit(X,Y)
    results = model.predict(dftest)
    
    pd.DataFrame(results, columns=['dep_delayed_15min']).to_csv("results_logistic.csv", index=False)
    return model

# %% [markdown]
"""
## Main Execution
"""
# %%
if __name__ == "__main__":
    print("Cleaning data...")
    clean_data()
    
    print("\nPreprocessing data...")
    preprocess_data()
    
    print("\nTraining CNN model...")
    cnn_model = train_cnn()
    
    print("\nTraining Decision Tree model...")
    dt_model = train_decision_tree()
    
    print("\nTraining Logistic Regression model...")
    lr_model = train_logistic_regression()
    
    print("\nAll models trained and predictions saved!")