import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)

def haversine_km(lat1, lon1, lat2, lon2):
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in km
    return 6371 * c

def manhattan_km(lat1, lon1, lat2, lon2):
    return (haversine_km(lat1, lon1, lat2, lon1) +
            haversine_km(lat1, lon1, lat1, lon2))

def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


start_time = time.time()
print("Reading File")
df = pd.read_csv("/content/drive/MyDrive/Taxirate/train.csv", nrows = 1_000_000)
print(f"Read file - {time.time() - start_time:.2f} seconds")

df = df[
    (df["pickup_longitude"].between(-74.3, -73.6)) &
    (df["pickup_latitude"].between(40.4, 41.0))
]


start_time = time.time()
print("converting data types")
float_cols = ["fare_amount","pickup_longitude","pickup_latitude",
              "dropoff_longitude","dropoff_latitude"]

df[float_cols] = df[float_cols].astype(np.float32)
df["passenger_count"] = df["passenger_count"].astype(np.int8)
print(f"Done - {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("Dropping NAs")
critical_cols = [
    "pickup_datetime","pickup_latitude","pickup_longitude",
    "dropoff_latitude","dropoff_longitude","passenger_count","fare_amount"
]

df = df.dropna(subset = critical_cols)
print(f"Dropped NAs - {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("changing to date_time")
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
print(f"Done - {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("Filtering out negatives")
df = df[df["fare_amount"] > 0]
print(f"Done - {time.time() - start_time:.2f} seconds")


print(df.columns)

print(df.head())

start_time = time.time()
print("Extracting Month")
df["month"] = df["pickup_datetime"].dt.month
print(f"Done - {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("Calculating Haversine Distance..")
df["haversine_km"] = haversine_km(df["pickup_latitude"], df["pickup_longitude"],
                                  df["dropoff_latitude"], df["dropoff_longitude"])
print(f"Done - {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("Calculating manhattan distance...")
df["manhattan_km"] = manhattan_km(df["pickup_latitude"], df["pickup_longitude"],
                                  df["dropoff_latitude"], df["dropoff_longitude"])
print(f"Done - {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("Calculating bearing distance")
df["bearing_deg"]  = bearing_deg( df["pickup_latitude"], df["pickup_longitude"],
                                  df["dropoff_latitude"], df["dropoff_longitude"])
print(f"Done - {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("removing outliers")
df = df[(df["fare_amount"].between(2.5, 200)) &
        (df["haversine_km"].between(0, 100)) &
        (df["passenger_count"].between(1, 6))].copy()
print(f"Done - {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("extracting hour")
df['hour'] = df['pickup_datetime'].dt.hour
print(f"Done - {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("Calculating sin/cos of hour")
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
print(f"Done - {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("Calculating sin/cos of day")
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
print(f"Done - {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("Calculating if it is a weekend")
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
print(f"Done - {time.time() - start_time:.2f} seconds")


train_columns = [
    "haversine_km", "hour_sin", "hour_cos", "manhattan_km", "is_weekend", 
    "day_sin", "day_cos", "passenger_count", 'bearing_deg', "month"
]
start_time = time.time()
print("Splitting into X and Y")
X = df[train_columns]
y = df["fare_amount"]
print(f"done - {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("Train-val split")
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=123, test_size=0.2)
print(f"Done - {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("Converting to Numpy")
X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
X_val_np   = X_val.to_numpy(dtype=np.float32, copy=False)
y_train_np = y_train.to_numpy(copy=False)
y_val_np   = y_val.to_numpy(copy=False)
print(f"Done - {time.time() - start_time:.2f} seconds")

del X, y, X_train, X_val, y_train, y_val
import gc; gc.collect()

start_time = time.time()
print("model selection")

model = XGBRegressor(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",   # FAST for millions of rows
    objective="reg:squarederror",
    random_state=123,
    n_jobs=4
)

print(f"Done - {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("Fitting Model")
model.fit(X_train_np, y_train_np)
print(f"Done - {time.time() - start_time:.2f} seconds")


start_time = time.time()
print("Calculating Y_hat")
y_hat = model.predict(X_train_np)
print(f"done - {time.time() - start_time:.2f} seconds")
print("fit  on training data")
print(mean_absolute_error(y_train_np, y_hat))

y_pred = model.predict(X_val_np)
print("fit on validation data")
print(mean_absolute_error(y_val_np, y_pred))

import joblib
joblib.dump(model, "model.joblib")
