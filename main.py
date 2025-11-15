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
df = pd.read_csv("train.csv")
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

print("loading test data")
df_test = pd.read_csv('test.csv')
print("Loaded test data")

print("converting data type")
float_cols = ["pickup_longitude","pickup_latitude",
              "dropoff_longitude","dropoff_latitude"]
df_test[float_cols] = df_test[float_cols].astype(np.float32)


print("Conversion")
df_test["pickup_datetime"] = pd.to_datetime(df_test["pickup_datetime"], errors="coerce")

df_test["month"] = df_test["pickup_datetime"].dt.month
df_test["haversine_km"] = haversine_km(df_test["pickup_latitude"], df_test["pickup_longitude"],
                                  df_test["dropoff_latitude"], df_test["dropoff_longitude"])
df_test["manhattan_km"] = manhattan_km(df_test["pickup_latitude"], df_test["pickup_longitude"],
                                  df_test["dropoff_latitude"], df_test["dropoff_longitude"])
df_test["bearing_deg"]  = bearing_deg( df_test["pickup_latitude"], df_test["pickup_longitude"],
                                  df_test["dropoff_latitude"], df_test["dropoff_longitude"])

df_test['hour'] = df_test['pickup_datetime'].dt.hour

df_test['hour_sin'] = np.sin(2 * np.pi * df_test['hour'] / 24)
df_test['hour_cos'] = np.cos(2 * np.pi * df_test['hour'] / 24)

df_test['day_of_week'] = df_test['pickup_datetime'].dt.dayofweek
df_test['day_sin'] = np.sin(2 * np.pi * df_test['day_of_week'] / 7)
df_test['day_cos'] = np.cos(2 * np.pi * df_test['day_of_week'] / 7)

df_test["is_weekend"] = (df_test["day_of_week"] >= 5).astype(int)
print(f"Done - {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("converting data types")
for c in ["haversine_km","manhattan_km","bearing_deg",
          "hour_sin","hour_cos","day_sin","day_cos"]:
    df_test[c] = df_test[c].astype(np.float32)
print(f"Done - {time.time() - start_time:.2f} seconds")

df_test["passenger_count"] = df_test["passenger_count"].astype(np.int8, errors="ignore")
df_test["is_weekend"]      = df_test["is_weekend"].astype(np.int8)
df_test["month"]           = df_test["month"].astype(np.int8)

# build NumPy test matrix
X_test_np = df_test[[
    "haversine_km","hour_sin","hour_cos","manhattan_km","is_weekend",
    "day_sin","day_cos","passenger_count","bearing_deg","month"
]].to_numpy(dtype=np.float32, copy=False)

start_time = time.time()
print("Testing")
X_test = df_test[train_columns]
y_pred = model.predict(X_test_np)
print(f"Done - {time.time() - start_time:.2f} seconds")
submission = pd.DataFrame({
    'key': df_test['key'], 
    'fare_amount': y_pred
})
submission.to_csv('submission.csv', index=False)

"""
for i in range(1, 20):
    print(f"Testing max depth {i}")
    model = RandomForestRegressor(
        max_depth=i, random_state = 123
    )

    model.fit(X_train, y_train)

    y_hat = model.predict(X_train)
    print("fit  on training data")
    print(mean_absolute_error(y_train, y_hat))

    y_pred = model.predict(X_test)
    print("fit on test data")
    print(mean_absolute_error(y_test, y_pred))

"""


#max depth 6 = best as similar train and test split
#moving to final model
"""
print("loading files....")
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
print("Loaded files :)")

print("Transforming dataset")
df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'])
df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])


df_train['hour'] = df_train['pickup_datetime'].dt.hour
df_test['hour'] = df_test['pickup_datetime'].dt.hour

df_train['hour_sin'] = np.sin(2 * np.pi * df_train['hour'] / 24)
df_test['hour_sin'] = np.sin(2 * np.pi * df_test['hour'] / 24)

df_train['hour_cos'] = np.cos(2 * np.pi * df_train['hour'] / 24)
df_test['hour_cos'] = np.cos(2 * np.pi * df_test['hour'] / 24)

df_train['day_of_week'] = df_train['pickup_datetime'].dt.dayofweek
df_test['day_of_week'] = df_test['pickup_datetime'].dt.dayofweek

df_train['day_sin'] = np.sin(2 * np.pi * df_train['day_of_week'] / 7)
df_test['day_sin'] = np.sin(2 * np.pi * df_test['day_of_week'] / 7)

df_train['day_cos'] = np.cos(2 * np.pi * df_train['day_of_week'] / 7)
df_test['day_cos'] = np.cos(2 * np.pi * df_test['day_of_week'] / 7)
print("Done with that")

print("Train test stuff")

X_full = df_train[train_columns]
y_full = df_train["fare_amount"]

X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, random_state=123, test_size=0.2)

print("Model selection")
model = RandomForestRegressor(
    max_depth = 6, random_state = 123
)

print("Fitting Model")
model.fit(X_train, y_train)
print("Model has been fitted")

import joblib
joblib.dump(model, 'taxi_fare_model.pkl')
print("Model saved to taxi_fare_model.pkl")

y_train_pred = model.predict(X_train)
print("fit  on training data")
print(mean_absolute_error(y_train, y_train_pred))

y_val_pred = model.predict(X_val)
print("fit on validation data")
print(mean_absolute_error(y_val, y_val_pred))

X_test = df_test[train_columns]
y_pred = model.predict(X_test)
submission = pd.DataFrame({
    'key': df_test['key'], 
    'fare_amount': y_pred
})
submission.to_csv('submission.csv', index=False)
"""