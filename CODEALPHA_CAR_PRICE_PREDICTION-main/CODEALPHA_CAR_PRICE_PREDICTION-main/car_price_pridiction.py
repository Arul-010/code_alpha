
df = pd.read_csv("/content/car data (1).csv")
print(df.head())
print(df.info())
df.drop(columns=["Car_Name"], inplace=True)
print("\nMissing values:\n", df.isnull().sum())
current_year = 2025
df["Car_Age"] = current_year - df["Year"]
df.drop(columns=["Year"], inplace=True)
encoder = LabelEncoder()
for col in ["Fuel_Type", "Selling_type", "Transmission"]:
    df[col] = encoder.fit_transform(df[col])
X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print("\nFeature Importance:\n", importance)
plt.figure()
importance.plot(kind="bar")
plt.title("Feature Importance in Car Price Prediction")
plt.ylabel("Importance Score")
plt.show()