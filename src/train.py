from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train, n_estimators: int = 100, max_depth: int = 5):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model
