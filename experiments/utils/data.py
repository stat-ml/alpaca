from sklearn.preprocessing import StandardScaler


def scale(train, val):
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    val = scaler.transform(val)
    return train, val, scaler
