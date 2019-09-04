import matplotlib.pyplot as plt
import numpy as np


def print_uq_at_error(model, estimator, x_val, y_val, x_train, y_train):
    estimations = estimator.estimate(x_val, x_train, y_train)
    predictions = model(x_val).cpu().numpy()
    errors = np.abs(predictions-y_val)/(predictions+y_val)
    plt.figure(figsize=(12, 9))
    plt.ylabel('Uncertainty')
    plt.xlabel('Error')
    plt.scatter(errors, estimations)
    plt.show()
