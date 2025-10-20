import numpy as np
import matplotlib.pyplot as plt


def computeThetaLSE(x, y, order=1, lmbd=0):
    """
    x - vektor vstupnich hodnot (doba studia)
    y - vektor vystupnich  hodnot (zisk bodu)
    order - rad polynomu
    lmbd - regularizacni parametr lambda

    """
    print(x.shape, y.shape)
    # Sestavení rozšiřené matice X
    # X = np.hstack([np.ones((x.shape[0], 1)), x])
    X = np.hstack([x**i for i in range(order + 1)])
    rows, cols = X.shape
    theta = np.dot(
        (np.dot(np.linalg.inv(np.dot(X.T, X) + lmbd * np.eye(cols)), X.T)), y
    )

    return theta


computeThetaLSE(np.array([[1], [2], [3]]), np.array([[5], [8], [11]]))

# Předpokládaná data
x_exp = np.array([[1], [2], [3]])
y_exp = np.array([[2.7], [7.4], [20.1]])

# Výpočet parametrů lineární regrese pro log(y)
theta = computeThetaLSE(x_exp, np.log(y_exp))

# Převod na exponenciální parametry
a = np.exp(theta[0])
b = theta[1]
print(f"a = {a}, b = {b}")

# Predikce exponenciální regrese
y_pred = a * np.exp(b * x_exp)

# Vykreslení
plt.scatter(x_exp, y_exp, color="blue", label="Data")
plt.plot(x_exp, y_pred, color="red", label="Exponenciální regrese")
plt.xlabel("Doba studia [hod]")
plt.ylabel("Výsledek [body]")
plt.title("Exponenciální regrese")
plt.legend()
plt.show()
