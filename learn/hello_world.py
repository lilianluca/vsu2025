import numpy as np
from sklearn.linear_model import LinearRegression

# 1. PŘÍPRAVA DAT
# X = Vstupní data (velikost bytu v m2)
# Scikit-learn vyžaduje, aby X byla matice (2D pole), proto používáme reshape(-1, 1)
X = np.array([20, 30, 40, 50, 60]).reshape(-1, 1)

# y = Výstupní data (cena v milionech) - to, co chceme předpovídat
y = np.array([40, 60, 80, 100, 120])

# 2. VYTVOŘENÍ MODELU
# Inicializujeme prázdný model lineární regrese
model = LinearRegression()

# 3. TRÉNOVÁNÍ (FIT)
# Tady probíhá "učení". Model hledá vztah mezi X a y.
# Hledá rovnici přímky y = wx + b, která nejlépe pasuje na data.
model.fit(X, y)

# 4. PŘEDPOVĚĎ (PREDICT)
# Zkusíme předpovědět cenu pro byt o velikosti 85 m2
novy_byt = np.array([[21]])
predikovana_cena = model.predict(novy_byt)

print(f"Předpokládaná cena pro 21 m2 je: {predikovana_cena[0]:.1f} milionů")

# --- Co se model ve skutečnosti naučil? ---
print(f"Váha (w): {model.coef_[0]}")  # Mělo by být 2
print(f"Bias (b): {model.intercept_}")  # Mělo by být blízko 0
