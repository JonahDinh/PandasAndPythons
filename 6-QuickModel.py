import keras
import numpy as np

my_model = keras.models.load_model("Models/wine.keras")

md_wine = np.array([[7.5, 0.71, 0, 1.8, 0.76, 10.88, 34.2, 0.00, 3.5, 0.55, 9.5]], dtype=np.float64)
y_pred = my_model.predict(md_wine)
print(y_pred)
