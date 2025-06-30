import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("cam2_tracking.csv")
unique_ids = df["id"].nunique()
print(f"Personas únicas detectadas: {unique_ids}")


for pid in df["id"].unique():
    person = df[df["id"] == pid]
    plt.plot(person["x"], person["y"], label=f"ID {pid}")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trayectorias de personas")
plt.legend()
plt.gca().invert_yaxis()  # si quieres que coincida con coordenadas de video
plt.show()
