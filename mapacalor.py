import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib

# Leer datos
df = pd.read_csv("cam2_heatmap.csv")
df.columns = df.columns.str.strip()
df['time'] = pd.to_datetime(df['time'])
df['minute'] = df['time'].dt.strftime('%H:%M')

# Ordenar minutos únicos para el slider
time_steps = sorted(df['minute'].unique())

# Crear figura y eje
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25)  # dejar espacio para slider

# Función para actualizar heatmap según el tiempo seleccionado
def update_heatmap(minute):
    ax.clear()
    subset = df[df['minute'] == minute]
    if subset.empty:
        ax.text(0.5, 0.5, 'No hay datos en este minuto', ha='center', va='center', fontsize=14)
    else:
        sns.kdeplot(x=subset['x'], y=subset['y'], fill=True, cmap="hot", bw_adjust=0.5, levels=100, thresh=0.05, ax=ax)
    ax.set_title(f"Mapa de calor para {minute}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.canvas.draw_idle()

# Dibujar heatmap inicial
update_heatmap(time_steps[0])

# Crear slider
ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])  # posición del slider
slider = Slider(ax_slider, 'Tiempo (HH:MM)', 0, len(time_steps)-1, valinit=0, valstep=1)

# Evento del slider
def on_slider_change(val):
    minute = time_steps[int(val)]
    update_heatmap(minute)

slider.on_changed(on_slider_change)

plt.show()
