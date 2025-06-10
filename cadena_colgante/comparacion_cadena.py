import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe
from matplotlib import image as mpimg


#============================================
#VISUALIZAR AMBOS RESULTADOS EN UNA ANIMACIÓN
#============================================
def animar_comparacion(resultados_numericos, resultados_teoricos):
    '''
    Función encargada de crear la animación de comparación entre los resultados numéricos y teóricos.
    '''
    print(plt.style.available)
    # Aplicar un estilo base 
    plt.style.use('classic')

    #Cogemos el número de columnas y filas de los resultados
    num_cols = resultados_numericos.shape[1]
    num_filas = resultados_numericos.shape[0]

    # Crear la figura y los ejes. Dos ejes x que comparten el eje y
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=100)
    ax2 = ax1.twiny()  

    # Crear un eje adicional para el fondo
    ax_bg = fig.add_axes(ax1.get_position(), zorder=0)  # Mismo tamaño que ax1, zorder=0 para que esté detrás
    ax_bg.set_xlim(-5, 5)  # Fija los límites del fondo
    ax_bg.set_ylim(-2, num_filas + 2) # Fija los límites del fondo
    ax_bg.axis('off')  # Desactiva los ejes del fondo

    # Mostrar la imagen de fondo en el eje fijo
    img = mpimg.imread("cadena_colgante/graficas/fondo_habita.png")
    ax_bg.imshow(img, extent=[-5, 5, -2, num_filas + 2], aspect='auto', alpha=0.4) # Ajusta el alpha para la transparencia


    # Gráfica inicial para cada conjunto
    linea_num, = ax1.plot(resultados_numericos.iloc[:, 0], np.arange(num_filas), color='brown',
                    marker='o', markersize=8, linestyle='-', linewidth=2, label='Numericos')

    linea_teo, = ax2.plot(resultados_teoricos.iloc[:, 0], np.arange(num_filas), color='gray',
                    marker='o', markersize=8, linestyle='-', linewidth=2, label='Teoricos') 

    # Agregar efectos de sombra para dar sensación de profundidad. linewidth tamaño sombras
    linea_num.set_path_effects([pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])
    linea_teo.set_path_effects([pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])

    #Colores, estilos y marcadores para las líneas
    linea_num.set_color('#D2B48C')  # Beige (parecido al color de una cuerda natural)
    linea_num.set_linewidth(5)      # Grosor para que se vea más como una cuerda
    linea_num.set_marker('o')  # Marcadores circulares
    linea_num.set_markersize(6)
    linea_num.set_linestyle('-')  # Línea sólida para conectar los marcadores


    linea_teo.set_color('#808080')  # Gris metálico
    linea_teo.set_linewidth(5)      # Grosor para que se vea más como una cadena
    linea_teo.set_marker('|')  # Marcadores verticales
    linea_teo.set_markersize(8)
    linea_teo.set_linestyle('-')  # Línea sólida para conectar los marcadores


    # Configuración de etiquetas y títulos de los ejes
    ax1.set_xlabel('Solución Numerica (marrón)')
    ax1.set_ylabel('u(x,t)')
    ax2.set_xlabel('Solución Teorica (gris)')

    # Definir la fracción fija en la que estará el 0 en el eje x
    # (0.5 significa que el 0 estará en el centro del rango del eje x)
    punto_fijo = 0.5
    
    tiempo_text = fig.text(0.5, 1, '', ha='center', va='top', fontsize=14, fontweight='bold')
    
    def update(frame):
        '''
        Función encargada de actualizar los datos en la animación.
        En cada frame se actualizan los datos de ambos conjuntos de resultados.
        '''
        
        # Extraer los datos para el frame actual como arrays
        nuevos_numericos = resultados_numericos.iloc[:, frame].to_numpy()
        nuevos_teoricos = resultados_teoricos.iloc[:, frame].to_numpy()
        
        # Punto constante en el de índice 0.
        # Creamos una máscara para excluir ese punto.
        mask = np.arange(num_filas) != 0 
        
        
        # Calcular los efectos de "gravedad" o curvatura para cada conjunto
        gravedad_num = np.sin(np.linspace(0, np.pi, num_filas))[mask] * 0.5
        gravedad_teo = np.cos(np.linspace(0, np.pi, num_filas)) * 0.5

        # Actualizar los datos de la línea numérica excluyendo el punto constante
        linea_num.set_xdata(nuevos_numericos[mask])
        linea_num.set_ydata(np.arange(num_filas)[mask] + gravedad_num)
        
        # Actualizar la línea teórica 
        linea_teo.set_xdata(nuevos_teoricos)
        linea_teo.set_ydata(np.arange(num_filas) + gravedad_teo)
        
        # Ajustamos los límites de los ejes para la parte numérica y teórica
        min_num = nuevos_numericos[mask].min()
        max_num = nuevos_numericos[mask].max()
        min_teo = nuevos_teoricos.min()
        max_teo = nuevos_teoricos.max()
        
        ax1.set_xlim(min_num - 0.1, max_num + 0.1)
        ax2.set_xlim(min_teo - 0.1, max_teo + 0.1)
        
        # Recalibrar el eje para fijar el 0 en una posición determinada 
        rango_numerico = max(abs(min_num) / punto_fijo, max_num / (1 - punto_fijo))
        new_lim1 = (-punto_fijo * rango_numerico, (1 - punto_fijo) * rango_numerico)
        ax1.set_xlim(new_lim1)
        rango_teorico = max(abs(min_teo) / punto_fijo, max_teo / (1 - punto_fijo))
        new_lim2 = (-punto_fijo * rango_teorico, (1 - punto_fijo) * rango_teorico)
        ax2.set_xlim(new_lim2)
        
        # Actualizar el título con el número de frame actual
        tiempo_text.set_text(f"Tiempo: {frame}")
        
        return linea_num, linea_teo

    ani = FuncAnimation(fig, update, frames=range(num_cols), interval=200, blit=False, repeat=False)
    ani.save("cadena_colgante/graficas/comparacion/comparacion_cadena_num_teo.gif", writer="pillow", fps=5)
    plt.show()


#============================================
#GUARDAR VISUALIZACIÓN DE RESULTADOS
#============================================
def fotos_animacion(resultados_numericos, resultados_teoricos):
    '''
    Función encargada de guardar la comparación entre los resultados numéricos y teóricos.
    '''

    #Cogemos el número de columnas y filas de los resultados
    num_cols = resultados_numericos.shape[1]
    num_filas = resultados_numericos.shape[0]
    # Lista de tiempos (índices de columna) que quieres guardar
    tiempos_a_guardar = [0, 50,150, 225, 300]  # Ejemplo: inicio, cuartos y final

    for frame in tiempos_a_guardar:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax2 = ax.twiny()
        # Fondo
        ax_bg = fig.add_axes(ax.get_position(), zorder=0)
        ax_bg.set_xlim(-5, 5)
        ax_bg.set_ylim(-2, num_filas + 2)
        ax_bg.axis('off')
        img = mpimg.imread("cadena_colgante/graficas/fondo_habita.png")
        ax_bg.imshow(img, extent=[-5, 5, -2, num_filas + 2], aspect='auto', alpha=0.4)

        # Datos
        nuevos_numericos = resultados_numericos.iloc[:, frame].to_numpy()
        nuevos_teoricos = resultados_teoricos.iloc[:, frame].to_numpy()
        mask = np.arange(num_filas) != 0
        gravedad_num = np.sin(np.linspace(0, np.pi, num_filas))[mask] * 0.5
        gravedad_teo = np.cos(np.linspace(0, np.pi, num_filas)) * 0.5

        # Gráficas
        ax.plot(nuevos_numericos[mask], np.arange(num_filas)[mask] + gravedad_num, 
                color='#D2B48C', marker='o', markersize=6, linewidth=5, label='Numérica')
        ax2.plot(nuevos_teoricos, np.arange(num_filas) + gravedad_teo, 
                color='#808080', marker='|', markersize=8, linewidth=5, label='Teórica')

        ax.set_xlabel('Solución Numérica (marrón)')
        ax.set_ylabel('u(x,t)')
        ax2.set_xlabel('Solución Teórica (gris)')
        ax.set_title(f"Tiempo: {frame}", fontsize=12, fontweight='bold')

        # Ajuste de límites igual que en la animación
        min_num = nuevos_numericos[mask].min()
        max_num = nuevos_numericos[mask].max()
        min_teo = nuevos_teoricos.min()
        max_teo = nuevos_teoricos.max()
        punto_fijo = 0.5
        rango_numerico = max(abs(min_num) / punto_fijo, max_num / (1 - punto_fijo))
        new_lim1 = (-punto_fijo * rango_numerico, (1 - punto_fijo) * rango_numerico)
        ax.set_xlim(new_lim1)
        rango_teorico = max(abs(min_teo) / punto_fijo, max_teo / (1 - punto_fijo))
        new_lim2 = (-punto_fijo * rango_teorico, (1 - punto_fijo) * rango_teorico)
        ax2.set_xlim(new_lim2)

        plt.savefig(f"cadena_colgante/graficas/comparacion/comparacion_cadenas_tiempo_{frame}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    # Cargar los archivos de resultados
    resultados_teoricos = pd.read_csv('cadena_colgante/soluciones/sol_cadena_teorica.csv', header=None)
    resultados_numericos = pd.read_csv('cadena_colgante/soluciones/sol_cadena_numerica.csv', header=None)

    # =========================================
    # Graficar los resultados (mapa de colores)
    # =========================================
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(resultados_teoricos, aspect='auto', cmap='viridis')
    plt.title("Solución Teórica")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(resultados_numericos, aspect='auto', cmap='viridis')
    plt.title("Solución Numérica")
    plt.colorbar()

    plt.savefig("cadena_colgante/graficas/comparacion/mapa_calor_cad_teo_cad_num.png", dpi=300)
    plt.show()


    # =========================================
    #Visualizar rango de oscilaciones
    # =========================================
    plt.plot(resultados_numericos, np.arange(len(resultados_numericos)), 'ro-', markersize=5)  # Invertir ejes
    plt.plot(resultados_teoricos, np.arange(len(resultados_teoricos)), 'bo-', markersize=5)  # Invertir ejes
    
    # Leyenda manual con puntos de color (dentro del área del gráfico)
    plt.text(0.70, 0.10, '● Solución numérica', color='red', transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.70, 0.05, '● Solución teórica', color='blue', transform=plt.gca().transAxes, fontsize=10)
    
    plt.xlabel("Altura u(x,t)")  # Etiqueta eje x
    plt.ylabel("Posición x")  # Etiqueta eje y
    plt.gca().invert_yaxis()  # Para que la cadena cuelgue desde arriba
    plt.grid()  # Mostrar cuadrícula
    plt.savefig("cadena_colgante/graficas/comparacion/comparacion_oscilaciones_cadenas.png", dpi=300)  # Guardar la figura
    plt.show()
    
    # Guardar imágenes de la animación
    fotos_animacion(resultados_numericos, resultados_teoricos)
    
    #Visualizar comparación entre resultados numéricos y teóricos
    animar_comparacion(resultados_numericos, resultados_teoricos)
    
    
    
