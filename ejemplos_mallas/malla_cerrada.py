import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


#========================
# Función color gradual
#========================
def color_gradual(point, val_min, val_max, cmap):
    '''
    Función que calcula un color gradual basado en la distancia de un punto al origen (0, 0, 0).
    point: Tupla (x, y, z) del punto para el cual se calculará el color.
    val_min: Valor mínimo de la distancia para normalizar el color.
    val_max: Valor máximo de la distancia para normalizar el color.
    cmap: Colormap de matplotlib que se utilizará para mapear la distancia a un color.
    '''
    # Calcular la distancia Euclidiana de un punto al origen (0, 0, 0)
    dist = np.linalg.norm(point)  
    
    # Normalizar la distancia entre 0 y 1
    norm_val = (dist - val_min) / (val_max - val_min)
    
    # Limitar el valor para que no llegue a 1 (evitar colores cercanos al blanco)
    norm_val = np.clip(norm_val, 0, 0.9)
    
    # Obtener el color desde el colormap
    color = cmap(norm_val)
    return color

#======
# Main  
#======
if __name__ == "__main__":
    '''
    Muestra una malla 3D cerrada en el eje Z con puntos de vértices, bordes e interiores.
    Los puntos de vértices son rojos, los puntos en los bordes son azules y los puntos interiores son verdes.
    Los colores de los puntos se determinan por un gradiente basado en la distancia al origen (0, 0, 0).
    '''
        
    # Crear una malla 3D con solo 5 puntos en cada dimensión
    n = 5  # número de puntos en cada dimensión
    x = np.linspace(0, 7, n)
    y = np.linspace(0, 7, n)
    z = np.linspace(0, 7, n)

    # Crear las coordenadas de la malla 3D
    X, Y, Z = np.meshgrid(x, y, z)

    # Aplanar las matrices para obtener las coordenadas de los puntos
    x_points = X.flatten()
    y_points = Y.flatten()
    z_points = Z.flatten()

    # Crear la figura y el eje 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Definir colores para cada tipo de punto
    vertex_color = 'r'   # color para los vértices (rojo)
    boundary_color = 'b' # color para los puntos en los bordes (azul)
    interior_color = 'g' # color para los puntos interiores (verde)

    boundary_cmap = cm.Blues  # Para los bordes
    interior_cmap = cm.Greens  # Para los puntos interiores
    vertex_cmap = cm.Reds  # Colormap para los vértices


    # Identificar los vértices (puntos extremos)
    # Son los puntos en los extremos de la malla
    vertices = [(x[0], y[0], z[0]), (x[-1], y[0], z[0]), (x[0], y[-1], z[0]), (x[-1], y[-1], z[0]),
                (x[0], y[0], z[-1]), (x[-1], y[0], z[-1]), (x[0], y[-1], z[-1]), (x[-1], y[-1], z[-1])]

    # Identificar los puntos de los bordes (pero no vértices)
    boundary_points = []

    for i in range(1, n-1):
        # Bordes en las aristas horizontales en el eje Z = 0
        boundary_points.append((x[i], y[0], z[0]))   # borde en x cuando y = c y z = e
        boundary_points.append((x[i], y[-1], z[0]))  # borde en x cuando y = d y z = e
        boundary_points.append((x[0], y[i], z[0]))   # borde en y cuando x = a y z = e
        boundary_points.append((x[-1], y[i], z[0]))  # borde en y cuando x = b y z = e
        
        # Bordes en las aristas verticales en el eje Z = L
        boundary_points.append((x[i], y[0], z[-1]))  # borde en x cuando y = c y z = f
        boundary_points.append((x[i], y[-1], z[-1])) # borde en x cuando y = d y z = f
        boundary_points.append((x[0], y[i], z[-1]))  # borde en y cuando x = a y z = f
        boundary_points.append((x[-1], y[i], z[-1]))  # borde en y cuando x = b y z = f
        
        # Bordes en las aristas verticales en el eje Z
        boundary_points.append((x[0], y[0], z[i]))   # borde en x = a y y = c
        boundary_points.append((x[-1], y[0], z[i]))  # borde en x = b y y = c
        boundary_points.append((x[0], y[-1], z[i]))   # borde en x = a y y = d
        boundary_points.append((x[-1], y[-1], z[i]))  # borde en x = b y y = d

    # Los puntos interiores son los puntos que no son ni vértices ni bordes
    interior_points = [(x_points[i], y_points[i], z_points[i])
                    for i in range(len(x_points))
                    if (x_points[i], y_points[i], z_points[i]) not in vertices and
                    (x_points[i], y_points[i], z_points[i]) not in boundary_points]



    # Obtener el valor mínimo y máximo de la distancia Euclidiana para la normalización del color
    distances = [np.linalg.norm((x_points[i], y_points[i], z_points[i])) for i in range(len(x_points))]
    dist_min = np.min(distances)
    dist_max = np.max(distances)

    # Dibujar los vértices (rojos) con un color fijo

    for vertex in vertices:
        color = color_gradual(vertex, dist_min, dist_max, vertex_cmap)  # Gradiente basado en la distancia
        ax.scatter(vertex[0], vertex[1], vertex[2], color=color, s=50)


    # Dibujar los puntos de los bordes (azules) con gradiente de color basado en la distancia
    for point in boundary_points:
        color = color_gradual(point, dist_min, dist_max, boundary_cmap)  # Gradiente basado en la distancia
        ax.scatter(point[0], point[1], point[2], color=color, s=30)


    # Dibujar los puntos interiores (verdes) con gradiente de color basado en la distancia
    for point in interior_points:
        color = color_gradual(point, dist_min, dist_max, interior_cmap)  # Gradiente basado en la distancia
        ax.scatter(point[0], point[1], point[2], color=color, s=10)


    ax.view_init(elev=30, azim=35)  # Mantén el ángulo de vista
    ax.dist = 9  # Ajusta la distancia de la cámara

    # Etiquetas y título
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Malla cerrada en tres variables')

    # Mostrar la leyenda solo una vez por tipo
    ax.scatter([], [], [], color=vertex_color, label="Vértices")
    ax.scatter([], [], [], color=boundary_color, label="Extremos")
    ax.scatter([], [], [], color=interior_color, label="Interiores")

    # Mostrar la leyenda
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Guardar la imagen en un archivo PNG
    plt.savefig('ejemplos_mallas/graficas/malla_cerrada.png', bbox_inches='tight')  # 'tight' ajusta los márgenes automáticamente

    plt.show()


