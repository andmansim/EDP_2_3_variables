import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_gradual_color_extreme(point, center_point, cmap):
    # Calcular la distancia Euclidiana de un punto al punto extremo (0,0,8)
    dist = np.linalg.norm(np.array(point) - np.array(center_point))
    
    # Normalizar la distancia entre 0 y 1, donde 0 es el punto más cercano (0,0,8)
    norm_val = dist / np.linalg.norm(np.array([8, 8, 8]))  # Usamos la distancia máxima posible como 8,8,8
    
    # Invertir la normalización para que los puntos más cercanos (0,0,8) sean más brillantes
    norm_val = np.clip(norm_val, 0, 1)
    
    # Obtener el color desde el colormap
    color = cmap(norm_val)
    return color

if __name__ == "__main__":
    # Crear una malla 3D con solo 5 puntos en cada dimensión
    n = 6  # número de puntos en cada dimensión
    x = np.linspace(0, 8, n)
    y = np.linspace(0, 8, n)
    z = np.linspace(0, 8, n)

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
    boundary_color = 'purple' # color para los puntos en los bordes (morado)
    interior_color = 'g' # color para los puntos interiores (verde)


    boundary_cmap = cm.Purples    # Para los bordes
    interior_cmap = cm.BuGn  # Para los puntos interiores
    vertex_cmap = cm.YlOrRd   # Colormap para los vértices

    # Identificar los vértices (puntos extremos) y la cara superior (cara Z = L)
    vertices = [(x[0], y[0], z[0]), (x[-1], y[0], z[0]), (x[0], y[-1], z[0]), (x[-1], y[-1], z[0])]

    # Identificar los puntos de los bordes (pero no vértices)
    boundary_points = []

    for i in range(n):
        
        puntos_boundary = [(x[i], y[0], z[0]), # borde en x cuando y = c y z = e
                            (x[i], y[-1], z[0]), # borde en x cuando y = d y z = e
                            (x[0], y[i], z[0]),  # borde en y cuando x = a y z = e
                            (x[-1], y[i], z[0]), # borde en y cuando x = b y z = e
                            (x[0], y[0], z[i]),  # borde en z cuando x = a y y = c
                            (x[-1], y[0], z[i]), # borde en z cuando x = b y y = c
                            (x[0], y[-1], z[i]),  # borde en z cuando x = a y y = d
                            (x[-1], y[-1], z[i])]  # borde en z cuando x = b y y = d
        for punto in puntos_boundary:
            if punto not in vertices:
                boundary_points.append(punto)
        

    # Los puntos interiores son los puntos que no son ni vértices ni bordes
    interior_points = [(x_points[i], y_points[i], z_points[i])
                    for i in range(len(x_points))
                    if (x_points[i], y_points[i], z_points[i]) not in vertices and
                    (x_points[i], y_points[i], z_points[i]) not in boundary_points]

    # Obtener el valor mínimo y máximo de la distancia Z para la normalización del color
    z_min = np.min(z_points)
    z_max = np.max(z_points)

    extreme_point = (0,0,8)

    for vertex in vertices:
        color = get_gradual_color_extreme(vertex, extreme_point, vertex_cmap)  # Gradiente basado en la distancia al extremo
        ax.scatter(vertex[0], vertex[1], vertex[2], color=color, s=50)

    # Dibujar los puntos de los bordes (morado) con gradiente de color basado en la distancia al extremo
    for point in boundary_points:
        color = get_gradual_color_extreme(point, extreme_point, boundary_cmap)  # Gradiente basado en la distancia al extremo
        ax.scatter(point[0], point[1], point[2], color=color, s=30)

    # Dibujar los puntos interiores (verdes) con gradiente de color basado en la distancia al extremo
    for point in interior_points:
        color = get_gradual_color_extreme(point, extreme_point, interior_cmap)  # Gradiente basado en la distancia al extremo
        ax.scatter(point[0], point[1], point[2], color=color, s=10)


    # Ajustar la vista y distancia de la cámara para evitar solapamiento
    ax.view_init(elev=20, azim=20)  # Mantén el ángulo de vista
    ax.dist = 9  # Ajusta la distancia de la cámara

    # Etiquetas y título
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Malla abierta en el eje Z')

    # Mostrar la leyenda solo una vez por tipo
    ax.scatter([], [], [], color=vertex_color, label="Vértices")
    ax.scatter([], [], [], color=boundary_color, label="Extremos")
    ax.scatter([], [], [], color=interior_color, label="Interiores")

    # Mostrar la leyenda
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Guardar la imagen en un archivo PNG
    plt.savefig('mallas/malla_abierta_z.png', bbox_inches='tight')  # 'tight' ajusta los márgenes automáticamente

    plt.show()
