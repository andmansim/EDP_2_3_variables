
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import matplotlib.animation as animation
from scipy.interpolate import RegularGridInterpolator
import matplotlib.patches as mpatches
from matplotlib.widgets import CheckButtons 
from matplotlib.animation import FuncAnimation
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#===============================
# Graficar superficies 3D
#===============================
def graficar_superficies(intervalos, w_final, X, Y, Z, titulo):
    
    '''
    Función que grafica la solución de la ecuación de Laplace en 3D, con superficies coloreadas
    para cada intervalo de valores de la solución.
    '''

    # Crear la figura
    fig = go.Figure()

    # Añadir superficies para cada intervalo
    for idx, (val_min, val_max, color) in enumerate(intervalos):

        # Filtrar los valores que caen dentro del intervalo
        mask = (w_final >= val_min) & (w_final <= val_max) #Está por si hace falta, pero no es necesario, dado que lo hace solo con el iso min y max
        
        # Crear la superficie para este intervalo
        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=w_final.flatten(),
            isomin=val_min, #Aquí filtramos los valores que caen dentro del intervalo
            isomax=val_max,
            opacity=0.6,
            colorscale=[[0, color], [1, color]],  # Colores sólidos para cada intervalo
            showscale=False,
            showlegend=True,  # Para desactivar la leyenda de la superficie
            name=f"Intervalo {val_min:.2f} - {val_max:.2f}"
        ))

    fig.update_layout(title=f"{titulo}", autosize=True)
    
    #Guardar la figura como HTML
    fig.write_html(f"{titulo}.html")
    fig.show()
    

#===============================
# Graficar puntos 3D
#===============================
def graficar_puntos (intervalos, w_final, X, Y, Z, titulo):
    
    '''
    Función que grafica la solución de la ecuación de Laplace en 3D, con puntos coloreados
    para cada intervalo de valores de la solución.
    '''
    
    # Graficar la solución en 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar todos los puntos en gris como referencia
    ax.scatter(X, Y, Z, c='lightgray', alpha=0.05, s=5)

    # Iterar sobre cada intervalo y graficar los puntos en su respectivo color
    for val_min, val_max, color in intervalos:
        mask = (w_final >= val_min) & (w_final <= val_max) #filtrar los valores que caen dentro del intervalo
        X_filt, Y_filt, Z_filt = X[mask], Y[mask], Z[mask]
        ax.scatter(X_filt, Y_filt, Z_filt, c=color, label=f'{val_min} ≤ W ≤ {val_max}', s=20)

    # Leyenda y etiquetas
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Solución de la ecuación de Laplace con intervalos resaltados")
    
    # Guardar la figura como PNG
    plt.savefig(f'{titulo}.png')

    plt.show()
    

#===============================
#Graficos de evolución temporal
#===============================
def visualizar_evolucion(w,historial_w, x):
    '''
    Visualizamos los distintos estados de la cadena colgante en el tiempo. 
    Esta se representa como una serie de puntos en el plano (u(x,t), x).
    Coge los datos del hitorico y los va actualizando en el tiempo.
    '''
    fig, ax = plt.subplots(figsize=(10, 6))
    
    #set_xlim y set_ylim son para definir los limites de los ejes
    #set_xlim hace que el eje x vaya desde el minimo valor de w hasta el maximo
    ax.set_xlim(np.min([np.min(w) for w in historial_w.values()]), 
                np.max([np.max(w) for w in historial_w.values()]))  # Eje X es la altura u(x,t)
    
    #set_ylim hace que el eje y vaya desde el minimo valor de x hasta el maximo
    ax.set_ylim(min(x), max(x))  # Eje Y es la posición x
    ax.set_xlabel("Altura u(x,t)")
    ax.set_ylabel("Posición x")

    #creamos [], [] para que no haya puntos iniciales
    scatter, = ax.plot([], [], 'bo-', markersize=5)  #bo- es para que los puntos sean azules y con lineas

    # Ordenamos los tiempos
    tiempos = sorted(historial_w.keys())  
    
    
    # print(tiempos)
    titulo = ax.set_title("Evolución de la función en el tiempo")


    def actualizar(frame):
        t = tiempos[frame]  # Tomamos el tiempo actual
        print(t)
        
        titulo.set_text(f"Tiempo: {t}")  # Actualiza el título con el tiempo actual
        
        # Tomamos toda la columna de w en el tiempo t
        w_actual = historial_w[t][:, :]  # Ahora tomamos todos los puntos en el tiempo t
        # print(w_actual)
        
        # Graficamos todos los puntos (x, w(x,t))
        scatter.set_data(w_actual[:,t], x)  # X es u(x,t) y Y es x
        
        return scatter, titulo
    #interval es el tiempo entre cada frame en milisegundos. Si aumentamos va más lento
    #blit=True hace que solo se actualicen los puntos que cambian, no toda la figura
    ani = animation.FuncAnimation(fig, actualizar, frames=len(tiempos), interval=210, blit=False, repeat=False)
    plt.show()


#==============================================
# Graficar superficies temporales 2 variables
#==============================================
def graficar_superficies_temporales (X, Y, w_final,titulo, func, transf):
    '''
    Graficar superficies con dos variables independientes. Es decir, los ejes son x, y, u(x,y)
    '''
    
    #graficar
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, w_final, cmap='viridis') 

    #etiquetas
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{titulo}')
    
    plt.savefig(f'graficas_simetria/{titulo}.png')
    
    
    # Definimos el directorio, el nombre del archivo y la extensión
    nombre_archivo = f"{titulo}.png"
    directorio = f"C:/Users/andre/Desktop/Cosas Uni/simetria_fotos/funcion_{func}/transform{transf}"
    ruta_completa = os.path.join(directorio, nombre_archivo)
    plt.savefig(ruta_completa, dpi=300, bbox_inches='tight')
    
    #mostramos
    plt.show()


#==========================================================
# Graficar superficies temporales 2 variables superpuestas
#==========================================================
def graficar_matrices_superpuestas(X, T, u, X_prime, T_prime, u_prime, titulo, func, transf ,alpha1=0.6, alpha2=0.6):

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar u(x,t) con la colormap 'viridis' y opacidad alpha1
    # ax.plot_surface(X, T, u, cmap='viridis', alpha=alpha1, edgecolor='none') #azul
    ax.scatter(X, T, u,  cmap='viridis', s=1)
    
    # Graficar u'(x',t') con la colormap 'plasma' y opacidad alpha2
    ax.plot_surface(X_prime, T_prime, u_prime, cmap='plasma', alpha=alpha2, edgecolor='none') #rosa

    # Etiquetas y título
    ax.set_xlabel("X")
    ax.set_ylabel("T")
    ax.set_zlabel("Valores de u")

    ax.set_title(f'{titulo}')
    
    plt.savefig(f'graficas_simetria/{titulo}.png')
    
    # Definimos el directorio, el nombre del archivo y la extensión
    nombre_archivo = f"{titulo}.png"
    directorio = f"C:/Users/andre/Desktop/Cosas Uni/simetria_fotos/funcion_{func}/transform{transf}"
    ruta_completa = os.path.join(directorio, nombre_archivo)
    plt.savefig(ruta_completa, dpi=300, bbox_inches='tight')

    # Mostrar la gráfica
    plt.show()
    

# ==============================================
# Animación de comparación de la transformación 
# con la solución original transformada
# ==============================================
def anima_comparacion(soluciones, primas, titulo, epsilons, Nx, Ny, c, d, nombre1, nombre2):
    '''
    Función que anima la comparación entre la solución original transformada y la solución de la transformación 1.
    Se grafican ambas soluciones en la misma figura para cada valor de epsilon.
    Adaptando la malla de referencia para que se vea la comparación entre ambas soluciones.
    '''
    # Interpolar las soluciones originales y transformadas en la malla de referencia
    def make_interpolators(sol_raw):
        '''
        Función que crea interpoladores para las soluciones originales y transformadas.
        '''
        # Crear un diccionario de interpoladores para cada epsilon
        interps = {}
        for eps, (Xr, Yr, Wr) in sol_raw.items():
            # Interpolador para la solución original
            # Xr y Yr son las mallas de la solución original
            x1 = np.unique(Xr[:,0]); t1 = np.unique(Yr[0,:])
            
            # invertir el orden de los ejes si es necesario
            if x1[1] < x1[0]: x1, Wr = x1[::-1], Wr[::-1,:]
            if t1[1] < t1[0]: t1, Wr = t1[::-1], Wr[:,::-1] 
            
            interps[eps] = RegularGridInterpolator((x1, t1), Wr, method='nearest', bounds_error=False, fill_value=None)
        return interps
    
    # Animación
    def update(frame):
        
        # Limpiar y actualizar la figura
        ax.clear()
        ax.set_xlabel("x'"); ax.set_ylabel('t'); ax.set_zlabel('u')
        ax.set_xlim(x_ref_min, x_ref_max); ax.set_ylim(c, d); ax.set_zlim(zmin, zmax)
        ax.legend(handles=[patch1, patch2], loc='upper left')
        
        # Extraer el valor de epsilon para el frame actual
        # y graficar la superficie correspondiente
        eps = epsilons[frame]
        Z1, Z2 = sol1[eps], sol2[eps]
        ax.scatter(X_ref, Y_ref, Z1, c=Z1, cmap='viridis', s=10, marker='o', alpha=0.9)
        ax.plot_surface(X_ref, Y_ref, Z2, cmap='plasma', edgecolor='none', alpha=0.5)
        ax.set_title(f"ε = {eps:.2f}", pad=15, fontsize=12)
        return []
    
    
    # Estraer los parámetros necesarios para calcular la malla de referencia
    all_x1 = np.hstack([X1.ravel() for X1,_,_ in soluciones.values()]) #Valores de x' de la transformación 1
    all_x2 = np.hstack([X2.ravel() for X2,_,_ in primas.values()]) #Valores de x' de la solución original transformada
    x_ref_min, x_ref_max = min(all_x1.min(), all_x2.min()), max(all_x1.max(), all_x2.max()) #Estraemos max y min de cada

    # Crear la malla de referencia
    x_ref = np.linspace(x_ref_min, x_ref_max, Nx)
    t_ref = np.linspace(c, d, Ny)
    X_ref, Y_ref = np.meshgrid(x_ref, t_ref, indexing='ij')
    pts_ref = np.vstack([X_ref.ravel(), Y_ref.ravel()]).T 

    #interpolamos los valores de la solución original transformada y transformada
    interps1 = make_interpolators(soluciones)
    interps2 = make_interpolators(primas)

    ## Interpolamos las soluciones en la malla de referencia
    sol1 = {eps: interps1[eps](pts_ref).reshape(X_ref.shape) for eps in epsilons}
    sol2 = {eps: interps2[eps](pts_ref).reshape(X_ref.shape) for eps in epsilons}

    # Graficar la superficie de la solución original transformada y la transformada
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x'")
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    
    # Definir límites de x y t
    ax.set_xlim(x_ref_min, x_ref_max)
    ax.set_ylim(c, d)
    
    # Definir límites de z
    zmin = min(Z.min() for Z in list(sol1.values()) + list(sol2.values()))
    zmax = max(Z.max() for Z in list(sol1.values()) + list(sol2.values()))
    ax.set_zlim(zmin, zmax)
    
    # Leyenda para superficies
    patch1 = mpatches.Patch(color=plt.cm.viridis(0.6), label=nombre1)
    patch2 = mpatches.Patch(color=plt.cm.plasma(0.6), label= nombre2)
    ax.legend(handles=[patch1, patch2], loc='upper left')

    # Graficar una superficie inicial
    ax.scatter(X_ref, Y_ref, sol1[epsilons[0]], c=sol1[epsilons[0]], cmap='viridis', s=10, marker='o', alpha=0.9)
    ax.plot_surface(X_ref, Y_ref, sol2[epsilons[0]], cmap='plasma', edgecolor='none', alpha=0.5)

    
    # Animación
    ani = animation.FuncAnimation(fig, update, frames=len(epsilons), interval=500, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Guardar la animación como un archivo de video
    ani.save(f"graficas_simetria/{titulo}.gif", writer="pillow", fps=5)
    plt.show()



# ===========================
# Función para animar
# campos vectoriales (2D)
# ===========================
def animar_derivadas(soluciones, epsilons, derivadas_numericas, derivadas_teoricas, transf):
    
    """
    Genera una animación 2D comparando las derivadas numéricas y teóricas.
    """
    # Calcular los límites dinámicos
    X_min = min(np.min(soluciones[eps][0]) for eps in epsilons)
    X_max = max(np.max(soluciones[eps][0]) for eps in epsilons)
    T_min = min(np.min(soluciones[eps][1]) for eps in epsilons)
    T_max = max(np.max(soluciones[eps][1]) for eps in epsilons)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(X_min, X_max)
    ax.set_ylim(T_min, T_max)
    ax.set_xlabel("X")
    ax.set_ylabel("T")
    ax.set_title(f"Comparación de derivadas (Transformación {transf})")

    # Inicializar los campos vectoriales
    quiver_num = ax.quiver([], [], [], [], color='blue', label='Numérica')
    quiver_teo = ax.quiver([], [], [], [], color='red', label='Teórica')
    ax.legend()
    
    # Crear una caja de selección para activar/desactivar campos
    ax_check = plt.axes([0.8, 0.4, 0.15, 0.15])  # Posición de la caja
    check = CheckButtons(ax_check, ['Numérica', 'Teórica'], [True, True])

    # Variables para controlar la visibilidad
    show_num = [True]
    show_teo = [True]

    def toggle(label):
        if label == 'Numérica':
            show_num[0] = not show_num[0]
        elif label == 'Teórica':
            show_teo[0] = not show_teo[0]
        fig.canvas.draw_idle()

    check.on_clicked(toggle)

    def actualizar(frame):
        ax.clear()
        ax.set_xlim(X_min, X_max)
        ax.set_ylim(T_min, T_max)
        ax.set_xlabel("X")
        ax.set_ylabel("T")
        ax.set_title(f"Comparación de derivadas (Transformación {transf}) - Epsilon: {epsilons[frame]:.2f}")

        X, T, U = soluciones[epsilons[frame]]
        dX_num, dT_num, _ = derivadas_numericas[epsilons[frame]]
        dX_teo, dT_teo, _ = derivadas_teoricas[epsilons[frame]]

        # Verificar que las matrices no estén vacías ni contengan NaN
        if np.any(np.isnan(X)) or np.any(np.isnan(T)) or np.any(np.isnan(dX_num)) or np.any(np.isnan(dT_num)):
            print(f"Advertencia: Datos inválidos en el frame {frame}.")
            return
        
        step = 20
        scale_factor = 0.5
        X_sample = X[::step, ::step]
        T_sample = T[::step, ::step]
        dX_num_sample = dX_num[::step, ::step] * scale_factor
        dT_num_sample = dT_num[::step, ::step] * scale_factor
        dX_teo_sample = dX_teo[::step, ::step] * scale_factor
        dT_teo_sample = dT_teo[::step, ::step] * scale_factor
        
        # Dibujar la superficie como un mapa de colores
        ax.contourf(X, T, U, levels=50, cmap='viridis', alpha=0.6)  # Superficie suavizada

        # Dibujar campos vectoriales según la selección
        if show_num[0]:
            quiver_num = ax.quiver(X_sample, T_sample, dX_num_sample, dT_num_sample, color='blue', label='Numérica')
        if show_teo[0]:
            quiver_teo = ax.quiver(X_sample, T_sample, dX_teo_sample, dT_teo_sample, color='red', label='Teórica')

        ax.legend()

    ani = FuncAnimation(fig, actualizar, frames=len(epsilons) - 1, interval=500)
    #guardamos la animación
    ani.save(f'graficas_simetria/animacion_derivadas_{transf}.gif', writer='imagemagick', fps=5)
    plt.show()
    

# ===========================
# Función para dibujar flechas 
# con cono en 3D
# ===========================
def dibujar_flecha_con_cono(ax, origen, vector, color, largo_punta, radio_base, segmentos=8):
    """
    Dibuja una flecha con un cono en la punta.
    - origen: punto de inicio
    - vector: dirección de la flecha
    - largo_punta: largo del cono
    - radio_base: radio de la base del cono
    """
    
    # Vector normalizado
    v = np.array(vector)
    mag = np.linalg.norm(v)
    if mag == 0:
        return
    v_hat = v / mag

    # Punto final del vector
    punta = np.array(origen) + v

    # Punto base del cono (donde empieza la punta)
    base_centro = punta - largo_punta * v_hat

    # Generar círculo en la base
    theta = np.linspace(0, 2 * np.pi, segmentos)
    circle = np.array([np.cos(theta), np.sin(theta), np.zeros_like(theta)])  # base en XY

    # Transformar círculo al plano perpendicular al vector
    # Usamos vectores ortogonales
    def ortonormal_basis(v_hat):
        if np.allclose(v_hat, [0, 0, 1]):
            u = np.array([1, 0, 0])
        else:
            u = np.cross(v_hat, [0, 0, 1])
            u = u / np.linalg.norm(u)
        w = np.cross(v_hat, u)
        return u, w

    u, w = ortonormal_basis(v_hat)
    base_pts = base_centro[:, None] + radio_base * (u[:, None] * circle[0] + w[:, None] * circle[1])

    # Crear caras del cono
    verts = []
    for i in range(segmentos):
        verts.append([base_pts[:, i], base_pts[:, (i + 1) % segmentos], punta])

    # Dibujar cuerpo (línea)
    cuerpo_fin = base_centro
    ax.plot(
        [origen[0], cuerpo_fin[0]],
        [origen[1], cuerpo_fin[1]],
        [origen[2], cuerpo_fin[2]],
        color=color,
        linewidth=1.5
    )

    # Dibujar el cono
    cone = Poly3DCollection(verts, color=color, alpha=1.0)
    ax.add_collection3d(cone)
 
    
# ===========================
# Función para animar
# campos vectoriales (3D)
# ===========================
def animar_derivadas_3d(soluciones, epsilons, derivadas_numericas, derivadas_teoricas, transf,func, punta, base):
    """
    Genera una animación 3D comparando las derivadas numéricas y teóricas.
    """
    # Calcular los límites dinámicos
    X_min = min(np.min(soluciones[eps][0]) for eps in epsilons)
    X_max = max(np.max(soluciones[eps][0]) for eps in epsilons)
    T_min = min(np.min(soluciones[eps][1]) for eps in epsilons)
    T_max = max(np.max(soluciones[eps][1]) for eps in epsilons)
    U_min = min(np.min(soluciones[eps][2]) for eps in epsilons)
    U_max = max(np.max(soluciones[eps][2]) for eps in epsilons)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(X_min, X_max)
    ax.set_ylim(T_min, T_max)
    ax.set_zlim(U_min, U_max)
    ax.set_xlabel("X")
    ax.set_ylabel("T")
    ax.set_zlabel("U")
    ax.set_title(f"Comparación de derivadas (Transformación {transf})")

    def actualizar(frame):
        ax.clear()
        ax.set_xlim(X_min, X_max)
        ax.set_ylim(T_min, T_max)
        ax.set_zlim(U_min, U_max)
        ax.set_xlabel("X")
        ax.set_ylabel("T")
        ax.set_zlabel("U")
        ax.set_title(f"Comparación de derivadas (Transformación {transf}) - Epsilon: {epsilons[frame]:.2f}")

        # Extraer datos para el frame actual
        X, T, U = soluciones[epsilons[frame]]
        dX_num, dT_num, dU_num = derivadas_numericas[epsilons[frame]]
        dX_teo, dT_teo, dU_teo = derivadas_teoricas[epsilons[frame]]
        
        # Etiquetas fijas en una esquina
        ax.text2D(0.02, 0.95, "Numérica: azul", transform=ax.transAxes, color='blue', fontsize=10)
        ax.text2D(0.02, 0.90, "Teórica: rojo", transform=ax.transAxes, color='red', fontsize=10)

        # Reducir densidad y escalar vectores
        step = 20
        if transf == 2:
            X_sample = X[::step, ::step]
            T_sample = T[::step, ::step]
            U_sample = U[::step, ::step]
            
            # Normalizar los vectores a una longitud fija
            fixed_length = 0.4  # Longitud fija para todas las flechas
            # Limitar las magnitudes a un rango razonable
            min_magnitude = 0.5  # Mínimo valor permitido para las magnitudes
            max_magnitude = 7.0  # Máximo valor permitido para las magnitudes

            
            # Calcular las magnitudes de los vectores
            magnitudes_num = np.sqrt(dX_num[::step, ::step]**2 + dT_num[::step, ::step]**2 + dU_num[::step, ::step]**2)
            magnitudes_teo = np.sqrt(dX_teo[::step, ::step]**2 + dT_teo[::step, ::step]**2 + dU_teo[::step, ::step]**2)

            # Limitar las magnitudes al rango definido
            magnitudes_num = np.clip(magnitudes_num, min_magnitude, max_magnitude)
            magnitudes_teo = np.clip(magnitudes_teo, min_magnitude, max_magnitude)

            # Normalizar los vectores para que tengan una longitud fija
            fixed_length = 1.0
            dX_num_sample = (dX_num[::step, ::step] / magnitudes_num) * fixed_length
            dT_num_sample = (dT_num[::step, ::step] / magnitudes_num) * fixed_length
            dU_num_sample = (dU_num[::step, ::step] / magnitudes_num) * fixed_length

            dX_teo_sample = (dX_teo[::step, ::step] / magnitudes_teo) * fixed_length
            dT_teo_sample = (dT_teo[::step, ::step] / magnitudes_teo) * fixed_length
            dU_teo_sample = (dU_teo[::step, ::step] / magnitudes_teo) * fixed_length
            
             
        elif transf == 1:
            scale_factor = 0.5
            X_sample = X[::step, ::step]
            T_sample = T[::step, ::step]
            U_sample = U[::step, ::step]
            
            # Normalizar los vectores numéricos
            magnitudes_num = np.sqrt(dX_num[::step, ::step]**2 + dT_num[::step, ::step]**2 + dU_num[::step, ::step]**2)
            magnitudes_num[magnitudes_num == 0] = 1  # Evitar división por cero
            dX_num_sample = (dX_num[::step, ::step] / magnitudes_num) * scale_factor
            dT_num_sample = (dT_num[::step, ::step] / magnitudes_num) * scale_factor
            dU_num_sample = (dU_num[::step, ::step] / magnitudes_num) * scale_factor

            # Normalizar los vectores teóricos
            magnitudes_teo = np.sqrt(dX_teo[::step, ::step]**2 + dT_teo[::step, ::step]**2 + dU_teo[::step, ::step]**2)
            magnitudes_teo[magnitudes_teo == 0] = 1  # Evitar división por cero
            dX_teo_sample = (dX_teo[::step, ::step] / magnitudes_teo) * scale_factor
            dT_teo_sample = (dT_teo[::step, ::step] / magnitudes_teo) * scale_factor
            dU_teo_sample = (dU_teo[::step, ::step] / magnitudes_teo) * scale_factor
            
        
        if show_num[0]:
            #Creamos la cabeza de las flechas para que se vean mejor
            for i in range(X_sample.shape[0]):
                for j in range(X_sample.shape[1]):
                    origen = np.array([X_sample[i, j], T_sample[i, j], U_sample[i, j]])
                    vector = np.array([dX_num_sample[i, j], dT_num_sample[i, j], dU_num_sample[i, j]])
                    dibujar_flecha_con_cono(ax, origen, vector, 'blue', punta, base)
                    
        if show_teo[0]:
            #Creamos la cabeza de las flechas para que se vean mejor
            for i in range(X_sample.shape[0]):
                for j in range(X_sample.shape[1]):
                    origen = np.array([X_sample[i, j], T_sample[i, j], U_sample[i, j]])
                    vector = np.array([dX_teo_sample[i, j], dT_teo_sample[i, j], dU_teo_sample[i, j]])
                    dibujar_flecha_con_cono(ax, origen, vector, 'red', punta, base)
        
        # Dibujar la superficie
        ax.plot_surface(X, T, U, cmap='viridis', alpha=0.6)

        ax.legend()

    # Crear una caja de selección para activar/desactivar campos
    ax_check = plt.axes([0.8, 0.4, 0.15, 0.15])  # Posición de la caja
    check = CheckButtons(ax_check, ['Numérica', 'Teórica'], [True, True])

    # Variables para controlar la visibilidad
    show_num = [True]
    show_teo = [True]

    def toggle(label):
        if label == 'Numérica':
            show_num[0] = not show_num[0]
        elif label == 'Teórica':
            show_teo[0] = not show_teo[0]
        fig.canvas.draw_idle()

    check.on_clicked(toggle)
    
    
    ani = FuncAnimation(fig, actualizar, frames=len(epsilons) - 1, interval=500)
    # Guardamos la animación
    ani.save(f'graficas_simetria/animacion_derivadas_3d_{transf}_{func}.gif', writer='imagemagick', fps=5)
    plt.show()


# ===========================
# Función para graficar
# campos vectoriales (3D)
# ===========================
def graficar_derivadas_3d(soluciones, epsilon, derivadas_numericas, derivadas_teoricas, transf, func, punta, base):
    """
    Genera una gráfica 3D comparando las derivadas numéricas y teóricas para un único epsilon.
    """
    # Extraer datos para el epsilon especificado
    X, T, U = soluciones[epsilon]
    dX_num, dT_num, dU_num = derivadas_numericas[epsilon]
    dX_teo, dT_teo, dU_teo = derivadas_teoricas[epsilon]

    # Calcular los límites dinámicos
    X_min, X_max = np.min(X), np.max(X)
    T_min, T_max = np.min(T), np.max(T)
    U_min, U_max = np.min(U), np.max(U)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(X_min, X_max)
    ax.set_ylim(T_min, T_max)
    ax.set_zlim(U_min, U_max)
    ax.set_xlabel("X")
    ax.set_ylabel("T")
    ax.set_zlabel("U")
    ax.set_title(f"Comparación de derivadas (Transformación {transf}) - Epsilon: {epsilon:.2f}")
    
    # Etiquetas fijas en una esquina
    ax.text2D(0.02, 0.95, "Numérica: azul", transform=ax.transAxes, color='blue', fontsize=10)
    ax.text2D(0.02, 0.90, "Teórica: rojo", transform=ax.transAxes, color='red', fontsize=10)

    # Reducir densidad y escalar vectores
    step = 20
    X_sample = X[::step, ::step]
    T_sample = T[::step, ::step]
    U_sample = U[::step, ::step]

    if transf == 2:
        # Normalizar los vectores a una longitud fija
        fixed_length = 0.5
        magnitudes_num = np.sqrt(dX_num[::step, ::step]**2 + dT_num[::step, ::step]**2 + dU_num[::step, ::step]**2)
        magnitudes_teo = np.sqrt(dX_teo[::step, ::step]**2 + dT_teo[::step, ::step]**2 + dU_teo[::step, ::step]**2)

        magnitudes_num = np.clip(magnitudes_num, 0.5, 7.0)
        magnitudes_teo = np.clip(magnitudes_teo, 0.5, 7.0)

        dX_num_sample = (dX_num[::step, ::step] / magnitudes_num) * fixed_length
        dT_num_sample = (dT_num[::step, ::step] / magnitudes_num) * fixed_length
        dU_num_sample = (dU_num[::step, ::step] / magnitudes_num) * fixed_length

        dX_teo_sample = (dX_teo[::step, ::step] / magnitudes_teo) * fixed_length
        dT_teo_sample = (dT_teo[::step, ::step] / magnitudes_teo) * fixed_length
        dU_teo_sample = (dU_teo[::step, ::step] / magnitudes_teo) * fixed_length

    elif transf == 1:
        scale_factor = 0.5
        magnitudes_num = np.sqrt(dX_num[::step, ::step]**2 + dT_num[::step, ::step]**2 + dU_num[::step, ::step]**2)
        magnitudes_teo = np.sqrt(dX_teo[::step, ::step]**2 + dT_teo[::step, ::step]**2 + dU_teo[::step, ::step]**2)

        magnitudes_num[magnitudes_num == 0] = 1  # Evitar división por cero
        magnitudes_teo[magnitudes_teo == 0] = 1  # Evitar división por cero

        dX_num_sample = (dX_num[::step, ::step] / magnitudes_num) * scale_factor
        dT_num_sample = (dT_num[::step, ::step] / magnitudes_num) * scale_factor
        dU_num_sample = (dU_num[::step, ::step] / magnitudes_num) * scale_factor

        dX_teo_sample = (dX_teo[::step, ::step] / magnitudes_teo) * scale_factor
        dT_teo_sample = (dT_teo[::step, ::step] / magnitudes_teo) * scale_factor
        dU_teo_sample = (dU_teo[::step, ::step] / magnitudes_teo) * scale_factor

   
    for i in range(X_sample.shape[0]):
                for j in range(X_sample.shape[1]):
                    origen = np.array([X_sample[i, j], T_sample[i, j], U_sample[i, j]])
                    vector = np.array([dX_num_sample[i, j], dT_num_sample[i, j], dU_num_sample[i, j]])
                    dibujar_flecha_con_cono(ax, origen, vector, 'blue', punta, base)
   
    for i in range(X_sample.shape[0]):
                for j in range(X_sample.shape[1]):
                    origen = np.array([X_sample[i, j], T_sample[i, j], U_sample[i, j]])
                    vector = np.array([dX_teo_sample[i, j], dT_teo_sample[i, j], dU_teo_sample[i, j]])
                    dibujar_flecha_con_cono(ax, origen, vector, 'red', punta, base)

    # Dibujar la superficie
    ax.plot_surface(X, T, U, cmap='viridis', alpha=0.6)

    ax.legend()
    # Definimos el directorio, el nombre del archivo y la extensión
    nombre_archivo = f"derivadas_3d_{transf}_{func}_{epsilon}.png"
    directorio = f"C:/Users/andre/Desktop/Cosas Uni/simetria_fotos/funcion_{func}/transform{transf}"
    ruta_completa = os.path.join(directorio, nombre_archivo)
    plt.savefig(ruta_completa, dpi=300, bbox_inches='tight')
    
    plt.savefig(f'graficas_simetria/derivadas_3d_{transf}_{func}_{epsilon}.png', dpi=300, bbox_inches='tight')
    plt.show()