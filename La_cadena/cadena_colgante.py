#Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


# ===========================
# Función inicial
# ===========================
def funcion_f(x):
    return np.exp(-0.5*x) * np.sin(np.pi * x) 
    
# ===========================
# Velocidad inicial
# ===========================
def funcion_v(x):
    return 0
    
# ===========================
# Crear matriz inicial
# ===========================
def inicializar_matriz(Nx, Ny, a, c, h, k):
    #Matriz
    x = np.linspace(a,b,Nx+1)
    y = np.linspace(c,d,Ny+1)
    X,Y = np.meshgrid(x,y, indexing='ij')
    w = np.zeros((Nx+1,Ny+1))
    
    #condiciones iniciales
    for i in range(1, Nx):
        w[i, 0] = funcion_f(a + i*h) #posición inicial u(x,0) = f(x)
        
        #ut(i, t = 0) = v(x), ut(x,t) = u(x_i, t_t+1) - u(x_i, t_j) / k
        w[i,1] = k * funcion_v(a + i*h) + w[i,0] 

    for j in range(1, Ny):
        w[Nx,j] = 0 #punto fijo u(L, t) = 0
    
    return w, X, Y, x, y
   
   
# ======================================
# Verificar parámetros de discretización
# ======================================
def verificar_discretizacion (edp_parte1,edp_parte2, i, j):
    
    if np.isnan(edp_parte1) or np.isinf(edp_parte1) or abs(edp_parte1) > 1e10:
        print(f"Valores no válidos detectados en pri: {edp_parte1}, i={i}, j={j}")
        return False
            
    if abs(edp_parte2) < 1e-10:
        print(f"Posible inestabilidad en seg = {edp_parte2} en i={i}, j={j}")
        return False
    return True


# ============================
# Método de evolución temporal
# ============================
def met_evolucion(w,a, Nx, Ny, h, k, g):
    
    historial_w = {}
    #Guardamos el estado inicial
    historial_w[0] = np.copy(w)

    for j in range(1, Ny): #Al ser un método de evolución debería de empezar en 0. 
        #Pero como tenemos un w[i,j-1], si empieza en 0 tomará el último valor de j-1 en vez del anterior. Por eso lo movemos a 1.
        for i in range(1, Nx):

            #ux = regresivo
            pri = w[i,j] * (2*h**2 - 2*g*(a+i*h)*k**2 + g*h*k**2) + w[i-1,j] * (g*(a+i*h)*k**2 - g*h*k**2) + w[i+1,j] * (g*(a+i*h)*k**2)+ w[i,j-1] * (-h**2)
            seg = h**2
            
            if not verificar_discretizacion(pri, seg, i, j):
                break
            else:
                w[i,j+1] = pri / seg
        
        historial_w[j] = np.copy(w)
        
    return w, historial_w 
    

# ============================
# Visualizar evolución
# ============================
def visualizar_evolucion(w,historial_w, x):
    '''
    Visualizamos los distintos estados de la cadena colgante en el tiempo. 
    Esta se representa como una serie de puntos en el plano (u(x,t), x).
    Coge los datos del histórico y los va actualizando en el tiempo.
    '''
    fig, ax = plt.subplots(figsize=(10, 6))
    
    #set_xlim y set_ylim son para definir los limites de los ejes
    #set_xlim hace que el eje x vaya desde el mínimo valor de w hasta el máximo
    ax.set_xlim(np.min([np.min(w) for w in historial_w.values()]), 
                np.max([np.max(w) for w in historial_w.values()]))  # Eje X es la altura u(x,t)
    
    #set_ylim hace que el eje y vaya desde el mínimo valor de x hasta el máximo
    ax.set_ylim(min(x), max(x))  # Eje Y es la posición x
    ax.set_xlabel("Altura u(x,t)")
    ax.set_ylabel("Posición x")
    titulo = ax.set_title("Evolución de la función en el tiempo")

    #creamos [], [] para que no haya puntos iniciales
    scatter, = ax.plot([], [], 'bo-', markersize=5)  #bo- es para que los puntos sean azules y con lineas

    # Ordenamos los tiempos
    tiempos = sorted(historial_w.keys())  
    

    def actualizar(frame):
        t = tiempos[frame]  # Tomamos el tiempo actual
        
        titulo.set_text(f"Tiempo: {t}")  # Actualiza el título con el tiempo actual
        
        # Tomamos toda la columna de w en el tiempo t
        w_actual = historial_w[t][:, :]  
        
        # Graficamos todos los puntos (x, w(x,t))
        scatter.set_data(w_actual[1:,t], x[1:])  # X es u(x,t) y Y es x
        
        return scatter, titulo
    #interval es el tiempo entre cada frame en milisegundos. Si aumentamos va más lento
    #blit=True hace que solo se actualicen los puntos que cambian, no toda la figura
    ani = animation.FuncAnimation(fig, actualizar, frames=len(tiempos), interval=210, blit=False, repeat=False)
    ani.save("cadena_colgante/graficas_cadena/animacion_cadena.gif", writer="pillow", fps=5)
    plt.show()
    
#==============================
#Función para guardar imagenes
#==============================
def guardar_imagen(x,y,i):
    '''
    Guardamos la imagen en el directorio especificado con el nombre especificado.
    '''
    # Definimos el directorio, el nombre del archivo y la extensión
    nombre_archivo = "cadena_colgante_paso_{}.png".format(i)
    directorio = "C:/Users/andre/Desktop/Cosas Uni/cadena_fotos/cadena_colgante"
    ruta_completa = os.path.join(directorio, nombre_archivo)

    # w_i = x[:, i]  # Tomamos la columna i de w
    w_i = x[1:, i]  # Excluimos el primer elemento (índice 0) de la columna i de w
    y_excluido = y[1:]  # Excluimos el primer elemento de y
    
    # Creamos la figura y el eje
    imagen, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(np.min([np.min(v) for v in x]), 
                np.max([np.max(v) for v in x]))
    ax.set_ylim(min(y_excluido), max(y_excluido))  # Eje Y es la posición x
    # pintamos la cadena colgante
    ax.plot(w_i, y_excluido, color='blue', marker='o', markersize=5, label='Cadena Colgante')
    ax.set_xlabel("Altura u(x,t)")
    ax.set_ylabel("Posición x")
    ax.set_title("Cadena colgante en el Tiempo {}".format(i))
    
    # Guardamos la figura
    plt.savefig(ruta_completa, dpi=300, bbox_inches='tight')
    plt.close(imagen)  # Cierra la figura para liberar memoria



#=================
#Código principal
#=================
if __name__ == "__main__":
    '''
    Ecuación de la cadena colgante en 2D. 
    Se resuelvela edp utt = g * (x * uxx + ux). 
    x = [0,3], 1 > t > 0
    u(x,0) = f(x), f(x) = exp(-0.5*x) * sin(pi * x)
    u(x_f,t) = 0
    ut(x,t=0) = v(x), v(x) = 0
    Nx = 50
    Ny = 300
    Visualizamos la evolución de la cadena colgante en el tiempo.
    Guardamos la solución en un archivo .csv y la animación en un archivo .gif.
    '''
    #Definir parámetros
    a = c = 0
    b = 3
    d = 1
    g = 9.8

    Nx = 50
    Ny = 300

    h = (b-a)/Nx 
    k = (d-c)/Ny 
    print(k, h)
    
    #Verificamos la estabilidad de la discretización
    if (k**2 / h**2) > 1 / g: #1/g = 0.102
        
        #Si no es estable no lanzamos la simulación
        print(k, k**2, h, h**2, 1/g)
        print(k**2 / h**2)
        print(f"La relación de estabilidad no se cumple. Ajusta `k` o `h`.")
        
    else: #Si es estable lanzamos la simulación
        w, X, Y, x, y = inicializar_matriz(Nx, Ny, a, c, h, k)
              
        # Ejecutar la evolución y visualizar
        w_final, historico = met_evolucion(w, a, Nx, Ny, h, k, g)
          
        #contamos cuantos valores son distintos de 0
        print('Valores distintos de 0:', np.count_nonzero(w_final))
        print('Longitud de la matriz:', len(w_final))
        print('Valor máximo:', np.max(w_final))
        print('Valor mínimo:', np.min(w_final))

        #visualizamos la evolución
        visualizar_evolucion(w,historico, x)
        
        #guardar resultados en un archivo .csv
        np.savetxt("cadena_colgante/solucion_cadena/w_cadena_colgante.csv", w_final, delimiter=",")
        
        guardar_imagen(w_final,x,0) #Guardamos la imagen del primer paso
        guardar_imagen(w_final,x,int((Ny)/3)) #guardamos la imagen en un carto del progreso
        guardar_imagen(w_final,x,int((Ny)*2/3)) #guardamos la imagen en dos carto del progreso
        guardar_imagen(w_final,x,Ny) #Guardamos la imagen del ultimo paso
        