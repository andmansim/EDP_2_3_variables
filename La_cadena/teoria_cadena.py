
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import j0, j1
from scipy.special import jn_zeros
import os

#==========================
# Función inicial f(x)
# utilizada en el calculo de a_n
#==========================
def f(x):
    '''
    Definimos la función f(x) como el estado inicial de la cadena colgante
    '''
    return np.exp(-0.5*x) * np.sin(np.pi * x)
   

#======================================
# Método de Simpson compuesto
# para resolver la integral de los a_n
#======================================

def funcion_g(x, L, j):
    '''
    Definimos la función g(x) como el término de la integral de los a_n
    g(x) = f(x) * J0(jn * sqrt(x/L))
    '''
    return f(x) * j0(j * np.sqrt(x / L))

def simpson_compuesto(L, Nx, j):
    '''
    Aplicamos el método de Simpson compuesto para calcular la integral de 0 a L de g(x) dx
    g(x) = f(x) * J0(jn * sqrt(x/L))
    '''
    if Nx % 2 != 0:
        raise ValueError("El número de subintervalos Nx debe ser par.")
    
    else:
        h = L / Nx
        #creamos una array con los valores de x
        x = np.linspace(0, L, Nx + 1)
        #creamos una array con los valores de g(x)
        g_x = funcion_g(x, L, j) #Como se le pasa la array x, esta ya tiene incluido x_i = i*h
        
        #guardamos el primer y último valor de g(x)
        #S es una array que contiene los valores en los extremos y cada valor de g(x) dependiendo de si es par o impar. 
        S = g_x[0] + g_x[-1]

        for i in range(1, Nx):
            if i % 2 == 1:  # Índices impares
                S += 4 * g_x[i]
            
            else:  # Índices pares
                S += 2 * g_x[i]

        return (h / 3) * S
        
        
#===============================
#Cálculo de los coeficientes a_n
#===============================
def coeficientes_a_n(L, Nx, j_values):
    '''
    Calculamos los coeficientes a_n usando la integral de 0 a L de f(x) * J0(j_m * sqrt(x/L))
    a_n = 1/(L*(J1(j_m))^2) * integral. 
    Son los 5 primeros coeficientes a_n, por lo que j_m = j_1, j_2, j_3, j_4, j_5
    '''
    resultado = []
    for j_m in j_values:
        #Para cada valor de j_m, calculamos el coeficiente a_n
        #calculamos los valores de la integral de 0 a L de f(x) * J0(j_m * sqrt(x/L))
        integral_val = simpson_compuesto(L, Nx, j_m)

        #Calculamos el coeficiente a_n = 1/(L*(J1(j_m))^2) * integral
        a_n = integral_val / (L * (j1(j_m))**2)

        resultado.append(a_n)
    return resultado
        

#===============================
#Funciones de la cadena colgante
#===============================
def u_xt(x, t, g, L, lista_j, coeficientes):
    """
    Función que construye u(x,t) usando los coeficientes a_n y la serie de funciones
    u(x,t) = sum_{n=1}^{5} a_n cos( sqrt(g)*j_n/(2 sqrt(L)) * t ) J_0(j_n*sqrt(x/L))
    """
    u_val = 0
    for j, a_n in zip(lista_j, coeficientes):
        u_val += a_n * np.cos(np.sqrt(g)*j/(2*np.sqrt(L)) *t) * j0(j * np.sqrt(x/L))

    return u_val


#=====================
#Funciones de graficar
#=====================
def visualizar_evolucion2(w,historial_w, x):
    '''
    Visualizamos los distintos estados de la cadena colgante en el tiempo. 
    Esta se representa como una serie de puntos en el plano (u(x,t), x).
    Coge los datos del hitorico y los va actualizando en el tiempo.
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
        
        titulo.set_text(f"Paso: {frame}")
    
        t = tiempos[frame]  # Tomamos el tiempo actual
        
        # Tomamos toda la columna de w en el tiempo t
        w_actual = historial_w[t]  # Ahora tomamos todos los puntos en el tiempo t
        
        # Graficamos todos los puntos (x, w(x,t))
        scatter.set_data(w_actual, x)  # X es u(x,t) y Y es x
        
        return scatter, titulo
    #interval es el tiempo entre cada frame en milisegundos. Si aumentamos va más lento
    #blit=True hace que solo se actualicen los puntos que cambian, no toda la figura
    ani = animation.FuncAnimation(fig, actualizar, frames=len(tiempos), interval=210, blit=False, repeat=False)
    ani.save("cadena_colgante/graficas_cadena/animacion_fokker_plank.gif", writer="pillow", fps=5)
    plt.show()

#==============================
#Funcione para guardar imagenes
#==============================
def guardar_imagen(x,y,i):
    '''
    Guardamos la imagen en el directorio especificado con el nombre especificado.
    '''
    # Definimos el directorio, el nombre del archivo y la extensión
    nombre_archivo = "cadena_teorica_paso_{}.png".format(i)
    directorio = "C:/Users/andre/Desktop/Cosas Uni/cadena_fotos/cadena_teorica"
    ruta_completa = os.path.join(directorio, nombre_archivo)
    
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
    ax.set_title("Cadena Teórica en el Tiempo {}".format(i))

    # Guardamos la figura
    plt.savefig(ruta_completa, dpi=300, bbox_inches='tight')
    plt.close(imagen)  # Cierra la figura para liberar memoria


#====
#Main
#====
if __name__ == "__main__":
    
    '''
    -------------------------------------------------------------------
    Solución teórica de la ecuación de la cadena. 
    -------------------------------------------------------------------
    u(x,y) = sumatorio de n = 1 a 5 (a_n cos(g^(1/2) j_n/(2 L^(1/2)) t ) J_0(j_n (x/L)^(1/2) ) )
    ut(x,0) = 0
    u(x,0) = f(x)
    t > 0
    f(x) = np.exp(-x) - x**2/2
    u(L,t) = 0

    a_n = 1/( L * (J_1(j_n))^2 ) * integral de 0 a L (f(x) J_0(j_n (x/L)^(1/2)) dx)
    J_0 (x) = 1 - x^2/4 + x^4/64 - x^6/1152 + x^8/24576 (son los términos del 0 al 4)
    J_1 (x) = x/2 - x^3/16 + x^5/384 - x^7/9216 + x^9/221184 (son los términos del 0 al 4) 
    j_n,0: j_1 = 2.40482, j_2 = 5.52007, j_3 = 8.65372, j_4 = 11.79153, j_5 = 14.93091
    '''
    
    # Definición de los parámetros
    a = c = 0
    d = 1
    L = 3
    g = 9.8
    
    Nx = 50
    Ny = 300
    
    j_values = jn_zeros(0, 5) #j_n,0: j_1 = 2.40482, j_2 = 5.52007, j_3 = 8.65372, j_4 = 11.79153, j_5 = 14.93091
    print("Valores de j_n:", j_values)
    
    for jn in j_values:
        print(f"J_0({jn}) = {j0(jn):.10f}")


    #verificamos que L/Nx sea par
    if (L % Nx != 0) and ((L // Nx) % 2 != 0): 
        raise ValueError("L/Nx debe ser par.")
    
    else: #Es par, entonces lo calculamos

        an = coeficientes_a_n(L, Nx, j_values)
        print("Coeficientes a_n:")
        for i, coeff in enumerate(an, start=1):
            print(f"a_{i} = {coeff:.6f}")
            
        # Definición de la malla
        x_valores = np.linspace(a, L, Nx + 1)  
        t_valores = np.linspace(c, d, Ny + 1)  
        
        #creamos una matriz de ceros de Nx+1 filas y Ny+1 columnas
        u = np.zeros((Nx + 1, Ny + 1))  
        
        historial_u = {}
        # Guardamos el estado inicial
        historial_u[0] = np.copy(u)  
  
        for j, t in enumerate(t_valores):
            for i, x in enumerate(x_valores):
                # Calculamos u(x,t) usando la función u_funcion
                u[i, j] = u_xt(x, t, g, L, j_values, an)
            historial_u[t] = u[:, j].copy()
     
        
        visualizar_evolucion2(u,historial_u, x_valores)
        
        print('u:', u)
        print('u.shape:', u.shape)
        print('Valor máximo de u:', np.max(u))
        print('Valor mínimo de u:', np.min(u))
        
        #guardamos el resultado en un archivo .txt
        np.savetxt("cadena_colgante/solucion_cadena/u_fokker_plank.csv", u, delimiter=",")
        
        
        guardar_imagen(u,x_valores,0) #Guardamos la imagen del primer paso
        guardar_imagen(u,x_valores,int((Ny)/3)) #guardamos la imagen en un carto del progreso
        guardar_imagen(u,x_valores,int((Ny)*2/3)) #guardamos la imagen en dos carto del progreso
        guardar_imagen(u,x_valores,Ny) #Guardamos la imagen del ultimo paso