#Importar librerías
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from funciones_graficar import visualizar_evolucion_numerica, guardar_imagen_cadena


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
    print('Valores de k y h: ',k, h)
    
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
        visualizar_evolucion_numerica(w,historico, x)
        
        #guardar resultados en un archivo .csv
        np.savetxt("cadena_colgante/soluciones/sol_cadena_numerica.csv", w_final, delimiter=",")
        
        #Guardar imágenes de la evolución
        guardar_imagen_cadena(w_final,x,0, 'numerica')
        guardar_imagen_cadena(w_final,x,int((Ny)/3), 'numerica') 
        guardar_imagen_cadena(w_final,x,int((Ny)*2/3), 'numerica') 
        guardar_imagen_cadena(w_final,x,Ny, 'numerica') 
        