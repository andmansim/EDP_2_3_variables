
import numpy as np
from scipy.special import j0, j1
from scipy.special import jn_zeros
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from funciones_graficar import visualizar_evolucion_teorica, guardar_imagen_cadena

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
     
        
        visualizar_evolucion_teorica(u,historial_u, x_valores)
        
        print('u:', u)
        print('u.shape:', u.shape)
        print('Valor máximo de u:', np.max(u))
        print('Valor mínimo de u:', np.min(u))
        
        #guardamos el resultado en un archivo .txt
        np.savetxt("cadena_colgante/soluciones/teorica/sol_cadena_teorica.csv", u, delimiter=",")
        
        # Guardamos las imágenes de la evolución de la cadena colgante
        guardar_imagen_cadena(u,x_valores,0, 'teorica') 
        guardar_imagen_cadena(u,x_valores,int((Ny)/3), 'teorica') 
        guardar_imagen_cadena(u,x_valores,int((Ny)*2/3), 'teorica') 
        guardar_imagen_cadena(u,x_valores,Ny, 'teorica') 