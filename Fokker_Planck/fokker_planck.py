
#Importar librerías
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scipy.interpolate import interp1d
from funciones_graficar import graficar_superficies_temporales, graficar_matrices_superpuestas, anima_comparacion, animar_derivadas_3d, graficar_derivadas_3d


# ===========================
# Función inicial
# ===========================
def funcion_f(x):
    return (x**2)

def funcion_f1(x):
    return np.sin(np.pi/5 * x)

def funcion_f2(x):
    return 1

# ===========================
# Configuración de parámetros
# ===========================
def configurar_parametros(a, b, c, d, Nx, Ny):
    h = (b - a) / Nx
    k = (d - c) / Ny
    print(f'k = {k}, h = {h}')
    return h, k


# ===========================
# Crear matrices iniciales
# ===========================
def inicializar_matriz(a, b, c, d, Nx, Ny, ε, funcion_inicial, transformacion):
    #inicializamos la malla normal
    x = np.linspace(a,  b, Nx + 1)
    y = np.linspace(c, d, Ny + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    w = np.zeros((Nx + 1, Ny + 1))

    if transformacion == 0:

        print("EDP sin transformación")
        # Condiciones iniciales y de frontera
        for i in range(1, Nx):
            w[i, 0] = funcion_inicial(a + i*h)  # u(x,0) = f(x)

        return X, Y, w
    
    elif transformacion == 1:
        # x' = x + ε * e^(-t) = (a + i*h) + ε * e^(-(c + j*k))
        # t' = t = c + j*k
        # u' = u

        X_trans = X + ε * np.exp(-Y)
        
        print(f"Grupo de Lie 1: Ecuación con transformación, epsilon = {ε}")
   
        # Grupo de Lie 1: aplicar la transformación a la condición inicial
        for i in range(1, Nx):
            
            w[i, 0] = funcion_inicial(X_trans[i,0] - ε) # u'(x',0) = u(x'-ε,0) --> u(x,0) = f(x), entonces u(x',0) = f(x' - ε)
            #la función está para los ejes iniciales --> se le quita la transformación
        
        return X_trans, Y, w

            
    elif transformacion == 2:
        # Grupo de Lie 2: aplicar la transformación a la condición inicial
        # x' = x - ε * e^(t) = (a + i*h) - ε * e^((c + j*k))
        # t' = t = c + j*k
        # u' = u * e^(x * ε * e^(t) - 1/2 * ε^2 * e^(2t))

        #calculamos la nueva X
        X_trans = X - ε * np.exp(Y)
        
        print(f"Grupo de Lie 2: Ecuación con transformación, epsilon = {ε}")
        
        for i in range(1, Nx):
            # x' = x - ε * e^(t)
            x_inicial = X_trans[i,0] + ε 
            param =  np.exp(x_inicial* ε  - 1/2 * ε**2 ) #depende de la x inicial --> todo va en función de ello
            
            w[i, 0] = funcion_inicial(x_inicial) * param #se calcula en los x_iniciales debido a que la función está para el sistema inicial 

        return X_trans, Y, w




# ===========================
# Discretización de la EDP
# ===========================
def ecuacion_inicial(w, i, j, h, k, a):
    # ux = regresivo
    pri = w[i+1,j] * (k) + w[i-1,j] * (k - (a+i*h)*h*k) + w[i,j] *(-2*k + (a+i*h)*h*k + h**2 + h**2 * k)
    seg = h**2
    
    return pri, seg



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


# ===========================
# Método de evolución temporal
# ===========================
def met_evolucion(w, Nx, Ny, h, k, a):
    
    for j in range(0, Ny): #Al ser de evolución comienza en 0 para actualizar todas las filas, salvo el borde. 
        for i in range(1, Nx): 
            
            #EDP
            edp_parte1, edp_parte2 = ecuacion_inicial(w, i, j, h, k, a)
            
            if not verificar_discretizacion(edp_parte1, edp_parte2, i, j):
                break
            else:
                w[i,j+1] = edp_parte1 / edp_parte2
    return w
        

# ===========================
# Verificar estabilidad
# ===========================
def verificar_estabilidad(k, h):
    if k / h**2 > 0.5:
        print("El método no converge")
        print(f"k/h^2 = {k/h**2}")
        return False
    return True



# ===================================================
# Aplicar el grupo de Lie 1 a la solución de la EDP
# Solución teórica
# ===================================================
def solucion_transformada1(X, T, w_final, epshidon):
    
    '''
    Coniste en cambiar los valores de x pero dejando t y u en su sitio. 
    (x+ε * e^(-t), t, u(x,t))
    '''
    
    #extraemos las dimensiones de la matriz X
    n_x, n_t = X.shape
    X_trans = np.zeros_like(X) #creamos una nueva malla para la transformación
    w_trans = np.zeros_like(w_final) #creamos una nueva matriz para la solución transformada
    
    for j in range(n_t):
        # cogemos los valores de las primeras filas de la malla
        t_val = T[0,j]
        
        # Definimos la malla transformada en esta fila
        x_prime = X[:, j] + epshidon * np.exp(-t_val)
        #Asociamos la nueva fila de x_prime para la malla transformada en la columna que estamos evaluando
        X_trans[:,j] = x_prime
        
        # La relación inversa es: x = x' - ε * e^(-t)
        #cogemos la inversa para interpolarla
        x_orig = x_prime - epshidon * np.exp(-t_val)
        
        # Interpolamos la solución inicial en la fila para obtener w_trans en x_orig
        interp_func = interp1d(X[:,j], w_final[:,j], kind='cubic',
                               bounds_error=False, fill_value="extrapolate")
        w_trans[:,j] = interp_func(x_orig)

    return X_trans, T, w_trans

# ===================================================
# Aplicar el grupo de Lie 2 a la solución de la EDP
# Solución teórica
# ===================================================
def solucion_transformada2(X, T, w_final, epshidon):
    '''
    Transformamos los puntos de la solución inicial (x - ε * e^(t), t, u(x,t)* e^(x * ε * e^(t) - 1/2 * ε^2 * e^(2t)))
    '''
    n_x, n_t = X.shape  # n_x = 51, n_t = 401
    X_trans = np.zeros_like(X)
    w_trans = np.zeros_like(w_final)
    
    for j in range(n_t):  # Iteramos sobre las columnas (cada instante t)
        # Se extrae el tiempo; asumimos que en la columna j, T es constante, usamos el primer valor:
        t_val = T[0, j]
        
        # Definir la malla transformada en este instante

        x_prime = X[:, j] - epshidon * np.exp(t_val)
        X_trans[:, j] = x_prime
        
        # La relación inversa para obtener x en la malla inicial:
        x_orig = x_prime + epshidon * np.exp(t_val)
        
        # Interpolamos la solución inicial en la columna j
        interp_func = interp1d(X[:, j], w_final[:, j], kind='linear',
                               bounds_error=False, fill_value="extrapolate")
        u_orig = interp_func(x_orig)
        
        # Factor de transformación para u
        factor = np.exp(x_orig * epshidon * np.exp(t_val) - 0.5 * epshidon**2 * np.exp(2*t_val))
        w_trans[:, j] = u_orig * factor
    
    
    return X_trans, T,  w_trans


# ===========================
# Función para calcular 
# derivadas numéricamente
# ===========================
def calcular_derivada_numerica(soluciones, epsilons):
        """
        Calcula la derivada numérica (f(x+h) - f(x)) / h para cada epsilon.
        """
        derivadas = {}
        for i in range(len(epsilons) - 1):
            eps = epsilons[i]
            eps_next = epsilons[i + 1]
            h = eps_next - eps

            X, T, U = soluciones[eps]
            X_next, T_next, U_next = soluciones[eps_next]

            # Derivada numérica
            dX = (X_next - X) / h
            dT = (T_next - T) / h
            dU = (U_next - U) / h

            derivadas[eps] = (dX, dT, dU)
        return derivadas
    
# ===========================
# Función para calcular
# derivadas teóricas
# ===========================    
def calcular_derivada_teorica(transf, soluciones, epsilons):
    """
    Calcula la derivada teórica para cada epsilon.
    """
    derivadas_teoricas = {}
    for eps in epsilons:
        X, T, U = soluciones[eps]

        if transf == 1:
            # Transformación 1: x = e^-t, t = 0, u = 0
            dX = np.exp(-T)
            dT = np.zeros_like(T)
            dU = np.zeros_like(U)
        elif transf == 2:
            # Transformación 2: x = -e^t, t = 0, u = x * u * e^t
            dX = -np.exp(T)
            dT = np.zeros_like(T)
            dU = X * U * np.exp(T)

        derivadas_teoricas[eps] = (dX, dT, dU)
    return derivadas_teoricas



# ======
# Main
# ======
if __name__ == "__main__":
    
    '''
    Ec de Fokker-Planck: uxx + x ux -ut + u = 0
    u(x,0) = f(x), f1(x) = x^2, f2(x) = sen(np.pi/5 * x), f3(x) = 1
    x = [0,5], 0 < t < 0.5
    Nx = 50
    Ny = 400
    Se resuelve la EDP, después se transforma la solución aplicandole los grupos de Lie 1 y 2.
    Se adapta los datos iniciales (de los grupos de Lie 1 y 2) para poder resolver las nuevas EDPs utilizando los datos de la EDP del enunciado.
    Comparamos las transformarción de las soluciones iniciales con las soluciones numéricas de las EDP de los grupos de Lie.
    '''
    # Parámetros
    a, c = 0, 0  # Límites iniciales
    b, d = 5, 0.5  # Límites finales
    Nx, Ny = 50, 400  # Resolución espacial y temporal
    # ε = 4  
    epsilons1 = np.round(np.linspace(4.0, 6.0, 11), 2)
    epsilons2 = np.round(np.linspace(0.3, 1, 11), 2)

    # Configuración, calculamos los parámetros h y k
    h, k = configurar_parametros(a, b, c, d, Nx, Ny)

    # Inicializar matriz y condiciones iniciales
    # f(x) = x^2
    transformacion = 0  
    X, Y, w= inicializar_matriz(a, b, c, d, Nx, Ny, None, funcion_f, transformacion)
    
    # Inicializar matriz y condiciones iniciales
    # f(x) = sen(np.pi/5 * x)
    transformacion = 0  
    X1, Y1, w1= inicializar_matriz(a, b, c, d, Nx, Ny, None, funcion_f1, transformacion)
    
    # Inicializar matriz y condiciones iniciales
    # f(x) = 1
    transformacion = 0
    X2, Y2, w2= inicializar_matriz(a, b, c, d, Nx, Ny, None, funcion_f2, transformacion)


    if verificar_estabilidad(k, h): #Si es estable, entonces resolvemos la ecuación
        
        # -------------------
        # EDP inicial
        # f(x) = x^2
        # -------------------
        # Resolver la EDP inicial, f(x) = x^2
        w_final = met_evolucion(w, Nx, Ny, h, k, a)
        
        #Resultados
        print(f"w_final: {w_final}")
        print("Máximo valor de w:", np.max(w_final))
        print("Mínimo valor de w:", np.min(w_final))
        
        # Graficar solución
        graficar_superficies_temporales(X, Y, w_final, 'Sol_EDP', 'Solución de la EDP con x^2', 'x2')
        
        #-------------------
        # EDP inicial
        # f(x) = sen(np.pi/5 * x)
        #-------------------
        #Resolver la EDP inicial, f(x) = sen(np.pi/5 * x)
        w_final1 = met_evolucion(w1, Nx, Ny, h, k, a)
        
        #Resultados
        print(f"w_final1: {w_final1}")
        print("Máximo valor de w:", np.max(w_final1))
        print("Mínimo valor de w:", np.min(w_final1))
        
        # Graficar solución
        graficar_superficies_temporales(X1, Y1, w_final1, 'Sol_EDP', 'Solución de la EDP con seno', 'seno')
        
        #-------------------
        # EDP inicial
        # f(x) = 1
        #-------------------
        #Resolver la EDP inicial, f(x) = 1
        w_final2 = met_evolucion(w2, Nx, Ny, h, k, a)
        
        #si los valores de la solución son menores que 1, los cambiamos a 1
        w_final2_copy = np.copy(w_final2)
        w_final2_copy[w_final2_copy < 1] = 1
        
        #Resultados
        print(f"w_final2: {w_final2}")
        print("Máximo valor de w:", np.max(w_final2))
        print("Mínimo valor de w:", np.min(w_final2))
        
        # Graficar solución
        graficar_superficies_temporales(X2, Y2, w_final2_copy, 'Sol_EDP', 'Solución de la EDP con 1', '1')
        
        
        # --------------------------
        # Grupo de Lie o transformación 1
        # f(x) = x^2
        #--------------------------
        soluciones_1, primas_1 = {}, {}
        transformacion = 1
        for eps in epsilons1:
            # Inicializar matriz y condiciones iniciales para la transformación 1
            X_1, Y_1, w_1 = inicializar_matriz(a, b, c, d, Nx, Ny, eps, funcion_f, transformacion)
            
            # Resolver la ecuación con la transformación 1
            w_final_transf_1 = met_evolucion(w_1, Nx, Ny, h, k, a)
            soluciones_1[eps] = (X_1, Y_1, w_final_transf_1)

            #Transformar la solución anterior (de la EDP) con  G1
            X_prime, Y_prime, w_prime = solucion_transformada1(X, Y, w_final, eps)
            primas_1[eps] = (X_prime, Y_prime, w_prime)
            
            #cogemos algunos valores de epsilon para graficar
            if eps == 4.0 or eps == 4.6 or eps == 5.0 or eps == 5.4 or eps == 5.8:
                graficar_matrices_superpuestas(X_1, Y_1, w_final_transf_1, X_prime, Y_prime, w_prime, f'Comparacion_G1_sol_real_sol_teo_x2_{eps}', 'x2', 1)
    
            
        graficar_matrices_superpuestas(X_1, Y_1, w_final_transf_1, X_prime, Y_prime, w_prime, f'Comparacion_G1_sol_real_sol_teo_x2_{eps}', 'x2', 1)
        anima_comparacion(soluciones_1, primas_1, 'x2', '1', 'G1_sol_num_vs_teo_x2', epsilons1, Nx, Ny, c, d, 'Solución numérica de la EDP G1', 'Solución teórica G1')
        
        #---------------------------
        # Grupo de Lie o transformación 2
        # f(x) = x^2
        #---------------------------
        soluciones_2, primas_2 = {}, {}
        transformacion = 2
        for eps in epsilons2:
            # Inicializar matriz y condiciones iniciales para G2
            X_2, Y_2, w_2 = inicializar_matriz(a, b, c, d, Nx, Ny, eps, funcion_f, transformacion)
            
            # Resolver la ecuación con la transformación 2
            w_final_transf_2 = met_evolucion(w_2, Nx, Ny, h, k, a)
            soluciones_2[eps] = (X_2, Y_2, w_final_transf_2)
            
            #Transformar la solución anterior (de la EDP) con  G2
            X_prime2, Y_prime2, w_prime2 = solucion_transformada2(X, Y, w_final, eps)
            primas_2[eps] = (X_prime2, Y_prime2, w_prime2)

            if eps == 0.3 or eps == 0.51 or eps == 0.72 or eps== 0.86 or eps == 0.93 :
                graficar_matrices_superpuestas(X_2, Y_2, w_final_transf_2, X_prime2, Y_prime2, w_prime2, f'Comparacion_G2_sol_real_sol_teo_x2__{eps}', 'x2', 2)
                
        graficar_matrices_superpuestas(X_2, Y_2, w_final_transf_2, X_prime2, Y_prime2, w_prime2, f'Comparacion_G2_sol_real_sol_teo_x2__{eps}', 'x2', 2)
        anima_comparacion(soluciones_2, primas_2, 'x2', '2', 'G2_sol_num_vs_teo_x2', epsilons2, Nx, Ny, c, d, 'Solución numérica de la EDP G2', 'Solución teórica G2')
      
        
        # ----------------------------
        # Comparar campos de vectores
        # f(x) = x^2
        # ----------------------------
        # Graficar campos de vectores para la transformación 1
        transf = 1
        derivadas_numericas = calcular_derivada_numerica(primas_1, epsilons1)
        derivadas_teoricas = calcular_derivada_teorica(transf, primas_1, epsilons1)
        animar_derivadas_3d(primas_1, epsilons1, derivadas_numericas, derivadas_teoricas, transf, 'x2', 0.3, 0.005)  
        print(derivadas_numericas.keys())
        for epsilon in [4.0, 4.6, 5.0, 5.4, 5.8]:
            graficar_derivadas_3d(primas_1, epsilon, derivadas_numericas, derivadas_teoricas, transf,'x2', 0.3, 0.005)

        # Graficar campos de vectores para la transformación 2
        transf = 2 
        derivadas_numericas = calcular_derivada_numerica(primas_2, epsilons2)
        derivadas_teoricas = calcular_derivada_teorica(transf, primas_2, epsilons2)
        animar_derivadas_3d(primas_2, epsilons2, derivadas_numericas, derivadas_teoricas, transf, 'x2', 0.3, 0.02)
        for epsilon in [0.3, 0.51, 0.72, 0.86, 0.93]:
            graficar_derivadas_3d(primas_2, epsilon, derivadas_numericas, derivadas_teoricas, transf,'x2', 0.3, 0.02)
            
            
        #---------------------------
        # Grupo de Lie o transformación 1
        # f(x) = sen(np.pi/5 * x)
        #---------------------------
        soluciones_1, primas_1 = {}, {}
        transformacion = 1
        for eps in epsilons1:
            # Inicializar matriz y condiciones iniciales para la transformación 1
            X_1, Y_1, w_1 = inicializar_matriz(a, b, c, d, Nx, Ny, eps, funcion_f1, transformacion)
            
            # Resolver la ecuación con la transformación 1
            w_final_transf_1 = met_evolucion(w_1, Nx, Ny, h, k, a)
            soluciones_1[eps] = (X_1, Y_1, w_final_transf_1)

            #Transformar la solución anterior (de la EDP) con  G1
            X_prime, Y_prime, w_prime = solucion_transformada1(X1, Y1, w_final1, eps)
            primas_1[eps] = (X_prime, Y_prime, w_prime)
            
            if eps == 4.0 or eps == 4.6 or eps == 5.0 or eps == 5.4 or eps == 5.8:
                graficar_matrices_superpuestas(X_1, Y_1, w_final_transf_1, X_prime, Y_prime, w_prime, f'Comparacion_G1_sol_real_sol_teo_sen_{eps}', 'seno', 1)
        
        graficar_matrices_superpuestas(X_1, Y_1, w_final_transf_1, X_prime, Y_prime, w_prime, f'Comparacion_G1_sol_real_sol_teo_sen_{eps}', 'seno', 1)
        anima_comparacion(soluciones_1, primas_1, 'seno', '1', 'G1_sol_num_vs_teo_sen', epsilons1, Nx, Ny, c, d, 'Solución numérica de la EDP G1', 'Solución teórica G1')
        
        
        #---------------------------
        # Grupo de Lie o transformación 2
        # f(x) = sen(np.pi/5 * x)
        #---------------------------
        soluciones_2, primas_2 = {}, {}
        transformacion = 2
        for eps in epsilons2:
            # Inicializar matriz y condiciones iniciales para la transformación 2
            X_2, Y_2, w_2 = inicializar_matriz(a, b, c, d, Nx, Ny, eps, funcion_f1, transformacion)
            
            # Resolver la ecuación con la transformación 2
            w_final_transf_2 = met_evolucion(w_2, Nx, Ny, h, k, a)
            soluciones_2[eps] = (X_2, Y_2, w_final_transf_2)
            
            #Transformar la solución anterior (de la EDP) con  G2
            X_prime2, Y_prime2, w_prime2 = solucion_transformada2(X1, Y1, w_final1, eps)
            primas_2[eps] = (X_prime2, Y_prime2, w_prime2)

            if eps == 0.3 or eps == 0.51 or eps == 0.72 or eps== 0.86 or eps == 0.93: 
                graficar_matrices_superpuestas(X_2, Y_2, w_final_transf_2, X_prime2, Y_prime2, w_prime2, f'Comparacion_G2_sol_real_sol_teo_sen_{eps}', 'seno', 2)
            
        graficar_matrices_superpuestas(X_2, Y_2, w_final_transf_2, X_prime2, Y_prime2, w_prime2, f'Comparacion_G2_sol_real_sol_teo_sen_{eps}', 'seno', 2)
        anima_comparacion(soluciones_2, primas_2, 'seno', '2', 'G2_sol_num_vs_teo_sen', epsilons2, Nx, Ny, c, d, 'Solución numérica de la EDP G2', 'Solución teórica G2')
        
        
        
        # ----------------------------
        # Comparar campos de vectores
        # f(x) = sen(np.pi/5 * x)
        # ----------------------------
        # Graficar campos de vectores para la transformación 1
        transf = 1
        derivadas_numericas = calcular_derivada_numerica(primas_1, epsilons1)
        derivadas_teoricas = calcular_derivada_teorica(transf, primas_1, epsilons1)
        animar_derivadas_3d(primas_1, epsilons1, derivadas_numericas, derivadas_teoricas, transf, 'seno', 0.3, 0.005)
        for epsilon in [4.0, 4.6, 5.0, 5.4, 5.8]:
            graficar_derivadas_3d(primas_1, epsilon, derivadas_numericas, derivadas_teoricas, transf, 'seno', 0.3, 0.005)
        
        # Graficar campos de vectores para la transformación 2
        transf = 2
        derivadas_numericas = calcular_derivada_numerica(primas_2, epsilons2)
        derivadas_teoricas = calcular_derivada_teorica(transf, primas_2, epsilons2)
        animar_derivadas_3d(primas_2, epsilons2, derivadas_numericas, derivadas_teoricas, transf,'seno', 0.3, 0.005)
        for epsilon in [0.3, 0.51, 0.72, 0.86, 0.93]:
            graficar_derivadas_3d(primas_2, epsilon, derivadas_numericas, derivadas_teoricas, transf,'seno', 0.3, 0.005)
        
        
        
        #---------------------------
        # Grupo de transformación 1
        # f(x) = 1
        #---------------------------
        soluciones_1, primas_1 = {}, {}
        transformacion = 1
        for eps in epsilons1:
            # Inicializar matriz y condiciones iniciales para la transformación 1
            X_1, Y_1, w_1 = inicializar_matriz(a, b, c, d, Nx, Ny, eps, funcion_f2, transformacion)
            
            # Resolver la ecuación con la transformación 1
            w_final_transf_1 = met_evolucion(w_1, Nx, Ny, h, k, a)
            
            #si el valor de la solución es menor que 1, lo cambiamos a 1
            w_final_transf_1_copy = np.copy(w_final_transf_1)
            w_final_transf_1_copy[w_final_transf_1_copy < 1] = 1
            
            soluciones_1[eps] = (X_1, Y_1, w_final_transf_1_copy)
            
            #Transformar la solución anterior (de la EDP) con  G1
            X_prime, Y_prime, w_prime = solucion_transformada1(X2, Y2, w_final2, eps)
            
            #si el valor de la solución es menor que 1, lo cambiamos a 1
            w_prime_copy = np.copy(w_prime)
            w_prime_copy[w_prime_copy < 1] = 1
            
            primas_1[eps] = (X_prime, Y_prime, w_prime_copy)
            
            if eps == 4.0 or eps == 4.6 or eps == 5.0 or eps == 5.4 or eps == 5.8:
                graficar_matrices_superpuestas(X_1, Y_1, w_final_transf_1_copy, X_prime, Y_prime, w_prime_copy, f'Comparacion_G1_sol_real_sol_teo_1_{eps}', '1', 1)
        
        graficar_matrices_superpuestas(X_1, Y_1, w_final_transf_1_copy, X_prime, Y_prime, w_prime_copy, f'Comparacion_G1_sol_real_sol_teo_1_{eps}', '1', 1)
        anima_comparacion(soluciones_1, primas_1, '1', '1', 'G1_sol_num_vs_teo_1', epsilons1, Nx, Ny, c, d, 'Solución numérica de la EDP G1', 'Solución teórica G1')
        
        
        #---------------------------
        # Grupo de transformación 2
        # f(x) = 1
        #---------------------------
        soluciones_2, primas_2 = {}, {}
        transformacion = 2
        for eps in epsilons2:
            # Inicializar matriz y condiciones iniciales para la transformación 2
            X_2, Y_2, w_2 = inicializar_matriz(a, b, c, d, Nx, Ny, eps, funcion_f2, transformacion)
            
            # Resolver la ecuación con la transformación 2
            w_final_transf_2 = met_evolucion(w_2, Nx, Ny, h, k, a)
            soluciones_2[eps] = (X_2, Y_2, w_final_transf_2)
            
            #Transformar la solución anterior (de la EDP) con  G2
            X_prime2, Y_prime2, w_prime2 = solucion_transformada2(X2, Y2, w_final2, eps)
            primas_2[eps] = (X_prime2, Y_prime2, w_prime2)
            
            if eps == 0.3 or eps == 0.51 or eps == 0.72 or eps== 0.86 or eps == 0.93: 
                graficar_matrices_superpuestas(X_2, Y_2, w_final_transf_2, X_prime2, Y_prime2, w_prime2, f'Comparacion_G2_sol_real_sol_teo_1_{eps}', '1', 2)
        
        graficar_matrices_superpuestas(X_2, Y_2, w_final_transf_2, X_prime2, Y_prime2, w_prime2, f'Comparacion_G2_sol_real_sol_teo_1_{eps}', '1', 2)
        anima_comparacion(soluciones_2, primas_2, '1', '2','G2_sol_num_vs_teo_1', epsilons2, Nx, Ny, c, d, 'Solución numérica de la EDP G2', 'Solución teórica G2')
        
        
        # ----------------------------
        # Comparar campos de vectores
        # f(x) = 1
        # ----------------------------
        # Graficar campos de vectores para la transformación 1
        transf = 1
        derivadas_numericas = calcular_derivada_numerica(primas_1, epsilons1)
        derivadas_teoricas = calcular_derivada_teorica(transf, primas_1, epsilons1)
        animar_derivadas_3d(primas_1, epsilons1, derivadas_numericas, derivadas_teoricas, transf, '1', 0.3, 0.005)
        for epsilon in [4.0, 4.6, 5.0, 5.4, 5.8]:
            graficar_derivadas_3d(primas_1, epsilon, derivadas_numericas, derivadas_teoricas, transf,'1', 0.3, 0.005)
        
        # Graficar campos de vectores para la transformación 2
        transf = 2
        derivadas_numericas = calcular_derivada_numerica(primas_2, epsilons2)
        derivadas_teoricas = calcular_derivada_teorica(transf, primas_2, epsilons2)
        animar_derivadas_3d(primas_2, epsilons2, derivadas_numericas, derivadas_teoricas, transf,'1', 0.3, 0.005)
        for epsilon in [0.3, 0.51, 0.72, 0.86, 0.93]:
            graficar_derivadas_3d(primas_2, epsilon, derivadas_numericas, derivadas_teoricas, transf,'1', 0.3, 0.005)