#Importamos las librerías necesarias
import numpy as np
from func_graficar import graficar_superficies

# ===========================
# Función inicial
# ===========================
def funcion_f(x,y,z):
    return np.sin(np.pi*x)*np.sin(np.pi*y)


# ===========================
# Crear matrices iniciales
# ===========================
def inicializar_matriz(a, b, c, d, e, f, Nx, Ny, Nz):
    
    '''
    Función encargada de inicializar la matriz w y los vectores X, Y, Z. 
    Además de aplicar las condiciones de frontera.
    '''
    # Creamos la malla
    # vectores de discretización
    x = np.linspace(a,b,Nx+1) 
    y = np.linspace(c,d,Ny+1) 
    z = np.linspace(e,f,Nz+1) 
    # Matrices tridimensionales de tamaño Nx+1, Ny+1, Nz+1 cada una
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij') 

    # Inicializar la matriz w con ceros
    w = np.zeros((Nx+1,Ny+1,Nz+1))


    # Aplicar las condiciones de frontera
    # x = a + i*h, y = c + j*k, z = e + l*p
    
    # Para el plano XY
    for i in range(1,Nx):
        for j in range(1,Ny):
            w[i,j,0] = 0
            w[i,j,Nz] = funcion_f(a + i*h, c+j*k, f) 

    # Para el plano XZ
    for i in range(1,Nx):
        for l in range(1,Nz):
            w[i,0,l] = 0
            w[i,Ny,l] = 0

    # Para el plano YZ        
    for j in range(1,Ny):
        for l in range(1,Nz):
            w[0,j,l] = 0
            w[Nx,j,l] = 0
            
    return w, X, Y, Z


# ======================================
# Verificar parámetros de discretización
# ======================================
def verificar_discretizacion (edp_parte1,edp_parte2, i, j):
    '''
    Función encargada de verificar los parámetros de discretización.
    Si alguno de los parámetros no es válido (muy pequeño), se imprime un mensaje de error y se retorna False.
    '''
    
    if np.isnan(edp_parte1) or np.isinf(edp_parte1) or abs(edp_parte1) > 1e10:
        print(f"Valores no válidos detectados en pri: {edp_parte1}, i={i}, j={j}")
        return False
            
    if abs(edp_parte2) < 1e-10:
        print(f"Posible inestabilidad en seg = {edp_parte2} en i={i}, j={j}")
        return False
    return True


#========================
# Método de Gauss-Seidel
#========================

def gauss_seidel_iterations(w, num_iter, tol=1e-5): 
    '''
    Aplicamos el metodo de gauss-seidel para resolver el problema del Laplaciano en 3D.
    w: matriz inicializada con ceros y condiciones de frontera.
    num_iter: número máximo de iteraciones.
    tol: tolerancia para el criterio de convergencia.
    historial: lista que almacena las soluciones en cada iteración.
    '''
    historial = []
    converged = False

    for it in range(num_iter):
        w_old = w.copy()
        
        for i in range(1, Nx):  
            for j in range(1, Ny):
                for l in range(1, Nz):
                    edp_parte1 = (k**2 * p**2 * (w[i+1,j,l] + w[i-1,j,l]) + h**2 * p**2 * (w[i,j+1,l] + w[i,j-1,l]) 
                                   + h**2 * k**2 * (w[i,j,l+1] + w[i,j,l-1]))
                    edp_parte2 = 2*(h**2 * k**2 + k**2 * p**2 + h**2 * p**2)
                    
                    if not verificar_discretizacion(edp_parte1, edp_parte2, i, j):
                        break
                    else:
                        w[i,j,l] = edp_parte1 / edp_parte2
                    
                    
        historial.append(w.copy())

        # Criterio de convergencia
        # Comprobamos si la solución ha convergido comparando la diferencia entre la solución actual y la anterior.
        # Si la diferencia es menor que la tolerancia, se considera que ha convergido.
        error = np.max(np.abs(w - w_old)) 
        if error < tol: 
            print(f"Convergencia alcanzada en {it} iteraciones.")
            converged = True
            break
    
    # Si no se ha alcanzado la convergencia, se imprime un mensaje de advertencia.
    # Esto puede ser debido a que el número de iteraciones es insuficiente o que la tolerancia es muy baja.
    if not converged: 
        print("Advertencia: No se alcanzó la convergencia en el número máximo de iteraciones.") 
    
    return w, historial




if __name__ == "__main__":
    
    '''
    El Lapaciano
    Au = 0 definido en el dominio D = (0,3)x(0,3)x(0,3)
    u(0,y,z) = u(3,y,z) = u(x,0,z) = u(x,3,z) = u(x,y,0) = 0
    u(x,y,3) = 2sin(pi*x)sin(pi*y), f(x,y,z) = 2sin(pi*x)sin(pi*y)
    Nx = Ny = Nz = 30
    '''

    # Dimensiones del dominio
    Nx = Ny = Nz = 30
    a = c = e = 0
    b = d = f = 3

    # Tamaño de los intervalos
    h = (b-a)/Nx
    k = (d-c)/Ny
    p = (f-e)/Nz
    
    w, X, Y, Z = inicializar_matriz(a, b, c, d, e, f, Nx, Ny, Nz)
       
    w_final, historial = gauss_seidel_iterations(w, num_iter=150)

    # Encontrar el valor mínimo y máximo de la solución
    min_val = np.min(w_final)
    max_val = np.max(w_final)

    print(f"El valor mínimo de la solución es: {min_val}")
    print(f"El valor máximo de la solución es: {max_val}")
    
    # Lista de colores disponibles en formato string, ordenados por el arcoiris
    colores_disponibles = ["red", "orange", "yellow", "green", "blue", "purple", "brown", "pink", "gray", "cyan", "magenta"]
    
    # Número de intervalos (limitado al número de colores disponibles)
    num_intervalos = min(len(colores_disponibles), 6)  #el min es para asegurarnos que no haya más intervalos que colores

    # Crear los intervalos automáticamente
    valores_intervalo = np.linspace(min_val, max_val, num_intervalos + 1)  # +1 porque define los bordes

    # Crear la lista de intervalos con colores en formato (numero1, numero2, "color")
    intervalos = [
        (valores_intervalo[i], valores_intervalo[i+1], colores_disponibles[i]) 
        for i in range(num_intervalos)
    ]

    print("Intervalos y colores:")
    for intervalo in intervalos:
        print(intervalo)
        
    #Graficar superficies de la solución
    graficar_superficies(intervalos, w_final, X, Y, Z, 'Laplaciano_3D')
