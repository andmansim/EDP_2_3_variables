# TFG_EDP_2_3_variables

Este repositorio contiene el código necesario para la resolución de ecuaciones en derivadas parciales (EDP) en dos y tres variables, utilizando el método de diferencias finitas.Se resuelven y evalúan distintas ecuaciones con el objetivo de verificar la validez del método y obtener una representación de su solución teórica, en algunos casos. 

El método de diferencias finitas se basa en varios submétodos. En este trabajo:
- Se emplea el método de **Gauss-Seidel** las ecuaciones **espaciales** con tres variables independientes.
- Se aplica el método de **diferencias progresivas** las ecuaciones **temporales** con dos variables independientes.

Además de su resolución, se incluye una representación animada o interativa, según el caso, para facilitar la visualización de su evolución.

La url del repositorio es: [https://github.com/andmansim/EDP_2_3_variables.git](https://github.com/andmansim/TFG_EDP_2_3_variables.git)

## Estructura del repositorio

A continuación, se describen las carpeta y archivos principales del repositorio. Todos ellos contiene comentarios que explican su funcionamiento:

### `Fokker_Planck/`
Contiene el código para la resolución de la ecuación de **Fokker-Planck** en dos variables independientes. Se evalúa el problema con tres condiciones iniciales distintas. En cada caso, se resuelve la EDP inicial y se compara la solución real de la EDP transformada mediante un **grupo de Lie** con la transformación de la solución inicial de la EDP mediante el mismo grupo. En concreto, se analizan dos grupos de Lie y sus álgebras, contrastando los resultados con la derivada numérica. 

### `cadena_colgante/` 
Incluye tres scripts:
- `cadena_numérica.py`: Resolución de la EDP mediante el método de diferencias finitas progresivas.
- `cadena_teorica.py`: Resolución de la solución teórica de la EDP usando la ecuación de **Bessel**.
- `comparacion_cadena.py`: Comparación gráfica entre ambas soluciones. 
   
### `Laplaciano/`
Código para la resolución del **Laplaciano** en tres variables independientes mediante el método de **Gauss-Seidel**.

###  `funciones_graficar.py`
Contiene las funciones utilizadas para representar las soluciones gráficamente, incluyendo animaciones de la evaolución temporal y la generación de imágenes en instantes específicos. También permite crear una visualización tridimensional del Laplaciano en formato HTML. La única excepción es `comparacion_cadena.py`, que incluye sus propias funciones de visualización. 

### `ejemplos_mallas/`
Código para  representar ejemplos de **mallas tridimensionales** (espaciales y temporales).

 ### `mp4/`
 Carpeta que contiene todas las **animaciones generadas** durante el desarrollo del trabajo, en formato video. 




