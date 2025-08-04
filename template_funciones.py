import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import networkx as nx # Construcción de la red en NetworkX
import scipy
import math
#%%
museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')
#%%
# En esta línea:
# Tomamos museos, lo convertimos al sistema de coordenadas de interés, extraemos su geometría (los puntos del mapa), 
# calculamos sus distancias a los otros puntos de df, redondeamos (obteniendo distancia en metros), y lo convertimos a un array 2D de numpy
D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()
#%% Construcción matriz de adyacencia.

def construye_adyacencia(D,m): 
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

#%% Bloque descomposición LU y métodos asociados (resolución de sistemas + inversibilidad)

def calculaLU(matriz):
    # Función que realiza la descomposición LU de una matriz pasada como parámetro
    # matriz es una matriz de NxN
    # Retorna la factorización LU a través de una lista con dos matrices L y U de NxN.
    L = np.eye(matriz.shape[0]) #primero pensamos a L como la Identidad
    U = matriz.copy() #en cambio a U, la definimos a priori como una copia de la matriz
    m=matriz.shape[0] #cantidad filas
    n=matriz.shape[1] #cantidad columnas
    
    if m!=n:
        print('Matriz no cuadrada')
        return #es condición necesaria que la matriz sea cuadrada para que sea inversible
    
    for j in range(n):
        for i in range(j+1, n): 
            # Construímos la función L y U
            L[i,j] = U[i,j] / U[j,j]
            U[i,:] = U[i,:] - L[i,j] * U[j,:]
    
    return L, U

#-------------------------------------------------------
#Nos interesa poder resolver un sistema de la forma Mx = b (M matriz, b vector conocido, x vector a determinar)
#Para ello, si M es inversible, aprovechamos su descomposición LU para resolver los sistemas:
    #Ly = b
    #Ux = y
# y así hallar el vector x
#-------------------------------------------------------

def resolver_sist_triang_inf (L, w): 
    #Resolvemos el sistema Ly = w. L y w son parámetros de entrada
    #L representa la matriz triangular inferior obtenida luego de haber hecho calculaLU(matriz)
    #w representa un vector obtenido de el archivo provisto visitas.txt
    
    y = np.zeros(w.shape)
    y[0] = w[0] #como L es triangular inferior, su primer elemento de y equivale al primer elemento de w
    for i in range (1,w.shape[0]):
        y[i] = w[i] - (L[i, :i]@y[:i]) #averiguamos los siguientes elementos de y_i a partir de w_i y los anteriores y_j (j < i) 
    return y #retorna el vector y que será usado en el siguiente sistema



def resolver_sist_triang_sup (U, y): #resolvemos el sistema Ux = y. U e y son parámetros
    x = np.zeros(y.shape)
    # como U triangular superior, el último elemento del vector x equivale al último elemento de i sobre el coeficiente U[N-1][N-1].
    x[y.shape[0] -1] = y[y.shape[0] -1] / U[U.shape[0]-1, U.shape[0]-1]
    # averiguaremos los elementos del vector X de atrás para adelante, es decir, empezamos por el último y terminamos con el primero.
    # la lógica del cálculo de X es similar a la del anterior sistema, aunque ahora deberemos tener en cuenta que U[i][i] no es necesariamente 1.
    for i in range (y.shape[0]-2, -1,-1): #averiguamos los siguientes elementos de x_i a partir de y_i y los coeficientes de U.
        x[i] = (y[i] - (U[i, y.shape[0] - 1:i:-1]@x[y.shape[0]-1:i:-1]))/U[i][i]
    return x #retorna el vector x buscado

#%% Bloque Matrices K y K_inv
def construye_matriz_de_grado (A): 
    #Función que crea a la matriz de grado K, , a partir de la matriz de adyacencia pasada como parámetro
    
    K = np.zeros(A.shape) #K presenta las mismas dimensiones que la matriz de adyacencia
    for i in range(A.shape[0]): #A es cuadrada, por lo tanto A.shape[0] = A.shape[1] 
        valor = 0 
        for k in range(A.shape[0]): 
            valor += A[i][k] #nota: A presenta en sus casilleros un 0 o un 1
            K[i][i] = valor #K presenta en su diagonal la suma por filas de A
    return K #retorna la matriz de grado K


def calcular_K_inversa(K):
    #Función que invierte la matriz de grado, que se pasa como parámetro
    
    K_inv = K.copy() #la inversa tiene la misma dimensión que K
    for i in range (136): 
        #K es una matriz diagonal, por lo tanto, su inversa es el resultado de invertir cada elemento de la diagonal
        if (K[i][i] == 0): #Evitamos que se produzca la división por 0
            K_inv[i][i] = 0
        else:
            K_inv[i][i] = 1/K[i][i] #K_inv presenta en su diagonal la inversa de la suma por filas de A
        
    return K_inv #retorna la inversa de la matriz de grado K

#%% Bloque construcción matriz de transición.
def calcula_matriz_C(A): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz de transiciones C
    K = construye_matriz_de_grado(A)     
    K_inv = calcular_K_inversa(K)
    A_t = A.T
    C = A_t @ K_inv # Calcula C multiplicando Kinv y A
    
    return C


#%% Punto 3: cálculo del Pagerank.
def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = A.shape[0] # Obtenemos el número de museos N a partir de la estructura de la matriz A
    M = (N/alfa)*(np.eye(N)-(1-alfa)*C)
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    b = np.ones((N,1)) # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p

def mostrar_pagerank(p):
    #Función que imprime para cada museo los puntajes pasados como parámetro 
    q = p.tolist() #pasamos a lista los puntajes
    for i in range(len(q)):
        print(f'El puntaje del museo {i} es {q[i][0]}') #imprimimos en pantalla el puntaje de cada museo i
        
#%% Funciones para crear gráficos de la red de museos a partir de m (cantidad de conexiones) y alfa (factor amortiguamiento).

#La función grafico se podrá usar para crear una imágen que contenga un único gráfico pasándole un único par (m, alfa).
#o bien, se podrá declarar cuántos gráficos se buscan que se impriman en la imágen al ser llamada por otra de nuestras funciones.
#en este segundo caso, los parámetros de entrada son (m, alfa, ax) con ax posición donde se hallará el gráfico en la imágen de salida.
def grafico (m, alfa, ax = None):
    A = construye_adyacencia(D, m) #construímos la matriz de adyacencia que marca las conexiones de nuestra red 
    G = nx.from_numpy_array(A) # Construimos la red a partir de la matriz de adyacencia
    # Construimos un layout a partir de las coordenadas geográficas
    G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}
    factor_escala = 1e4 # Escalamos los nodos 10 mil veces para que sean bien visibles
    unico_grafico = False; #evalúa si se le pasó o no a la función un ax
    
    if ax is None: #evaluamos caso en el que no se haya pasado (imágen de salida tendrá un único gráfico)
        fig, ax = plt.subplots(figsize=(10, 10))
        barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)  # Graficamos los barrios
        unico_grafico = True #cambiamos el valor booleano de nuestra variable definida anteriormente
        
    else: #varios gráficos a realizar
        barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)  # Usamos el eje proporcionado
    
    #sector común a ambos casos
    pr = calcula_pagerank(A, alfa) #calculamos los puntajes 
    labels = {n: str(n) for i, n in enumerate(G.nodes)} # Nombres para nodos a graficar
    nx.draw_networkx(G,G_layout,node_size = pr*factor_escala, ax=ax,with_labels=False) # Graficamos red
    nx.draw_networkx_labels(G, G_layout, labels=labels, font_size=6, font_color="k", ax=ax) # Agregamos los nombres
    
    # Si ax es None (gráfico individual), mostramos el gráfico
    if unico_grafico is True:
        #Detalles estéticos
        ax.set_title('Visualización red de museos') 
        ax.legend([f'Tamaño de nodos proporcional a su Pagerank, con m = {m:f} y alfa = {alfa:f}'], loc = 'lower center')
        plt.show()
        

#------------------------------------------------

def agrupar_graficos_variacion_m(M, alfa):
    #Función para crear una única imágen que contenga todos los gráficos solicitados
    #Recibe como parámetro de entrada una lista de m (M) y un único alfa
    #La imágen tendrá los gráficos uno al lado del otro
    fig, axs = plt.subplots(1, len(M), figsize=(20, 5)) #la cantidad de columnas es equivalente a cuántos gráficos distintos se harán
    i = 0 #inicializamos índice para hablar de la posición de los gráficos en la imágen
    for m in M: #recorremos la lista pasada como parámetro, tomando cada m que la compone
        ax = axs[i]  # Seleccionamos un subplot
        grafico(m, alfa, ax = ax) # Indicamos el lugar que ocupa gráfico en la imágen
        axs[i].set_title(f'Gráfico para m={m}')
        i += 1 #aumentamos el índice para que, en caso de que haya más gráficos a realizan, se coloquen al lado del último.
    
    #Detalles estéticos
    fig.suptitle('Visualización red de museos')
    fig.legend([f'Tamaño de nodos proporcional a su Pagerank, con alfa = {alfa:f}'], loc = 'lower center')
    plt.show()


#------------------------------------------------

def agrupar_graficos_variacion_alfa (m, alfas):
    #Función para crear una única imágen que contenga todos los gráficos solicitados
    #Recibe como parámetro de entrada un único m y una lista de alfas
    
    #la imágen contendrá todos los gráficos
    #lo colocamos en dos filas y cuatro columnas
    # al ser 7, al espacio restante que no presentará gráfico lo dejamos vacío
    fig, axs = plt.subplots(2, 4, figsize=(20, 10)) 
    
    """para mayor practicidad, lo pasamos a una lista, donde a partir del subíndice 4 
    nos estaremos refiriendo a la segunda fila"""
    axs = axs.ravel() 
    i = 0 #inicializamos índice para hablar de la posición de los gráficos en la imágen
    for alfa in alfas:
        ax = axs[i]  # Seleccionamos un subplot
        grafico(m, alfa, ax=ax)  # Indicamos el lugar que ocupa gráfico en la imágen
        ax.set_title(f'Gráfico para alfa={str(alfa)}') 
        i += 1 #aumentamos el índice para que, en caso de que haya más gráficos a realizan, se coloquen donde corresponda.
        
    # Desactivamos el espacio no usado (el último)
    for j in range(len(alfas), len(axs)):
        axs[j].axis('off')  # Apagamos el subplot restante para que quede en blanco
    
    
    fig.suptitle('Visualización red de museos')
    fig.legend([f'Tamaño de nodos proporcional a su Pagerank, con m = {m:f}'], loc = 'lower center')
    plt.show()



#%% Funciones para crear lineplots donde se muestran los Page Rank de los museos con mayor puntaje

def graficos_pagerank_por_m(M, alfa):
    #Función para un gráfico donde se muestre la variación de los puntajes de los tres museos con mayor Pagerank para distintos m
    #Se muestra la evolución de éstos puntajes al modificarse el m
    #Recibe como parámetro una lista de m (M) y un único alfa  
    
    #Creamos un diccionario.
    museosCentrales = {}
    Nprincipales = 3 # Cantidad de principales.
    for m in M: #recorremos la lista de las distintas cantidades de conexiones propuestas. 
        A = construye_adyacencia(D, m) #en cada caso, la matriz de adyacencia es otra.
        p = calcula_pagerank(A, alfa) #calculamos el pagerank de cada museo en cada caso.
        principales = np.argsort(p.flatten())[-Nprincipales:] # Identificamos a los 3 principales.
        for museo in principales:
            #tomamos a cada museo que haya sido principal para algún m.
            #lo convertiremos en clave de nuestro diccionario.
            if str(museo) not in museosCentrales: #decisión estética que la clave sea string.
                museosCentrales[str(museo)] = [] #a priori, le asignamos una lista vacía a cada museo principal.
                
    #Ya teniendo todos los museos centrales, la idea es obtener el pagerank de cada uno para cada m.
    #No tiene importancia si ese museo no fue "principal" para un determinado m.
    for m in M:
        #la lógica es, para cada cantidad de conexiones, agarrar los pagerank de cada museo central.
        for museo in museosCentrales:
            A = construye_adyacencia(D, m)
            p = calcula_pagerank(A, alfa)
            museosCentrales[museo].append(p[np.int64(museo)]) #recordar que museo es str.
    
    plt.figure(figsize=(13, 10)) #decisión estética sobre ancho y largo de la imágen.
    
    for museo, pagerank in museosCentrales.items():
        plt.plot(M, pagerank, label = museo) #elaboramos un lineplot a partir de la información compilada.
    
    #Detalles de presentación
    plt.title("PageRank por cantidad de vecinos (m)", fontsize=14)
    plt.xlabel("m [Cantidad de vecinos]", fontsize=12)
    plt.ylabel("PageRank", fontsize=12)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
    plt.grid(True)
    plt.show()


#------------------------------------------------

def graficos_pagerank_por_alfa(m, alfas):
    #Función para un gráfico donde se muestre la variación de los puntajes de los tres museos con mayor Pagerank para distintos alfas
    #Se muestra la evolución de éstos puntajes al modificarse el alfa
    #Recibe como parámetro un único m y una lista de alfas  
    
    #Creamos un diccionario.
    museosCentrales = {}
    Nprincipales = 3 # Cantidad de museos principales
    for alfa in alfas: #recorremos la lista de las distintas factores de amortiguamiento propuestos
        A = construye_adyacencia(D, m) #en cada caso, la matriz de adyacencia es otra.
        p = calcula_pagerank(A, alfa) #calculamos el pagerank de cada museo en cada caso.
        principales = np.argsort(p.flatten())[-Nprincipales:] # Identificamos a los 3 principales
        for museo in principales:
            #tomamos a cada museo que haya sido principal para algún alfa.
            #lo convertiremos en clave de nuestro diccionario.
            if str(museo) not in museosCentrales: #decisión estética que la clave sea string.
                museosCentrales[str(museo)] = [] #a priori, le asignamos una lista vacía a cada museo principal.
    
    #Ya teniendo todos los museos centrales, la idea es obtener el pagerank de cada uno para cada alfa.
    #No tiene importancia si ese museo no fue "principal" para un determinado alfa.
    for alfa in alfas:
        #la lógica es, para cada factor de amortiguamiento, agarrar los pagerank de cada museo central.
        for museo in museosCentrales:
            A = construye_adyacencia(D, m)
            p = calcula_pagerank(A, alfa)
            museosCentrales[museo].append(p[np.int64(museo)]) #recordar que museo es str.
    
    plt.figure(figsize=(11, 8)) #decisión estética sobre ancho y largo de la imágen.
    
    for museo, pagerank in museosCentrales.items():
        plt.plot(alfas, pagerank, label = museo) #elaboramos un lineplot a partir de la información compilada.
    
    #Detalles de presentación
    plt.title("PageRank por factor de amortiguamiento (α)", fontsize=14)
    plt.xlabel("α [Factor de amortiguamiento]", fontsize=12)
    plt.ylabel("PageRank", fontsize=12)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
    plt.grid(True)
    plt.show()



#%% Punto 5 parte 1: creación matriz de transiciones y matriz B
def calcula_matriz_C_continua(D): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    D[D == 0] = np.inf #aplicamos esto para evitar la división por 0.
    F = 1/D #si D era 0, ahora queda 1 sobre np.inf que es igual a 0.
    np.fill_diagonal(F,0)
    K = construye_matriz_de_grado(F) #Calcula la matriz K, que tiene en su diagonal la suma por filas de F. 
    K_inv = calcular_K_inversa(K) # Calcula inversa de la matriz K. 
    C = F.T @ K_inv # Calcula C multiplicando Kinv y F 
    return C

#------------------------------------------------

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matriz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0]) #B comienza siendo la matriz identidad.
    C_k = C.copy() #inicializamos a C_k como C^1.
    #Ahora realizamos la sumatoria de los C^k. Si solo hubo una única instancia, entonces B será la identidad.
    for i in range(cantidad_de_visitas-1): 
        B += C_k #Adicionamos C^k.
        C_k @= C #Preparamos C^(k+1) en caso de que exista una siguiente iteración.
        # Sumamos las matrices de transición para cada cantidad de pasos.
    return B


#%% Punto 5 - parte 2 : resolución del sistema pedido y cálculo de norma
def resolver_sist (B):
    # Función para resolver el sistema Bv = w; w matriz conocida, proveniente de 'visitas.txt'
    # Recibe como parámetro la matriz B descripta en la ecuación (4) del PDF
    w = np.loadtxt('visitas.txt').T #obtenemos w
    L, U = calculaLU(B) # usamos la descomposición LU, y resolvemos los sistemas
    y = resolver_sist_triang_inf(L, w)
    v = resolver_sist_triang_sup(U, y)
    return v #devuelve el vector v descripto en el punto 4.

#------------------------------------------------

def calcular_norma_1 (v): 
    #Función para calcular la norma_1 de un vector pasado como parámetro
    #La norma-1 es la suma del módulo de cada coordenada del vector
    sumatoria = 0; #inicializamos una variable que guarde las sumas
    for personas in v: #agarramos cada componente del vector para agregarlo a nuestra variable
        sumatoria += abs(personas)
    print(f'La norma 1 del vector v ingresado es: {sumatoria.round()}') #redondeamos, devolviendo un número entero, pues se trata de cantidad de personas
    return 

#%% Punto 6: cálculo de condición
def calcular_inversa (matriz): 
    #Función utilizada para calcular la inversa una matriz pasada como parámetro
    
    I = np.eye(matriz.shape[0])
    L, U = calculaLU(matriz) #agarramos su descomposición LU
    inversa = np.zeros(matriz.shape) #la inicializamos con 0
    for i in range(matriz.shape[0]):
        e = I[:, i] #agarramos en cada iteración un canónico, con los que resolvemos los sistemas
        #aprovechamos la descomposición LU
        y = resolver_sist_triang_inf(L, e)
        x = resolver_sist_triang_sup(U, y)
        inversa[:, i] = x #la solución final la definimos como columna de la inversa
    return inversa #retorna la inversa de la matriz pasada como parámetro



def condicion_1_B (B): 
    #Calcula la condición 1 de la matriz
    
    #cond_1(B) = ||B||_1 * ||B_inv||_1
    B_inv = calcular_inversa(B) #calculamos la inversa de la matriz B
    #Por propiedad, la norma 1 de una matriz es la columna cuya suma de los módulos de sus coeficientes sea mayor
    max_B = 0; #inicializamos la suma de la columna maximal de B_inv como 0
    max_B_inv = 0; #inicializamos la suma de la columna maximal de B_inv como 0
    n = B.shape[0] #B matriz cuadrada, B.shape[0] = B.shape[1] (y B_inv.shape = B.shape)
    for j in range (n): #recorre las columnas
        sumaColsB = 0 #inicializamos la suma de la columna j de B_inv como 0 
        sumaColsBinv = 0 #inicializamos la suma de la columna j de B_inv como 0
        for i in range(n):
            #sumamos los módulos de las filas bajo la columna j
            sumaColsB += np.abs(B[i][j])
            sumaColsBinv += np.abs(B_inv[i][j])
        #comparamos con la que está definida como la columna maximal, y realizamos una redifinición si encontramos una que supere.    
        if sumaColsB > max_B :
            max_B = sumaColsB
        if sumaColsBinv > max_B_inv:
            max_B_inv = sumaColsBinv
            
    #finalmente, multiplicamos las normas_1 de B y B_inv
    condicion_1 = max_B * max_B_inv
    return condicion_1

