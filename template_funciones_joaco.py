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


#%%
def calcula_L(A):
    # La función recibe la matriz de adyacencia A y calcula la matriz laplaciana
    # Have fun!!
    K = construye_matriz_de_grado(A)
    L = K - A
    return L

def calcula_R(A):
    # La funcion recibe la matriz de adyacencia A y calcula la matriz de modularidad
    # Have fun!!
    K = construye_matriz_de_grado(A)
    P = np.zeros(A.shape)
    dos_E = np.sum(K)
    for i in range(A.shape[0]):
        for j in range (A.shape[0]):
            P[i,j] = K[i][i]*K[j][j]/dos_E
    
    R = A - P
    return R

def calcula_s(v):
    #La función recibe un vector v y calcula el vector s, donde cada s_i es 1 si v_1 >0; o -1 en caso contrario.
    s = np.zeros(v.shape())
    for h in range(v.shape[1]):
        if v[0,h] > 0:
            s[0,h] = 1
        else:
            s[0,h] = -1
    
    return s

def calcula_lambda(L,v):
    s = calcula_s(v)
    
    lambdon = 1/4 * s.T @ L @ s
    
    # Recibe L y v y retorna el corte asociado
    # Have fun!
    return lambdon

def calcula_Q(R,v):
    s = calcula_s(v)
    Q = s.T @ R @ s
    
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    return Q

#%%


def metpot1(A,tol=1e-8,maxrep=np.inf):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   v = np.random.uniform(-1, 1, A.shape[0]) # Generamos un vector de partida aleatorio, entre -1 y 1
   v = v = v / np.linalg.norm(v) # Lo normalizamos
   v1 = A @ v # Aplicamos la matriz una vez
   v1 = v1 / np.linalg.norm(v1) # normalizamos
   l = (v.T @ (A @ v)) / (v.T @ v) # Calculamos el autovalor estimado
   l1 = (v1.T @ (A @ v1)) / (v1.T @ v1) # Y el estimado en el siguiente paso
   nrep = 0 # Contador
   while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada 
      v = v1 # actualizamos v y repetimos
      l = l1
      v1 = A @ v # Calculo nuevo v1
      v1 = v1 / np.linalg.norm(v1) # Normalizo
      l1 = (v1.T @ (A @ v1)) / (v1.T @ v1) # Calculo autovalor
      nrep += 1 # Un pasito mas
   if not nrep < maxrep:
      print('MaxRep alcanzado')
   l = (v.T @ (A @ v)) / (v.T @ v) # Calculamos el autovalor
   return v1,l,nrep<maxrep

# %%

def deflaciona(A,tol=1e-8,maxrep=np.inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1,_ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    deflA = A - l1 * np.outer(v1, v1)/(v1.T @ v1) # Sugerencia, usar la funcion outer de numpy
    return v1, l1, deflA



def metpotI(A,mu,tol=1e-8,maxrep=np.inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    M = A + mu * np.eye(A.shape[0])
    M_inv = calcular_inversa(M)
    v_n, autoval_min_inv, _ = metpot1(M_inv, tol, maxrep)
    autoval_min = 1/autoval_min_inv
    
    return v_n, autoval_min


def metpotI2(A,mu,tol=1e-8,maxrep=np.inf):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   X = A + mu*np.eye(A.shape[0]) # Calculamos la matriz A shifteada en mu
   iX = calcular_inversa(X) # La invertimos
   _, _, defliX = deflaciona(iX, tol, maxrep) # La deflacionamos
   v,l,_ =  metpot1(defliX, tol, maxrep) # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu
   return v,l,_


def metpot2(A,v1,l1,tol=1e-8,maxrep=np.inf):
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeors autovectores y autovalores de A}
   # Have fun!
   deflA = A - l1*np.outer(v1, v1)/(v1.T @ v1)
   return metpot1(deflA,tol,maxrep)


#%% CREAMOS L Y R

# Matriz A de ejemplo
A_ejemplo = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0]
])

#Construímos L.
L = calcula_L(A_ejemplo)

#Construímos R.
R = calcula_R(A_ejemplo)

#%%
def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        v,l,_ = metpotI2(L, 1) # Encontramos el segundo autovector de L
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        indices_pos = np.where(v > 0)[0]
        indices_neg = np.where(v < 0)[0]

        Ap = A[indices_pos][:, indices_pos] # Asociado al signo positivo
        Am = A[indices_neg][:, indices_neg] # Asociado al signo negativo
        
        return(
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )        

res = laplaciano_iterativo(A_ejemplo, 2)

#%%
def modularidad_iterativo(A=None,R=None,nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return([nombres_s]) #retorna el unico nodo como comunidad
    else:
        v,l,_ = metpot1(R) # Primer autovector y autovalor de R
       # usamos metpot1 para obtener el autovector principal (asociado al mayor autovalor) de la matriz de modularidad R
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0]) # suma de conexiones dentro de las comunidades propuestas por el autovector v
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return([nombres_s]) #retornamos todos los nodos como una sola comunidad
        else:
            # partición de R según signos del autovector
            indices_pos = np.where(v > 0)[0]
            indices_neg = np.where(v < 0)[0]
            
            # Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            # dividimos a R en dos submatrices segun el signo de v
            Rp = R[indices_pos][:, indices_pos] # Parte de R asociada a los valores positivos de v
            Rm = R[indices_neg][:, indices_neg] # Parte asociada a los valores negativos de v
            # calculamos los autovectores principales de las submatrices para evaluar particiones mas finas
            vp,lp,_ = metpot1(Rp)  # autovector principal de Rp
            vm,lm,_ = metpot1(Rm) # autovector principal de Rm
        
            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 += np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # llamadas recursivas para cada particion
                comunidades_pos = modularidad_iterativo(R=Rp, nombres_s=[ni for ni, vi in zip(nombres_s, v) if vi > 0])
                comunidades_neg = modularidad_iterativo(R=Rm, nombres_s=[ni for ni, vi in zip(nombres_s, v) if vi < 0])
                
                # Sino, repetimos para los subniveles
                return(comunidades_pos + comunidades_neg)


res2 = modularidad_iterativo(A_ejemplo, R)


#%% FUNCIÓN GRAFICAR COMUNIDADES

def graficar_comunidades(A, comunidades, ax):
    #La función recibe la matriz de adyacencia, una segmentación de comunidades y una posición en una figura a mostrar.
    #A partir de estos datos, grafica los museos en el mapa diferenciando por color las distintas segmentaciones.
    
    #Primero nos encargamos de pasar los nodos a un diccionario.
    #Este diccionario contendrá el número de comunidad a la que pertenece.
    #Como la variable 'comunidades' es una lista de listas, el número de comunidad lo determina la posición de la sublista en donde está el museo.
    nodo_a_comunidad = {}
    for idx, comunidad in enumerate(comunidades):
        for nodo in comunidad:
            nodo_a_comunidad[nodo] = idx



    # Construímos el gráfico a partir de la matriz de adyacencia
    G = nx.from_numpy_array(A)
    # Construimos un layout a partir de las coordenadas geográficas
    G_layout = {i: v for i, v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'], museos.to_crs("EPSG:22184").get_coordinates()['y']))}
    factor_escala = 400  # Escalamos los nodos 400 veces para que sean bien visibles
    
    # Crear figuras y graficar barrios
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)  # Graficamos los barrios
    

    # Extraemos las comunidades (en orden para consistencia de clases)
    comunidades_unicas = sorted(set(nodo_a_comunidad.values()))
    
    # Seleccionamos una cantidad de colores en base al número de comunidades.
    cmap = cm.get_cmap('tab20', len(comunidades_unicas))
    
    # Creamos un diccionario que asocia cada número de comunidad a un color.
    color_por_comunidad = {com: cmap(i) for i, com in enumerate(comunidades_unicas)}
    
    # Nos armamos la lista de colores de los museos en función del segmento al que pertenecen.
    node_colors = [color_por_comunidad[nodo_a_comunidad[nodo]] for nodo in G.nodes()]
    
    # Finalmente graficamos los nodos con los coloreados asignados en función de su comunidad.
    nx.draw_networkx_nodes(G, G_layout, node_color=node_colors, node_size=factor_escala, ax=ax)
    
    #Agregamos las etiquetas a los nodos
    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, G_layout, labels=labels, font_size=6, font_color="k", ax=ax)
    
    return ax
    
#%% APLICAMOS LOS MÉTODOS VISTOS EN NUESTRA MATRIZ DE MUSEOS

#Seleccionamos cantidades de enlaces de los museos 
lista_m = [3,5,10,50]

for m in lista_m:
    #Construímos la matriz de adyacencia
    A = construye_adyacencia(D, m)
    #La simetrizamos
    A_simetrica = np.ceil(1/2*(A +A.T))

    #MODULARIDAD
    R = calcula_R(A_simetrica)
    segmentacion_con_modularidad = modularidad_iterativo(A_simetrica, R)
    #print(f'Las particiones con el método modularidad con m= {m} es {segmentacion_con_modularidad}')
    
    
    #LAPLACIANO
    #Como 'cantidad óptima de niveles', tomamos el logaritmo en base dos de la cantidad de comunidades generadas con el método de modularidad
    niveles_optimos_laplacianos = int(math.log2(len(segmentacion_con_modularidad)))
    segmentacion_con_laplaciano = laplaciano_iterativo(A_simetrica, niveles_optimos_laplacianos)
    print(f'Las particiones con el método corte mínimo con {niveles_optimos_laplacianos} cortes con m = {m} es {segmentacion_con_laplaciano}')
    
    
    #GRAFICAMOS DISPOSICIÓN DE COMUNIDADES EN BASE A AMBOS MÉTODOS
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))  # Dos gráficos en fila
    
    graficar_comunidades(A,segmentacion_con_modularidad, ax[0])
    #Detalles de presentación
    ax[0].set_title("Segmentación en comunidades con el método de modularidad", fontsize=14)

    graficar_comunidades(A,segmentacion_con_laplaciano, ax[1])
    #Detalles de presentación
    ax[1].set_title(f"Segmentación en comunidades con el método del corte mínimo con {str(niveles_optimos_laplacianos)} niveles", fontsize=14)
    fig.text(0.5, 0.08, f'La matriz de adyacencia fue generada con {str(m)} enlaces por museo', ha='center', fontsize=12)
    plt.show()
    
    
#En ambas, a medida que aumentamos la cantidad de conexiones en la matriz de adyacencia, baja la cantidad de segmentaciones realizadas.
#En esta línea, a medida que aumenta la cantidad de conexiones, las comunidades formadas en ambos métodos van siendo más similares entre sí.
#Es decir, realizamos subdivisiones similares
#Las comunidades más grandes suelen ser las formadas a partir de los museos que se encuentran por fuera de la metrópolis
#Es decir, aquellos que no se encuentran en centros de aglomeración de museos.
#Paralelamente, los sectores donde más cantidad de comunidades se generan son aquellos focos centrales/más densos de museos.








