import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import geopandas as gpd # Para hacer cosas geográficas
import networkx as nx # Construcción de la red en NetworkX
import math
import template_funciones as tf1
#%%
museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')
#%%
# En esta línea:
# Tomamos museos, lo convertimos al sistema de coordenadas de interés, extraemos su geometría (los puntos del mapa), 
# calculamos sus distancias a los otros puntos de df, redondeamos (obteniendo distancia en metros), y lo convertimos a un array 2D de numpy
D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()

#%% BLOQUE DE CONSTRUCCIÓN DEL LAPLACIANO Y MATRIZ DE MODULARIDAD

def calcula_L(A):
    # La función recibe la matriz de adyacencia A y calcula la matriz laplaciana
    # Have fun!!
    K = tf1.construye_matriz_de_grado(A)
    L = K - A
    return L

def calcula_R(A):
    # La funcion recibe la matriz de adyacencia A y calcula la matriz de modularidad
    # Have fun!!
    K = tf1.construye_matriz_de_grado(A)
    P = np.zeros(A.shape)
    dos_E = np.sum(K)
    for i in range(A.shape[0]):
        for j in range (A.shape[0]):
            P[i,j] = K[i][i]*K[j][j]/dos_E
    
    R = A - P
    return R

def calcula_s(v):
    #La función recibe un vector v y calcula el vector s, donde cada s_i es 1 si v_1 >0; o -1 en caso contrario.
    s = np.zeros(v.shape)
    for h in range(v.shape[0]):
        if v[h] > 0:
            s[h] = 1
        else:
            s[h] = -1
    
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


#%% FUNCIONES MÉTODO DE LA POTENCIA Y DEFLACIÓN

def metpot1(A,tol=1e-8,maxrep=np.inf):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   np.random.seed(42)
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


def deflaciona(A,tol=1e-8,maxrep=np.inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1,_ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    deflA = A - l1 * np.outer(v1, v1)/(v1.T @ v1) # Sugerencia, usar la funcion outer de numpy
    return v1, l1, deflA



def metpotI(A,mu,tol=1e-8,maxrep=np.inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    M = A + mu * np.eye(A.shape[0])
    M_inv = tf1.calcular_inversa(M)
    
    return metpot1(M_inv, tol, maxrep)


def metpotI2(A,mu,tol=1e-8,maxrep=np.inf):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   X = A + mu*np.eye(A.shape[0]) # Calculamos la matriz A shifteada en mu
   iX = tf1.calcular_inversa(X) # La invertimos
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


#%% BLOQUE LAPLACIANO Y MODULARIDAD ITERATIVOS

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


#%% FUNCIÓN PARA GRAFICAR COMUNIDADES SEGMENTADAS POR LOS MÉTODOS 

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
    G_layout = {i: v for i, v in enumerate(zip(tf1.museos.to_crs("EPSG:22184").get_coordinates()['x'], museos.to_crs("EPSG:22184").get_coordinates()['y']))}
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
    
