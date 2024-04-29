# Importing Image from PIL package
from PIL import Image as im
import numpy as np

import os
import matplotlib.pyplot as plt
from scipy import stats

import StatisticBackground
import TemporalDifferences
import MarkovChain

##### CREATING FRAMES FROM A VIDEO
# Importing all necessary libraries
import cv2
import os

ROUTE_VIDEO="video\HDV_0801.MP4"
ROUTE_FOTOGRAMS="fotogramas_completo"
FOTOGRAMS_PER_SECOND = 1
PROJECT_NAME = "PruebaDeploy"
MAX_GROUPS = 6
STARTING_FRAME = 470
ENDING_FRAME = 485

def video2fotograms(route_video,route_fotograms):
    # Read the video from specified path
    cam = cv2.VideoCapture(route_video)

    try:
        
        # creating a folder named data
        if not os.path.exists(route_fotograms):
            os.makedirs(route_fotograms)
    
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
    
    fps = FOTOGRAMS_PER_SECOND
    currentframe = 0
    counter = 0
    while(True):
        counter+=1
        ret,frame = cam.read()
        if counter > 30/fps:
            # reading from frame
            if ret:
                # if video is still left continue creating images
                name = route_fotograms+ '/frame' + str(currentframe) + '.jpg'
                print ('Creating...' + name)
        
                # writing the extracted images
                cv2.imwrite(name, frame)
        
                # increasing counter so that it will
                # show how many frames are created
                currentframe += counter
                counter = 0
            else:
                break
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


# video2fotograms(route_video=ROUTE_VIDEO, route_fotograms=ROUTE_FOTOGRAMS)

##### GENERATING DIFFERENCE MATRIX
archivos = os.listdir(ROUTE_FOTOGRAMS)
imgs = []

for filename in archivos:
    imgs.append( im.open(ROUTE_FOTOGRAMS+"/"+filename))
 
if ENDING_FRAME:
    imgs= imgs[STARTING_FRAME:ENDING_FRAME] 
    
if not os.path.exists(PROJECT_NAME):
    os.mkdir(f"{PROJECT_NAME}")  
elif not any(os.scandir(PROJECT_NAME)):
    pass    
else:
    raise(Exception("Proyecto en Uso"))  

print("obteniendo fondo...")
chain = MarkovChain.getMarkovHue(imgs)
MarkovChain.saveMarkovChain(chain,"CadenaLarga")

markov = MarkovChain.loadMarkovChain("CadenaLarga")
print("obteniendo sombra...")
shadow = MarkovChain.getShadowByHue(imgs,markov, sensibility=15)

plt.imsave(f"{PROJECT_NAME}/differences8.jpg", shadow , cmap='gray')
# shadow = np.save(f"{PROJECT_NAME}/differences6.npy")
np.save(f"{PROJECT_NAME}/differences8",shadow)


# shadow = TemporalDifferences.getShadowByHue(imgs, sensibility = 20)


shadow = stats.zscore(shadow)
plt.imsave(f"{PROJECT_NAME}/Normalized.jpg", shadow , cmap='gray')

def getCommonElements(np_arr,_max=1):
    for y in range(0, len(np_arr)):
        for x in range(0, len(np_arr[0])):
            if(np_arr[y][x] < _max):
                np_arr[y][x] = 0
            elif(np_arr[y][x] >= _max*2):
                np_arr[y][x] = 2
            elif(np_arr[y][x] >= _max):
                np_arr[y][x] = 1
    return np_arr
            
# GETTING ONLY MOST REPRESENTATIVE VALUES
shadow = getCommonElements(shadow, _max=1)
np.save(f"{PROJECT_NAME}/MiniNormalized",shadow)
plt.imsave(f"{PROJECT_NAME}/MiniNormalized.jpg", shadow , cmap='gray')


# Delete all with less than n neightbours
def deleteLonely(np_arr, min_neightbours = 16, _range = 2):
    futures_vals = np.full_like(np_arr, 0)
    for y in range(0, len(np_arr)):
        for x in range(0, len(np_arr[0])):
            neightbours = np_arr[y][x]
            for m in range(-(_range),_range+1):
                for n in range(-(_range),_range+1):
                    if(n == 0 and m == 0 ):
                        continue
                    try:
                        if(np_arr[y+m*2][x+n*2] > 0):
                            neightbours += 1
                    except IndexError:
                        neightbours += 0
            if(neightbours>=min_neightbours):
                futures_vals[y][x] = 1
            else:
                futures_vals[y][x] = 0
    return futures_vals
                
# shadow = np.load("MiniNormalizedFullShadowNaranjaAltaSensibilidad.npy")
shadow = deleteLonely(shadow,min_neightbours=18, _range=2)
shadow = deleteLonely(shadow,min_neightbours=26, _range=3)

plt.imsave(f"{PROJECT_NAME}/Lonely.jpg", shadow , cmap='gray')
# np.save("LonelyMiniNormalizedFullShadowNaranjaAltaSensibilidad.npy", shadow)


def joinIslands(np_arr, max_gap=4):
    for lap in range(0, max_gap):
        for y in range(0, len(np_arr)):
            for x in range(0, len(np_arr[0])):
                try:
                    if(np_arr[y][x-(max_gap-lap)*2]>0 and np_arr[y][x+(max_gap-lap)*2]>0):
                        np_arr[y][x]=1
                except IndexError:
                    pass
                
    for lap in range(0, max_gap):
        for y in range(0, len(np_arr)):
            for x in range(0, len(np_arr[0])):
                try:
                    if(np_arr[y-(max_gap-lap)*2][x]>0 and np_arr[y+(max_gap-lap)*2][x]>0):
                        np_arr[y][x]=1
                except IndexError:
                    pass   
    
    for lap in range(0, max_gap):
        for y in range(0, len(np_arr)):
            for x in range(0, len(np_arr[0])):
                try:
                    if(np_arr[y-(max_gap-lap)*2][x-(max_gap-lap)*2]>0 and np_arr[y+(max_gap-lap)*2][x+(max_gap-lap)*2]>0):
                        np_arr[y][x]=1
                except IndexError:
                    pass       

    for lap in range(0, max_gap):
        for y in range(0, len(np_arr)):
            for x in range(0, len(np_arr[0])):
                try:
                    if(np_arr[y+(max_gap-lap)*2][x-(max_gap-lap)*2]>0 and np_arr[y-(max_gap-lap)*2][x+(max_gap-lap)*2]>0):
                        np_arr[y][x]=1
                except IndexError:
                    pass  
    
    return np_arr     
      

# shadow = np.load("LonelyMiniNormalizedFullShadowNaranjaAltaSensibilidad.npy")
shadow= joinIslands(shadow,max_gap=7)
plt.imsave(f"{PROJECT_NAME}/Joined.jpg", shadow , cmap='gray')
# np.save("JoinedLonelyMiniNormalizedFullShadowNaranjaAltaSensibilidad.npy", shadow)


# shadow = np.load("JoinedLonelyMiniNormalizedFullShadowNaranjaAltaSensibilidad.npy")

def floodFill8Directions(matriz):
    filas = len(matriz)
    columnas = len(matriz[0])
    valor_actual = 2

    def get_neighbors(i, j):
        neighbors = []
        if i > 0:
            neighbors.append((i - 1, j))
            if j > 0:
                neighbors.append((i - 1, j - 1))
            if j < columnas - 1:
                neighbors.append((i - 1, j + 1))
        if i < filas - 1:
            neighbors.append((i + 1, j))
            if j > 0:
                neighbors.append((i + 1, j - 1))
            if j < columnas - 1:
                neighbors.append((i + 1, j + 1))
        if j > 0:
            neighbors.append((i, j - 1))
        if j < columnas - 1:
            neighbors.append((i, j + 1))
        return neighbors

    for i in range(filas):
        for j in range(columnas):
            if matriz[i][j] == 1:
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if matriz[x][y] == 1:
                        matriz[x][y] = valor_actual
                        neighbors = get_neighbors(x, y)
                        stack.extend(neighbors)
                valor_actual += 1
    
    return matriz

shadow = floodFill8Directions(shadow)

plt.imsave(f"{PROJECT_NAME}/FloodFill.jpg", shadow  )
# np.save("FloodFillJoinedLonelyMiniNormalizedFullShadowNaranjaAltaSensibilidad.npy", shadow)

# shadow = np.load("FloodFillJoinedLonelyMiniNormalizedFullShadowNaranjaAltaSensibilidad.npy")

# contabilizar cada isla

# DELETING SMALLEST ISLAND
def deleteSmallestIslands(matrix,max_groups=5 ): 
   
    def island_count(matrix):
        counter = {}
        filas = len(matrix)
        columnas = len(matrix[0])
        
        for y in range(0,filas):
            for x in range(0,columnas):
                if(matrix[y][x] in counter):
                    counter[int(matrix[y][x])] +=1
                else:  
                    counter[int(matrix[y][x])] = 1
        
        return(counter)
            
    def getListWithGroups(counter,matrix):
        sorted_dict = sorted(counter.values())
        sorted_dict= sorted_dict[-max_groups-1:]

        for island in counter.keys():
            if(counter[island] in sorted_dict and island != 0):
                print (island)
        
        return sorted_dict

    counter  = island_count(shadow)
    sorted_dict = getListWithGroups(counter = counter, matrix = shadow)

    filas = len(matrix)
    columnas = len(matrix[0])
    for y in range(0,filas):
        for x in range(0,columnas):
            if(not counter[matrix[y][x]] in sorted_dict):
                matrix[y][x] = 0
    
    return matrix


shadow = deleteSmallestIslands(max_groups=MAX_GROUPS, matrix=shadow)
plt.imsave(f"{PROJECT_NAME}/BiggestIslands.jpg", shadow , )
# np.save("BiggestIslandsFloodFillJoinedLonelyMiniNormalizedFullShadowNaranjaAltaSensibilidad.npy", shadow)

# shadow = np.load("BiggestIslandsFloodFillJoinedLonelyMiniNormalizedFullShadowNaranjaAltaSensibilidad.npy")


# Función para dividir la matriz en sectores
def dividir_en_sectores(matriz, n_horizontal, n_vertical):
    matrix_final = np.zeros((n_vertical, n_horizontal))
    ancho_horizontal = int(len(matriz[0])/n_horizontal)
    ancho_vertical = int(len(matriz)/n_vertical)
    for sector_vertical in range(0,n_vertical):
        for sector_horizontal in range(0,n_horizontal):
            counter = {}
            for y in range(0,ancho_vertical):
                for x in range(0, ancho_horizontal):
                    posx = x+ (sector_horizontal * ancho_horizontal)
                    posy = y+ (sector_vertical * ancho_vertical)
                    if(matriz[posy][posx] != 0):
                        if(matriz[posy][posx] in counter):
                            counter[int(matriz[posy][posx])] +=1
                        else:  
                            counter[int(matriz[posy][posx])] = 1
                    matriz[posy][posx]= sector_vertical
            max_ = 0
            valores = list(counter.values())        
            keys = list(counter.keys())
            if(len(valores) > 0 and len(keys) > 0):    
                max_ = keys[valores.index(max(valores))]
            
            matrix_final[sector_vertical][sector_horizontal] = max_
    return(matrix_final)
                
                    
                            
shadow = dividir_en_sectores(shadow,9,16)
np.save(f"{PROJECT_NAME}/sectores.npy", shadow)

  
# Convertir a lista de listas
# Construir las listas de tuplas separadas por valor del elemento
def shadow2list(matrix_final):
    indices = np.nonzero(matrix_final)
    listas_por_valor = {}
    for x, y in zip(indices[0], indices[1]):
        val = matrix_final[x, y]
        if val not in listas_por_valor:
            listas_por_valor[val] = []
        listas_por_valor[val].append((x, y))

    # Crear las variables para cada lista de tuplas
    for val, lista in listas_por_valor.items():
        locals()['lista' + str(val)] = lista
        
    # Imprimir todas las listas de tuplas
    for val, lista in listas_por_valor.items():
        print("lista{} = {}".format(int(val), lista))

shadow2list(shadow)

