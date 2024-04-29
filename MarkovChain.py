import numpy as np
import pickle
from math import fabs

def getMarkovHue(imgs_arr, sensibility = 5,scale=2):
    pxs = []
    
    for img in imgs_arr:
        pxs.append( img.load())
        
    # video size
    width, height = imgs_arr[0].size
    
    width = int(width/2)
    height = int(height/2)
    
    # Define the sizes
    outer_rows = height
    outer_cols = width
    inner_rows = 36
    inner_cols = 1

    # Create an empty 3x3 matrix
    outer_matrix = np.empty((outer_rows, outer_cols), dtype=object)

    # Fill each cell with a 2x2 matrix of zeros
    for i in range(outer_rows):
        for j in range(outer_cols):
            outer_matrix[i, j] = [0]*36

    img_count = 0
    # Generate mov shadow
    for image_index in range(0, len(pxs)-1):
        px1 = pxs[image_index]
        px2 = pxs[image_index+1]
        
        for y in range(0, int(height)):
            for x in range(0, int(width)):
                updateMarkovChain(px1,px2,x*2,y*2, outer_matrix ,sensibility)
        img_count += 1
        
        pxs[image_index] = None
        
        print("imagen finalizada:" + str(img_count) + "/" + str(len(imgs_arr)-1))
    return  outer_matrix 

def updateMarkovChain(px1,px2,x,y,array,sensibility):
    current_hue = int(rgb2Hue(px1[x,y][0], px1[x,y][1], px1[x,y][2]))    

    array[int(y/2)][int(x/2)][ int(current_hue/10)] += 1
    if( int(current_hue/10) < 35):
        array[int(y/2)][int(x/2)][int(current_hue/10) + 1] += 0.5
    else:
        array[int(y/2)][int(x/2)][0] += 0.5
    if( int(current_hue/10) > 0):
        array[int(y/2)][int(x/2)][int(current_hue/10) - 1] += 0.5
    else:
        array[int(y/2)][int(x/2)][35] += 0.5

def saveMarkovChain(array,name):
    with open(name, "wb") as file:
        pickle.dump(array, file)

def loadMarkovChain(filename):
    # Cargar la matriz desde el archivo
    with open(filename, "rb") as file:
        loaded_matrix = pickle.load(file)
        # Visualizar la matriz
        return(loaded_matrix)
  
    
def getShadowByHue(imgs_arr, markovChain, sensibility = 1,scale=2, ):
    pxs = []
    for img in imgs_arr:
        pxs.append( img.load())
        
    # video size
    width, height = imgs_arr[0].size

    width = int(width/2)
    height = int(height/2)
    
    array = np.zeros(int(height*width), dtype=int)
    # Reshape the array into a 
    # familiar resoluition
    array = np.reshape(array, (height,width))
    
    img_count = 0
    # Generate mov shadow
    for image_index in range(0, len(pxs)-1):
        px1 = pxs[image_index]
        px2 = pxs[image_index+1]
        for y in range(0, height):
            for x in range(0, width):
                updateTempDifference(px1,px2,x*2,y*2,array,sensibility, markovChain)
        img_count += 1
        
        print("imagen finalizada:" + str(img_count) + "/" + str(len(imgs_arr)-1))
    return array

def updateTempDifference(px1,px2,x,y,array,sensibility, markovChain):
    difference = 0
    current_hue = int( rgb2Hue(px1[x,y][0], px1[x,y][1], px1[x,y][2]) )
    next_hue = int(rgb2Hue(px2[x,y][0], px2[x,y][1], px2[x,y][2]) )
    difference += abs( current_hue - next_hue)
    
    if(current_hue == 0 or next_hue == 0):
        return

    difference2 = 0
    if(current_hue < next_hue):
        difference2 = current_hue + (360 - next_hue)
    else:
        difference2 = next_hue + (360 - current_hue) 
        
    difference = min(difference,difference2)
    
    odds = markovChain[int(y/2)][int(x/2)][int(next_hue/10)]
        
    if(int(difference)> sensibility):
        if(markovChain[int(y/2),int(x/2)][int(next_hue/10)] < 8 ):
            array[int(y/2),int(x/2)] += 1

            
def rgb2Hue(r,g,b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    return h