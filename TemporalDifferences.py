import numpy as np
from math import fabs

def getShadowByRGB(imgs_arr, sensibility = 10):
    pxs = []
    for img in imgs_arr:
        pxs.append( img.load())
        
    # video size
    width, height = imgs[0].size

    array = np.zeros((height*width), dtype=int)
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
                difference = 0
                for color in range(0,3):
                    difference += fabs( px1[x,y][color] - px2[x,y][color]  )
                if(int(difference)> SENSIBILITY):
                    array[y,x] += 512
        img_count += 1
        print("imagen finalizada:" + str(img_count) + "/" + str(len(imgs)-1))

    array = np.dot(10,array)
    np.save('usandoHue.png',array)

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

def getShadowByHue(imgs_arr, sensibility = 5,scale=2):
    pxs = []
    for img in imgs_arr:
        pxs.append( img.load())
        
    # video size
    width, height = imgs_arr[0].size

    width = int(width / 2)
    height = int(height / 2)
    
    array = np.zeros( height*width, dtype=int)
    # Reshape the array into a 
    # familiar resoluition
    array = np.reshape(array, (height,width))

    img_count = 0
    # Generate mov shadow
    for image_index in range(0, len(pxs)-2):
        px1 = pxs[image_index]
        px2 = pxs[image_index+1]
        for y in range(0, int(height)):
            for x in range(0, int(width)):
                updateTempDifference(px1,px2,x*2,y*2,array,sensibility)
        img_count += 1
        
        print("imagen finalizada:" + str(img_count) + "/" + str(len(imgs_arr)-1))
    return array

def updateTempDifference(px1,px2,x,y,array,sensibility):
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
    
    
    if(int(difference)> sensibility):
        array[int(y/2),int(x/2)] += 1
        
