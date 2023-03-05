from mask import maskimg
from det import detect
import cv2
import matplotlib.pyplot as plt
img=cv2.imread('frames/305.png')
res=detect(img)
if res['predictions'][0]['class']=='cricket-ball':
    x,y,w,h=res['predictions'][0]['x'],res['predictions'][0]['y'],res['predictions'][0]['width'],res['predictions'][0]['height']
    masked_image=maskimg(img[:,:,::-1],x,y,w,h)
    #plt.imshow(masked_image)
    plt.show()