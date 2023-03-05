import os
import cv2
from det import detect
from converttoframes import convert_to_frames
from mask import maskimg
from predict_trajectory import predict_trajectory
from plot import plot_trajectory_as_mask
from tqdm import tqdm
from processstump import get
import os
from generate import gen
def process(video):
    path1 ="frames"
    path2="masked"
    path3="trajectory"
# Check whether the specified path exists or not
    isExist = os.path.exists(path1)
    if not isExist:
        os.makedirs(path1)
    isExist = os.path.exists(path2)
    if not isExist:
        os.makedirs(path2)
    isExist = os.path.exists(path3)
    if not isExist:
        os.makedirs(path3)
    convert_to_frames(video)
# Define the path to the directory containing the images
    dir_path = 'frames'
    prev=[]
# Get a list of all image file names in the directory
    img_files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
   # print(img_files)
# Loop over each image file name and load the image
    for img_file in tqdm(img_files):
    # Construct the full file path
        file_path = os.path.join(dir_path, img_file)
    # Load the image
        img = cv2.imread(file_path)
        res=detect(img)
        if res['predictions']==[]:
            cv2.imwrite('masked/'+img_file,img)
            cv2.imwrite('trajectory/'+img_file,img)
        else:
            if res['predictions'][0]['class']=='cricket-ball':
                x,y,w,h=res['predictions'][0]['x'],res['predictions'][0]['y'],res['predictions'][0]['width'],res['predictions'][0]['height']
                masked_image=maskimg(img,x,y,w,h)
                if prev is None:
                    prev=[[int(x+w//2),int(y+h//2)]]
                else:
                    prev=prev.append([x,y,w,h])
                result=predict_trajectory(prev)
                maskstump=get(masked_image)
                plot=plot_trajectory_as_mask(masked_image,result)
                cv2.imwrite('masked/'+img_file,masked_image[:,:,::-1])
                cv2.imwrite('trajectory/'+img_file,plot[:,:,::-1])
            else:
                cv2.imwrite('masked/'+img_file,img)
                cv2.imwrite('trajectory/'+img_file,img)
    gen(video)

        
    
