from stump import pred_stump
import cv2
def get(image):
    res=pred_stump(image)
    x=res['predictions'][0]['x']
    y=res['predictions'][0]['y']
    w=res['predictions'][0]['width']
    h=res['predictions'][0]['height']
    x_stump = x + w//2
    y_stump = y + h/2 + h/4
    w_stump = w//2
    h_stump = 10    
    cv2.rectangle(image, (int(x_stump - w_stump/2), int(y_stump - h_stump/2)), (int(x_stump + w_stump/2), int(y_stump + h_stump/2)), (0, 250, 0), -1)
    return image

