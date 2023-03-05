import cv2
# Load image and bounding box coordinates
def maskimg(image,x,y,w,h):
# Create a mask for the ball
    mask = cv2.circle(
        img = image,
        center = (int(x + w//2), int(y + h//2)),
        radius = int(w//2),
        color = (0, 0,255),
        thickness = -1
    )
# Apply the mask to the image
    #masked_image = cv2.bitwise_and(image, mask)
    return mask