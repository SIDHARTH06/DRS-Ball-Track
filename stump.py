from roboflow import Roboflow
rf = Roboflow(api_key="PYIeBdrA2u4Z7c8wNi3G")
project = rf.workspace().project("batsman-detection-ozpnz")
def pred_stump(image):
    model = project.version(1).model
    return model.predict(image, confidence=40, overlap=30).json()

