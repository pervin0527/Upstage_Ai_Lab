import keras
import numpy as np

from PIL.Image import Image
from model_loader import model

def predict(image: Image):
    image = np.asarray(Image.resize(image, (224, 224)))[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5 - 1.0

    results = keras.applications.imagenet_utils.decode_predictions(model.predict(image), top=3)[0]
    print(results)

    ## API 생성
    result_list = []
    for i in results:
        result_list.append({
            'class' : i[1],
            'confidence' : f'{i[2] * 100:.2f}%'
        })

    return result_list