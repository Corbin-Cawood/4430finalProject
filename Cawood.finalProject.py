import tensorflow
from tensorflow import lite

oldModel = tensorflow.keras.models.load_model('./combinedModel.h5')
converter = lite.TFLiteConverter.from_keras_model(oldModel)
converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
tfmodel = converter.convert()
f = open('model.tflite', 'wb')
f.write(tfmodel)
f.close()
