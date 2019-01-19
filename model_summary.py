from keras.models import load_model

model = load_model('model.h5')
print(model.summary())
print('Conv2d Layer 1 Dropout rate', model.layers[3].rate)
print('Conv2d Layer 2 Dropout rate', model.layers[5].rate)
print('Conv2d Layer 3 Dropout rate', model.layers[7].rate)
print('Conv2d Layer 4 Dropout rate', model.layers[9].rate)
print('Conv2d Layer 5 Dropout rate', model.layers[11].rate)
print('Dense Layer 1 Dropout rate', model.layers[14].rate)
print('Dense Layer 2 Dropout rate', model.layers[16].rate)
