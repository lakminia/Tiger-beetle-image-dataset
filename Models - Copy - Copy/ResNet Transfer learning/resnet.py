import numpy as np
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
#from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
from keras.applications import ResNet50
from keras.applications.resnet50  import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers



TRAIN_DIR = 'Training_data'
TEST_DIR = 'Testing_data'

CLASSES = 9
    
# setup model
base_model = ResNet50(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.5)(x)
predictions = Dense(CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
   
# transfer learning
'''
for layer in base_model.layers:
    layer.trainable = False
'''

for layer in model.layers[:20]:
    layer.trainable = False    
#op=optimizers.RMSprop(lr=0.0001)            
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])




WIDTH = 224
HEIGHT = 224
BATCH_SIZE = 16

# data prep
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(HEIGHT, WIDTH),
		batch_size=BATCH_SIZE,
		class_mode='categorical')
    
validation_generator = validation_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')


x_batch, y_batch = next(train_generator)

plt.figure(figsize=(12, 9))
for k, (img, lbl) in enumerate(zip(x_batch, y_batch)):
    plt.subplot(4, 8, k+1)
    plt.imshow((img + 1) / 2)
    plt.axis('off')



EPOCHS = 100
BATCH_SIZE = 16
STEPS_PER_EPOCH = 160
#STEPS_PER_EPOCH = 10
VALIDATION_STEPS = 32

MODEL_FILE = 'resnet_.model'
top_model_weights_path = 'resnet_.model'

# Set callback functions to early stop training and save the best model so far


    
checkpoint = ModelCheckpoint(filepath=top_model_weights_path,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            mode='max')    

history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,callbacks=[checkpoint])
  
model.save(MODEL_FILE)    


def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))
  plt.figure(1)

  plt.subplot(211)  
  plt.title('Training and validation accuracy')
  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  
  
 # plt.figure()
  plt.subplot(212)
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()
  
plot_training(history)







