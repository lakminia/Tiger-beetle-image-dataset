import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras import metrics
import model

#Start
#top_model_weights_path = 'bottleneck_fc_model_confusion.h5'
train_data_path = 'Training_data'
test_data_path = 'Testing_data'
img_rows = 224
img_cols = 224
epochs = 200
batch_size = 8
num_of_train_samples = 900
num_of_test_samples = 315
#621/130

def top_5_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)
def top_1_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)
def top_2_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)
def top_3_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


#Image Generator
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_path,
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
np.save('class_indices_confusion_1.npy', train_generator.class_indices)


validation_generator = test_datagen.flow_from_directory(test_data_path,
                                                        target_size=(img_rows, img_cols),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

model = model.SqueezeNet(nb_classes=9, input_shape=(img_rows, img_cols,3))
# Build model


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',top_1_accuracy,top_2_accuracy,top_3_accuracy,top_5_accuracy])
model.summary()


'''
checkpoint = ModelCheckpoint(filepath=top_model_weights_path,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            mode='max')

    
train_data = train_data.astype('float32')
validation_data = validation_data.astype('float32')
train_data /= 255
validation_data /= 255    

   
history = model.fit(train_data, 
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=validation_data,shuffle=True,callbacks=[checkpoint])
print(history)

model.save_weights(top_model_weights_path)

(eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))

plt.figure(1)

    # summarize history for accuracy

plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




'''
#Train
top_model_weights_path = 'bottleneck_fc_confusion_model_2.h5'
checkpoint = ModelCheckpoint(filepath=top_model_weights_path,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            mode='max')
history=model.fit_generator(train_generator,
                    steps_per_epoch=num_of_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=num_of_test_samples // batch_size,
                            callbacks=[checkpoint])

model.save(top_model_weights_path)


acc = history.history['acc']
val_acc = history.history['val_acc']
val_acc_top_5=history.history['val_top_5_accuracy']
val_acc_top_1=history.history['val_top_1_accuracy']
val_acc_top_2=history.history['val_top_2_accuracy']
val_acc_top_3=history.history['val_top_3_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc,  label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.plot(epochs, val_acc_top_5,  label='Validation acc_top_5')
plt.plot(epochs, val_acc_top_1,  label='Validation acc_top_1')
plt.plot(epochs, val_acc_top_2,  label='Validation acc_top_2')
plt.plot(epochs, val_acc_top_3,  label='Validation acc_top_3')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss,  label='Training loss')
plt.plot(epochs, val_loss,  label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()





#Confution Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(validation_generator.classes)
print('sss')
print(y_pred)
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['callytron','calomera','Cylindera','derocrania','Hypathea','lophyra','myriochela','neocollyris','tricondyla']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

