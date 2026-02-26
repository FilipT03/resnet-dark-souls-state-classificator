import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os, sys
import json

DATASET_PATH = '../dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16 

EPOCHS_TOP = 10      
EPOCHS_FINE_TUNE = 100

def main(evaulation_only = False):

    # First, we need to prepare the data for the model. ImageDataGenerator allows us to automatically create everything needed for training.
    # Here we also define data augmentation, which expends the dataset by modifiying images in various ways.
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, 
        rotation_range=30,      
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # No shuffling for validation to keep the class order
    val_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50
    # We don't need the top layer since those are specific to the data ResNet is trained on, and we want to train on our data
    # Other layers contain the feature identification, which is still helpful 
    # Transfer-learning workflow inspiration: https://www.tensorflow.org/guide/keras/transfer_learning#the_typical_transfer-learning_workflow
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    predictions = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    history = None
    if not evaulation_only:
        print("\nTop layers training:")
        # https://www.tensorflow.org/guide/keras/transfer_learning#train_the_top_layer
        model.compile(optimizer=Adam(learning_rate=1e-3),
                        loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=[tf.keras.metrics.CategoricalAccuracy()])

        history_top = model.fit(train_generator, epochs=EPOCHS_TOP, validation_data=val_generator)

        print("\nFine tuning:")
        # We will keep 100 layers frozen to make the model more stable. Since those include the more basic features, it they will be useful as is.
        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False

        # https://www.tensorflow.org/guide/keras/transfer_learning#do_a_round_of_fine-tuning_of_the_entire_model
        model.compile(optimizer=Adam(learning_rate=1e-5), 
                        loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=[tf.keras.metrics.CategoricalAccuracy()])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1), # stop model if no progress
            ModelCheckpoint('best_darksouls_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1), # make checkpoints
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1) # reduce learning rate if stuck
        ]

        history_fine_tune = model.fit(
            train_generator,
            epochs=EPOCHS_FINE_TUNE,
            validation_data=val_generator,
            callbacks=callbacks
        )

        history = {}
        for key in history_top.history.keys():
            if key in history_fine_tune.history:
                history[key] = history_top.history[key] + history_fine_tune.history[key]

        with open('training_history.json', 'w') as f:
            json.dump(history, f)
        print("Training complete.")
    
    evaluate(model, val_generator, history)


def evaluate(model, val_generator, history=None):
    if history is None and os.path.exists('training_history.json'):
        with open('training_history.json', 'r') as f:
            history = json.load(f)

    if history is not None:
        acc = history['accuracy']
        val_acc = history['val_accuracy']
        loss = history['loss']
        val_loss = history['val_loss']
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Loss')
        plt.savefig('accuracy_loss.png')

    # Confusion Matrix
    model.load_weights('best_darksouls_model.h5') 
    
    Y_pred = model.predict(val_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = val_generator.classes
    class_labels = list(val_generator.class_indices.keys())

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    print(classification_report(y_true, y_pred, target_names=class_labels))

if __name__ == "__main__":
    parametrs = sys.argv[1:]
    if len(parametrs) == 0 or parametrs[0] != '--evaluate':
        model_exists = os.path.exists('best_darksouls_model.h5')
        if model_exists:
            print("Model already exists. Do you want to overwrite it? (y/n)")
            choice = input().lower()
            if choice == 'y':
                main()
            else:
                sys.exit()
        else:
            main()
    else:
        if not os.path.exists('best_darksouls_model.h5'):
            print("No trained model found.")
            sys.exit()
        main(evaulation_only=True)
