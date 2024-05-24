import matplotlib.pyplot as plt
import tensorflow
import numpy as np
import cv2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Loads the data required for detecting the license plates from cascade classifier.
plate_cascade = cv2.CascadeClassifier('PLATES.xml')


def detect_plate(img, text=''):
    plate_img = img.copy()
    roi = img.copy()
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=7)
    for (x, y, w, h) in plate_rect:
        roi_ = roi[y:y + h, x:x + w, :]
        plate = roi[y:y + h, x:x + w, :]
        cv2.rectangle(plate_img, (x + 2, y), (x + w - 3, y + h - 5), (51, 181, 155), 3)
    if text != '':
        plate_img = cv2.putText(plate_img, text, (x - w // 2, y - h // 2),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (51, 181, 155), 1, cv2.LINE_AA)

    return plate_img, plate


def display(img_, title=''):
    img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    ax.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()


img = cv2.imread('Car.jpg')
display(img, 'Input image')

output_img, plate = detect_plate(img)
display(output_img, 'Detected license plate in the input image')


# Function to segment characters
def segment_characters(image):
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))
    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]
    img_binary_lp[0:3, :] = 255
    img_binary_lp[:, 0:3] = 255
    img_binary_lp[72:75, :] = 255
    img_binary_lp[:, 330:333] = 255
    dimensions = [LP_WIDTH / 6, LP_WIDTH / 2, LP_HEIGHT / 10, 2 * LP_HEIGHT / 3]
    char_list = find_contours(dimensions, img_binary_lp)
    return char_list


# Generate characters
def find_contours(dimensions, img):
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    ii = cv2.imread('contour.jpg')
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs:
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
            x_cntr_list.append(intX)
            char_copy = np.zeros((44, 24))
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))
            cv2.rectangle(ii, (intX, intY), (intWidth + intX, intY + intHeight), (50, 21, 200), 2)
            plt.imshow(ii, cmap='gray')
            char = cv2.subtract(255, char)
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0
            img_res.append(char_copy)
    plt.show()
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])
    img_res = np.array(img_res_copy)
    return img_res


char = segment_characters(plate)

for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(char[i], cmap='gray')
    plt.axis('off')

# Data generators
train_datagen = ImageDataGenerator(rescale=1. / 255, width_shift_range=0.1, height_shift_range=0.1)
path = r'C:\Users\ASUS\Desktop\indian plates\data\data'
train_generator = train_datagen.flow_from_directory(
    path + '/train',
    target_size=(28, 28),
    batch_size=32,
    class_mode='sparse')

validation_generator = train_datagen.flow_from_directory(
    path + '/val',
    target_size=(28, 28),
    batch_size=32,
    class_mode='sparse')

tensorflow.keras.backend.clear_session()
model = Sequential([
    Conv2D(16, (22, 22), input_shape=(28, 28, 3), activation='relu', padding='same'),
    Conv2D(32, (16, 16), input_shape=(28, 28, 3), activation='relu', padding='same'),
    Conv2D(64, (8, 8), input_shape=(28, 28, 3), activation='relu', padding='same'),
    Conv2D(64, (4, 4), input_shape=(28, 28, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(4, 4)),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(36, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

model.summary()

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=60
)


# Plotting loss and accuracy graphs
def plot_loss_accuracy(history):
    # Plotting loss
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over epochs')
    plt.show()

    # Plotting accuracy
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')
    plt.show()


# Call the function to plot loss and accuracy graphs
plot_loss_accuracy(history)


# Predicting the output
def fix_dimension(img):
    new_img = np.zeros((28, 28, 3))
    for i in range(3):
        new_img[:, :, i] = img
    return new_img


def show_results():
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, c in enumerate(characters):
        dic[i] = c

    output = []
    for i, ch in enumerate(char):  # iterating over the characters
        img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)  # preparing image for the model
        y_ = model.predict(img)[0]  # predicting the class
        y_ = np.argmax(y_)  # getting the index of the maximum value
        character = dic[y_]  #
        output.append(character)  # storing the result in a list

    plate_number = ''.join(output)

    return plate_number

print(f'number plate: {show_results()}')

print(show_results())

plate_number = show_results()
output_img, plate = detect_plate(img, plate_number)
display(output_img, 'detected license plate number in the input image')

# Save the trained model
#model.save('license_plate_model.h5')
