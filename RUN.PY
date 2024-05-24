
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import tensorflow
import numpy as np
import cv2
from PIL import Image, ImageTk

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from plate import show_results
bush = show_results()
class LicensePlateDetector:
    def __init__(self):
        self.plate_cascade = cv2.CascadeClassifier('PLATES.xml')
        self.model = self.create_model()

    def create_model(self):
        # Create the model architecture
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
        return model

    def detect_plate(self, img, text=''):
        plate_img = img.copy()
        roi = img.copy()
        plate_rect = self.plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=7)
        for (x, y, w, h) in plate_rect:
            roi_ = roi[y:y + h, x:x + w, :]
            plate = roi[y:y + h, x:x + w, :]
            cv2.rectangle(plate_img, (x + 2, y), (x + w - 3, y + h - 5), (51, 181, 155), 3)
        if text != '':
            plate_img = cv2.putText(plate_img, text, (x - w // 2, y - h // 2),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (51, 181, 155), 1, cv2.LINE_AA)
        return plate_img, plate

    def segment_characters(self, image):
        # Segment characters from the license plate image
        img_lp = cv2.resize(image, (333, 75))
        img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
        _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
        img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))
        LP_WIDTH = img_binary_lp.shape[0]
        LP_HEIGHT = img_binary_lp.shape[1]
        img_binary_lp[0:3, :] = 255
        img_binary_lp[:, 0:3] = 255
        img_binary_lp[72:75, :] =255
        img_binary_lp[:, 330:333] = 255
        dimensions = [LP_WIDTH / 6, LP_WIDTH / 2, LP_HEIGHT / 10, 2 * LP_HEIGHT / 3]
        char_list = self.find_contours(dimensions, img_binary_lp)
        return char_list

    def find_contours(self, dimensions, img):
        # Find contours in the binary image
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
                char = img[intY:intY + intHeight, intX:intX +intWidth]
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

    def fix_dimension(self, img):
        new_img = np.zeros((28, 28, 3))
        for i in range(3):
            new_img[:, :, i] = img
        return new_img

    def show_results(self, char):
        dic = {}
        characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i, c in enumerate(characters):
            dic[i] = c

        output = []
        for i, ch in enumerate(char):  # iterating over the characters
            img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
            img = self.fix_dimension(img_)
            img = img.reshape(1, 28, 28, 3)  # preparing image for the model
            y_ = self.model.predict(img)[0]  # predicting the class
            y_ = np.argmax(y_)  # getting the index of the maximum value
            character = dic[y_]  #
            output.append(character)  # storing the result in a list

        plate_number = ''.join(output)

        return plate_number

class GUI:
    def __init__(self, master):
        self.master= master
        self.master.geometry("500x500")
        self.master.title("License Plate Detector")
        self.master.configure(bg='white')
        self.detector = LicensePlateDetector()
        self.create_widgets()

    def create_widgets(self):
        self.load_button = tk.Button(self.master, text="Load Image", command=self.load_image, bg='#4CAF50', fg='white', font=('Arial', 12, 'bold'))
        self.load_button.pack(pady=20)

        self.canvas = tk.Canvas(self.master, width=400, height=400, bg='white')
        self.canvas.pack(pady=20)

        self.image_label = tk.Label(self.master, text="No image loaded", bg='white', font=('Arial', 12))
        self.image_label.pack()

        self.result_label = tk.Label(self.master, text="No result", bg='white', font=('Arial', 12))
        self.result_label.pack()

    def load_image(self):
        filepath = filedialog.askopenfilename()
        img = cv2.imread(filepath)
        self.image_label.config(text="Image loaded")
        self.result_label.config(text="")
        self.canvas.delete("all")
        output_img, plate = self.detector.detect_plate(img)
        char = self.detector.segment_characters(plate)
        plate_number = self.detector.show_results(char)
        self.result_label.config(text=f"License plate number: {bush}")

        # Convert the OpenCV image to PIL format
        img_pil = Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        img_pil = img_pil.resize((400, 400))

        # Display the image on the canvas
        self.photo = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Add the 'sapele' text to the canvas
        self.canvas.create_text(10, 10, anchor=tk.NW, fill='red')


root = tk.Tk()
gui = GUI(root)
root.mainloop()