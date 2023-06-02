import sys
# For application
if getattr(sys, 'frozen', False):
    sys.stdout = open(os.path.join(os.path.dirname(sys.executable), 'stdout.log'), 'w')
from tkinter import * 
from tkinter import ttk
import cv2 
import time
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import os
from sklearn.svm import SVC
from tkinter import messagebox
from PIL import Image, ImageTk

root = None

haar_cascade = cv2.CascadeClassifier('C:/Users/nikhi/OneDrive/1.UNI Work/VScode/FaceRecognitionProject/haarcascade_frontalface_default.xml')
model = VGGFace(model="resnet50", include_top=False, input_shape=(224,224,3), pooling="avg", weights="vggface")
dataset_path = "C:/Users/nikhi/OneDrive/1.UNI Work/VScode/FaceRecognitionProject/Dataset"
dataset = []

# Loading dataset using progress bar
def load_dataset(progress):
    progress["value"] = 0
    progress["maximum"] = len(os.listdir(dataset_path))
    for filename in os.listdir(dataset_path):
        filepath = os.path.join(dataset_path, filename)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        label = filename.split("_")[0]  
        features = feature_extraction(image)
        dataset.append((features, label))
        progress["value"] =  1 + progress["value"]
        progress.update()

def update_dataset(progress):
    dataset.clear()
    progress["value"] = 0
    progress["maximum"] = len(os.listdir(dataset_path))
    for filename in os.listdir(dataset_path):
        filepath = os.path.join(dataset_path, filename)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        label = filename.split("_")[0]  
        features = feature_extraction(image)
        dataset.append((features, label))
        progress["value"] =  1 + progress["value"]
        progress.update()

# Extracting features from face
def feature_extraction(face):
    face = cv2.resize(face, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    face = np.expand_dims(face, axis=0)
    face = face.astype("float32")
    face = preprocess_input(face)
    features = model.predict(face)
    return features.flatten()

# Face recognition process using SVM
def face_recognition(face):
    input_features = feature_extraction(face)
    
    distances = []
    for features, _ in dataset:
        distance = np.linalg.norm(input_features - features)
        distances.append(distance)

    threshold = 100
    valid_distances = []
    for valid_distance in distances:
        if valid_distance < threshold:
            valid_distances.append(valid_distance)
    if not valid_distances:
        return "Unknown"

    features = []
    labels = []
    for feature, label in dataset:
        features.append(feature)
        labels.append(label)

    if len(set(labels)) == 1:
        return labels[0]

    svm = SVC(kernel="linear", probability=True)
    svm.fit(features, labels)
    prediction = svm.predict_proba([input_features])[0]
    label = svm.predict([input_features])[0]
    confidence = prediction.max()
    
    if confidence < 0.6:
        return "Unknown" + "  " + str(round(confidence * 100)) + "%"
    return label + "  " + str(round(confidence * 100)) + "%"

# Face detection process incoporating face recognition
def faceDetection():
    cam = cv2.VideoCapture(0) 
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.putText(img,"Press 'esc' to quit",(10,30),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,0),thickness=1)
            face = gray[y:y+h, x:x+w]
            label = face_recognition(face)
            if (label.startswith("Unknown")):
                cv2.putText(img,label,(x,y-5),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255))
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
            elif (label.lower().startswith("nikhil")):
                cv2.putText(img,label,(x,y-5),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,255))
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 3, cv2.LINE_4)
            else:
                cv2.putText(img,label,(x,y-5),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0))
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.imshow("camera", img)
        if cv2.waitKey(1) & 0xff == 27: 
            break
    cam.release()
    cv2.destroyAllWindows()

# Capturing face for dataset
def capture_face(name,top): 
    if name == "":
        messagebox.showerror("ERROR", "Please enter your name")
    elif name.lower() == "unknown":
        messagebox.showerror("ERROR", "Please enter a valid name")
    else:
        sample_size = 30
        cam = cv2.VideoCapture(0) 
        count = 0
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = haar_cascade.detectMultiScale(gray, 1.3, 5)
            if count < sample_size:
                for (x, y, w, h) in faces:
                    cropped_face = gray[y:y+h,x:x+w :]
                    cv2.imwrite("C:/Users/nikhi/OneDrive/1.UNI Work/VScode/FaceRecognitionProject/Dataset/{}_{}.jpg".format(name,count), cropped_face)
                    count += 1  
                    cv2.putText(img,str(count) + "/" + str(sample_size) ,(x,y-5),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0))
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
                    time.sleep(0.1)
                cv2.imshow("camera", img) 
                if cv2.waitKey(1) & 0xff == 27:
                    break   
            else:
                break
        cam.release()
        cv2.destroyAllWindows()
    top.destroy()     

# Deleting face from dataset
def delete_face(name,top):
    found = False
    if name == "":
        messagebox.showerror("ERROR", "Please enter a name")
    else:
        response = messagebox.askyesno("Delete", "Are you sure you want to delete " + name + "?")
        for filename in os.listdir(dataset_path):
            label = filename.split("_")[0]  
            if label == name:
                filepath = os.path.join(dataset_path, filename)
                if response == 1:
                    found = True
                    os.remove(filepath)
                else:
                    break 
        if found == True:
            progress_bar2()
            messagebox.showinfo("Success", "Face deleted successfully")
        else:
            messagebox.showerror("ERROR", "No face found for " + name)
    top.destroy()

#GUI
def home_screen():
    global root
    root = Tk()
    root.title("Face Recognition")
    root.geometry("600x400")
    root.iconbitmap("C:/Users/nikhi/OneDrive/1.UNI Work/VScode/FaceRecognitionProject/FRicon.ico")


    mylabel = Label(root, text="Please wait while the dataset is being loaded and trained...", font=("Times", 10))
    mylabel.grid(row=0, column=4, columnspan=2, padx=10, pady=10)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("purple.Horizontal.TProgressbar", foreground="purple", background="purple", troughcolor="silver")

    progress = ttk.Progressbar(root,style="purple.Horizontal.TProgressbar", orient=HORIZONTAL, length=300, mode="determinate")
    progress.grid(row=1, column=4, columnspan=2, padx=10, pady=10)     

    mylabel.destroy()
    progress.destroy()

    Label1 = Label(root, text="Face Recognition System", font=("Times", 20, "underline"))
    Label1.grid(row=0, column=4, columnspan=2, padx=150, pady=10)

    button1 = Button(root, text = "Live stream", command = faceDetection, width = 20, background="light gray",activebackground="light green")
    button2 = Button(root, text = "Add New Face", command = add_face_window, width = 20, background="light gray",activebackground="light green")
    button3 = Button(root, text = "Delete face", command = remove_face_window, width = 20, background="light gray", activebackground="light green")
    button4 = Button(root, text = "Train", command = progress_bar, width = 20,background="light gray",activebackground="purple",activeforeground="white")
    button5 = Button(root, text = "View Faces", command = authentication, width = 20, background="light gray",activebackground="light green", foreground="red")


    button1.grid(row = 1, column = 4, columnspan = 3, padx = 10, pady = 10)
    button2.grid(row = 2, column = 4, columnspan = 3, padx = 10, pady = 10)
    button3.grid(row = 3, column = 4, columnspan = 3, padx = 10, pady = 10)
    button4.grid(row = 4, column = 4, columnspan = 3, padx = 10, pady = 10)
    button5.grid(row = 5, column = 4, columnspan = 3, padx = 10, pady = 10)

    root.mainloop()

def add_face_window():
    top = Toplevel()
    top.title("Add Face")
    top.geometry("220x150")
    top.iconbitmap("C:/Users/nikhi/OneDrive/1.UNI Work/VScode/FaceRecognitionProject/FRicon.ico")
    
    mylabel = Label(top, text = "Enter your name",font = ("Times", 10))
    mylabel.grid(row = 0, column = 1,columnspan = 2,padx = 10, pady = 10)

    name = Entry(top, width = 30)
    name.grid(row = 1, column = 1, padx = 10, pady = 10)

    button1 = Button(top, text = "Capture", command = lambda: capture_face(name.get(),top), width = 20, activebackground="blue",activeforeground="white")
    button1.grid(row = 2, column = 1, columnspan = 3, padx = 10, pady = 10)

def remove_face_window():
    top = Toplevel()
    top.title("Delete Face")
    top.geometry("220x150")
    top.iconbitmap("C:/Users/nikhi/OneDrive/1.UNI Work/VScode/FaceRecognitionProject/FRicon.ico")

    mylabel = Label(top, text = "Enter the exact name",font = ("Times", 10))
    mylabel.grid(row = 0, column = 1,columnspan = 3,padx = 10, pady = 10)

    name = Entry(top, width = 30)
    name.grid(row = 1, column = 1, padx = 10, pady = 10)

    button1 = Button(top, text = "Delete",command = lambda: delete_face(name.get(),top), width = 20, activebackground="red")
    button1.grid(row = 2, column = 1, columnspan = 3, padx = 10, pady = 10)

def progress_bar():
    global root
    top = Toplevel(root)
    top.geometry("400x100")
    top.title("Loading Dataset...")
    top.iconbitmap("C:/Users/nikhi/OneDrive/1.UNI Work/VScode/FaceRecognitionProject/FRicon.ico")

    mylabel = Label(top, text = "Please wait while the dataset is being loaded and trained...",font = ("Times", 10))
    mylabel.grid(row = 0, column = 0,columnspan = 2,padx = 10, pady = 10)

    progress = ttk.Progressbar(top,style="purple.Horizontal.TProgressbar",orient = HORIZONTAL, length = 300, mode = "determinate")
    progress.grid(row = 1, column = 0,columnspan = 3,padx = 10, pady = 10)

    load_dataset(progress)
    messagebox.showinfo("Success", "Dataset loaded and trained successfully")
    top.destroy()

def progress_bar2():
    global root
    top = Toplevel(root)
    top.geometry("400x100")
    top.title("Deleting Face...")
    top.iconbitmap("C:/Users/nikhi/OneDrive/1.UNI Work/VScode/FaceRecognitionProject/FRicon.ico")

    mylabel = Label(top, text = "Please wait while the face is being deleted",font = ("Times", 10))
    mylabel.grid(row = 0, column = 0,columnspan = 2,padx = 10, pady = 10)

    progress = ttk.Progressbar(top,style="purple.Horizontal.TProgressbar",orient = HORIZONTAL, length = 300, mode = "determinate")
    progress.grid(row = 1, column = 0,columnspan = 3,padx = 10, pady = 10)

    update_dataset(progress)
    top.destroy()

def view_faces(input_username,input_password,top):
    top.destroy()
    username = "admin"
    password = "password"
    labels_displayed = []
    index = 0
    if input_username == username and input_password == password:
        top = Toplevel()
        top.title("People in Dataset")
        top.geometry("620x500")
        top.iconbitmap("C:/Users/nikhi/OneDrive/1.UNI Work/VScode/FaceRecognitionProject/FRicon.ico")
        for filename in os.listdir(dataset_path):
            label = filename.split("_")[0]
            if label not in labels_displayed:
                labels_displayed.append(label)
                filepath = os.path.join(dataset_path, filename)
                img = Image.open(filepath)
                img = img.resize((100, 100))
                img = ImageTk.PhotoImage(img)

                faces = Label(top, image=img)
                faces.image = img
                faces.grid(row=index // 5 * 2, column=index % 5, padx=10, pady=10)

                name = Label(top, text=label, font=("Arial", 10, "bold"), fg="red")
                name.grid(row=(index // 5 * 2) + 1, column=index % 5, padx=10, pady=10)
                index += 1 
    else:
        messagebox.showerror("Error", "Incorrect username or password")
        authentication()

def authentication():
    top = Toplevel()
    top.title("Login")
    top.geometry("400x150")
    top.iconbitmap("C:/Users/nikhi/uni/FRicon.ico")



    mylabel = Label(top, text = "Username:",font = ("Times", 10))
    mylabel.grid(row = 0, column = 1,padx = 10, pady = 10)

    input_username = Entry(top, width = 30)
    input_username.grid(row = 0, column = 2, padx = 10, pady = 10)
    input_username.insert(0, "admin")

    mylabel2 = Label(top, text = "Password:",font = ("Times", 10))
    mylabel2.grid(row = 1, column = 1,padx = 10, pady = 10)

    input_password = Entry(top, width = 30, show = "*",)
    input_password.grid(row = 1, column = 2, padx = 10, pady = 10)
    input_password.insert(0, "password")

    button1 = Button(top, text = "Authenticate", command = lambda: view_faces(input_username.get(),input_password.get(),top), width = 20, activebackground=("turquoise"))
    button1.grid(row = 3, column = 2, padx = 10, pady = 10)


home_screen()