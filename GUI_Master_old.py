import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image , ImageTk 
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import time
import CNNModel 
import sqlite3

import sys
import os


#import tfModel_test as tf_test
global fn
fn=""
##############################################+=============================================================
root = tk.Tk()
root.configure(background="seashell2")
#root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Early Detection of Plant Leaf Disease Using Deep Learning Approach")


#430
#++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 =Image.open('images/master.jpg')
image2 =image2.resize((w,h), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0) #, relwidth=1, relheight=1)

#img=ImageTk.PhotoImage(Image.open("B4.png"))

#img2=ImageTk.PhotoImage(Image.open("B5.jpg"))

#img3=ImageTk.PhotoImage(Image.open("B7.jpg"))


#
#lbl = tk.Label(root, text="Early Detection of Plant Leaf Disease Using Deep Learning Approach", font=('times', 32,' bold '), height=1, width=60,bg="maroon",fg="white")
#lbl.place(x=0, y=3)


frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=220, height=350, bd=5, font=('times', 14, ' bold '),bg="LightPink2")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=10, y=90)

    
    
###########################################################################
def train_model():
 
    update_label("Model Training Start...............")
    
    start = time.time()

    X= CNNModel.main()
    
    end = time.time()
        
    ET="Execution Time: {0:.4} seconds \n".format(end-start)
    
    msg="Model Training Completed.."+'\n'+ X + '\n'+ ET

    print(msg)

import functools
import operator


def convert_str_to_tuple(tup):
    s = functools.reduce(operator.add, (tup))
    return s

def test_model_proc(fn):
    from tensorflow.keras.models import load_model
    #from keras.optimizers import Adam

#    global fn
    
    IMAGE_SIZE = 64
    LEARN_RATE = 1.0e-4
    CH=3
    print(fn)
    if fn!="":
        # Model Architecture and Compilation
       
        model = load_model('plant_model.h5')
            
        img = Image.open(fn)
        img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
        img = np.array(img)
        
        img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)
        
        img = img.astype('float32')
        img = img / 255.0
        print('img shape:',img)
        prediction = model.predict(img)
        print(np.argmax(prediction))
        plant=np.argmax(prediction)
        print(plant)
        
        
        
        if plant == 0:
            Cd="Apple_scab Disease Detected"+'\n'+'Solution : '+'Prune infected leaves, fruit, and branches to prevent further spread'+'\n'+'Apply fungicides, such as copper-based sprays or sulfur, during the growing season'+'\n'+'Improve air circulation around trees by pruning and thinning branches'+'\n'+'Remove fallen leaves and debris from around the base of the tree to reduce overwintering spores'+'\n'+'Practice good sanitation by removing plant debris and fallen leaves'            
        elif plant == 1:
            Cd="Strawberry_Leaf_scorch Disease Detected"+'\n'+'Solution : '+'Remove infected plants and destroy them to prevent the spread of the disease'+'\n'+'Plant disease-resistant strawberry varieties'+'\n'+'Provide proper nutrition and irrigation to maintain plant health'+'\n'+'Avoid overhead watering, as wet leaves can promote the spread of the disease'+'\n'+'Apply copper-based fungicides during the growing season, following label instructions'+'\n'+'Practice good sanitation by removing plant debris and fallen leaves'
        elif plant == 2:
            Cd="Corn_Northern_Blight Disease Detected"+'\n'+'Solution : '+'Rotate crops to prevent the buildup of the disease in the soil'+'\n'+'Plant disease-resistant corn varieties'+'\n'+'Till the soil in the fall to bury crop residue and reduce overwintering spores'+'\n'+'Apply fungicides during the growing season, following label instructions'+'\n'+'Practice good sanitation by removing and destroying infected plants and crop debris'+'\n'+'Avoid planting corn in low-lying areas '
        
        A=Cd
        return A

############################################################
def update_label(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=80, font=("bold", 18), bg='LightPink', fg='black')
    result_label.place(x=300, y=400)

###############################################################################


def test_model():
    global fn
    if fn!="":
        update_label("Model Testing Start...............")
        
        start = time.time()
    
        X=test_model_proc(fn)
        
        #X1="Selected Image is {0}".format(X)
        x2=format(X)#+" Detected"
        
        end = time.time()
            
        ET="Execution Time: {0:.4} seconds \n".format(end-start)
        
        msg="Image Testing Completed.."+'\n'+ x2 + '\n'+ ET
        fn=""
    else:
        msg="Please Select Image For Prediction...."
        
    update_label(msg)
    
#############################################################################
    
def openimage():
   
    global fn
    fileName = askopenfilename(initialdir='test_set', title='Select image for Aanalysis ',
                               filetypes=[("all files", "*.*")])
    IMAGE_SIZE=200
    imgpath = fileName
    fn = fileName


#        img = Image.open(imgpath).convert("L")
    img = Image.open(imgpath)
    
    img = img.resize((IMAGE_SIZE,200))
    img = np.array(img)
#        img = img / 255.0
#        img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)


    x1 = int(img.shape[0])
    y1 = int(img.shape[1])



    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(im)
    img = tk.Label(root, image=imgtk, height=250, width=250)
    img.image = imgtk
    img.place(x=300, y=100)
  
#############################################################################    

def convert_grey():
    global fn    
    IMAGE_SIZE=200
    
    img = Image.open(fn)
    img = img.resize((IMAGE_SIZE,200))
    img = np.array(img)
    
    x1 = int(img.shape[0])
    y1 = int(img.shape[1])

    gs = cv2.cvtColor(cv2.imread(fn, 1), cv2.COLOR_RGB2GRAY)

    gs = cv2.resize(gs, (x1, y1))

    retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(threshold)

    im = Image.fromarray(gs)
    imgtk = ImageTk.PhotoImage(image=im)
    
    #result_label1 = tk.Label(root, image=imgtk, width=250, font=("bold", 25), bg='bisque2', fg='black',height=250)
    #result_label1.place(x=300, y=400)
    img2 = tk.Label(root, image=imgtk, height=250, width=250,bg='white')
    img2.image = imgtk
    img2.place(x=580, y=100)

    im = Image.fromarray(threshold)
    imgtk = ImageTk.PhotoImage(image=im)

    img3 = tk.Label(root, image=imgtk, height=250, width=250)
    img3.image = imgtk
    img3.place(x=880, y=100)
    #result_label1 = tk.Label(root, image=imgtk, width=250,height=250, font=("bold", 25), bg='bisque2', fg='black')
    #result_label1.place(x=300, y=400)


#################################################################################################################
def window():
    root.destroy()




button1 = tk.Button(frame_alpr, text=" Select_Image ", command=openimage,width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
button1.place(x=10, y=40)

button2 = tk.Button(frame_alpr, text="Image_preprocess", command=convert_grey, width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
button2.place(x=10, y=100)

#button3 = tk.Button(frame_alpr, text="Train Model", command=train_model, width=12, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
#button3.place(x=50, y=260)
# #
button4 = tk.Button(frame_alpr, text="CNN_Prediction", command=test_model,width=15, height=1,bg="white",fg="black", font=('times', 15, ' bold '))
button4.place(x=10, y=160)
#
#
#button5 = tk.Button(frame_alpr, text="button5", command=window,width=8, height=1, font=('times', 15, ' bold '),bg="yellow4",fg="white")
#button5.place(x=450, y=20)


exit = tk.Button(frame_alpr, text="Exit", command=window, width=15, height=1, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=10, y=220)



root.mainloop()