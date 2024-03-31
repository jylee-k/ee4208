import cv2
from FaceIDLight.tools import FaceID
import threading, time
import sys
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np



class Demonstrator:
    def __init__(self, gal_dir: str = None, stream_id: int = 0, model_type: str = "mobileNet"):
        #Model for face recognition 
        self.FaceID = FaceID(gal_dir=gal_dir, model_type=model_type)

        # Set OpenCV defaults
        self.color = (0, 0, 255)
        self.font = cv2.QT_FONT_NORMAL

        #camera_init
                # camera 
        self.stream_id = stream_id
        self.currentFrame = None
        self.ret = False
        self.stop = False
        self.capture = cv2.VideoCapture(stream_id)
        self.thread = threading.Thread(target=self.update_frame)

        self.flag = False
        self.stop = False

        self.new_name = 'cat'
    
    def update_frame(self):
        while True:
            self.ret, self.currentFrame = self.capture.read()
            while self.currentFrame is None:  # Continually grab frames until we get a good one
                self.capture.read()
            if self.stop:
                break

    # Get current frame
    def get_frame(self):
        return self.ret, self.currentFrame

    def screen(self, function):
        self.thread.start()
        window_name = "Streaming from {}".format(self.stream_id)
        cv2.namedWindow(window_name)
        last = 0
        while True:
            ret, frame = self.get_frame()
            if ret:
                frame = function(frame)
                frame = cv2.putText(
                    frame,
                    "FPS{:5.1f}".format(1 / (time.time() - last)),
                    (frame.shape[1] - 80, 30),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 255, 0),
                )
                last = time.time()
                cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.stop = True
        self.thread.join()
        cv2.destroyWindow(window_name)
        self.capture.release()

    def annotate(self, img, results):
        if not results:
            return img
        num = 0
        for result in results:
            face, detections, ids = result
            (
                bbox,
                points,
                conf,
            ) = detections
            name, gal_face, dist, id_conf = ids
            
            # Bbox as int
            bbox = bbox.astype(int)

            # Point as int
            points = points.astype(int)

            # Add BoundingBox
            img = cv2.rectangle(img, tuple(bbox[0]), tuple(bbox[1]), self.color)

            # Add LandmarkPoints
            for point in points:
                img = cv2.circle(img, tuple(point), 5, self.color)

            # Add Confidence Value
            img = cv2.putText(
                img,
                "Detect-Conf: {:0.2f}%".format(conf * 100),
                (int(bbox[0, 0]), int(bbox[0, 1]) - 20),
                self.font,
                0.7,
                (0, 0, 255),
            )

            img = cv2.putText(
                img, "{}".format(name), (int(bbox[0, 0]), int(bbox[0, 1]) - 80), self.font, 0.7, (255, 255, 0)
            )
            img = cv2.putText(
                img,
                "Emb-Dist: {:0.2f}".format(dist),
                (int(bbox[0, 0]), int(bbox[0, 1] - 60)),
                self.font,
                0.7,
                self.color,
            )
            img = cv2.putText(
                img,
                "ID-Conf: {:0.2f} %".format(id_conf * 100),
                (int(bbox[0, 0]), int(bbox[0, 1] - 40)),
                self.font,
                0.7,
                self.color,
            )

            # Add gal face onto img
            if name != "Other":
                img[0 : 0 + gal_face.shape[1], num : num + gal_face.shape[0]] = gal_face
                # Match
                img = cv2.putText(img, "GalleryMatch", (num, 112 + 10), self.font, 0.4, (255, 255, 255))

            num += 112
        return img

    def identification(self, frame):
        if self.FaceID.recognize_faces(frame):
            results,  other_ids = self.FaceID.recognize_faces(frame)
        else:
            results, other_ids = None, []
        frame = self.annotate(frame, results)

        return frame, other_ids
        #return frame

    def run(self):
        self.screen(self.identification)

    def show(self):

        camIndex = 0
        self.cap = cv2.VideoCapture(camIndex)
        success, frame = self.cap.read()
        if not success:
            # if camIndex == 0:
            print("Error, No webcam found!")
            sys.exit(1)

        # mainWindow = tk.Tk(screenName="Online Face Detection and Recognition System")
        mainWindow = tk.Tk()
        mainWindow.resizable(width=False, height=False)
        mainWindow.title("Online Face Detection and Recognition System")
        
        mainWindow.bind('<Escape>', lambda e: mainWindow.quit())
        lmain = tk.Label(mainWindow, compound=tk.CENTER, anchor=tk.CENTER, relief=tk.RAISED)

        self.button = tk.Button(mainWindow, text="Add person", command=self.show_entry, state=tk.DISABLED)
        self.button_quit = tk.Button(mainWindow, text="Quit", command=mainWindow.quit)
        # ipdb.set_trace()
        lmain.pack()
        self.button.place(bordermode=tk.INSIDE, relx=0.8, rely=0.1, anchor=tk.CENTER, width=300, height=50)
        self.button_quit.place(bordermode=tk.INSIDE, relx=0.8, rely=0.8, anchor=tk.CENTER, width=300, height=50)
        self.button.focus()
        # button.place(bordermode=tk.INSIDE, relx=0.5, rely=0.9, anchor=tk.CENTER, width=300, height=50)
        # button.focus()
        # button_changeCam.place(bordermode=tk.INSIDE, relx=0.85, rely=0.1, anchor=tk.CENTER, width=150, height=50)

        self.mainWindow = mainWindow
        self.lmain = lmain

        self.name_entry = tk.Entry(self.mainWindow)
        self.confirm_button = tk.Button(self.mainWindow, text="Confirm", command=self.add_to_gallery)
        
        self.show_frame()
        mainWindow.mainloop()
    
    def show_frame(self):
    
        _, frame = self.cap.read()

        # frame = func(frame)
        '''
        frame = self.identification(frame)
        self.flag = True
        self.other_ids = True
        '''
        
        frame, other_ids = self.identification(frame)
        
        self.flag = True
        self.other_ids = other_ids

        
        

        if self.flag:
            self.button.configure(state=tk.NORMAL)

        else:
            self.button.configure(state=tk.DISABLED)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    
        self.prevImg = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=self.prevImg)
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)
        
        if not self.stop:
            self.lmain.after(10, self.show_frame)
    
    '''
    def add_person(self):
        # global cancel, button, button1, button2
        self.stop = True
        
        self.button.place_forget()

        self.button1 = tk.Button(self.mainWindow, text="Add to Gallery", command=self.show_entry)
        self.button2 = tk.Button(self.mainWindow, text="Try Again", command=self.resume)
        self.button1.place(anchor=tk.CENTER, relx=0.2, rely=0.9, width=150, height=50)
        self.button2.place(anchor=tk.CENTER, relx=0.8, rely=0.9, width=150, height=50)
        self.button1.focus()
    '''
    
    # def show_entry_fields(self):
    #     print("Name: %s" % (self.e1.get()))
    #     # print(len(e1.get()))
    #     self.new_name = self.e1.get()

    def show_entry(self):
        #self.button1.place_forget()
        self.button.place_forget()

        self.button2 = tk.Button(self.mainWindow, text="Add another person", command=self.show_entry)
        self.button2.place(anchor=tk.CENTER, relx=0.8, rely=0.3, width=150, height=50)

        self.name_entry.place(anchor=tk.CENTER, relx=0.8, rely=0.1, width=150, height=50)
        self.name_entry.focus()

        self.confirm_button.place(anchor=tk.CENTER, relx=0.8, rely=0.2, width=150, height=50)
            
        # self.mainWindow.bind('<Return>', self.add_to_gallery)  # Bind the Enter key


        # self.inputtest = tk.Tk()
        # tk.Label(self.inputtest, text="Name").grid(row=0)
        # self.e1 = tk.Entry(self.inputtest)
        # self.e1.grid(row=0, column=1)
        # # print("\ne1.get():", e1.get())
        # tk.Button(self.inputtest, text='comfirm name', command=self.show_entry_fields).grid(row=3, column=1, sticky=tk.W, pady=4)
        # tk.Button(self.inputtest, text='add_to_gallery', command=self.add_to_gallery).grid(row=3, column=0, sticky=tk.W, pady=4)
    '''
    def add_name(self):
        self.new_name = self.name_entry.get()
        print("Name entered:", self.new_name)
        
        # clear the text entry widget and place the button back in its position
        self.name_entry.delete(0, tk.END)
        self.name_entry.place_forget()
        self.button1.place(anchor=tk.CENTER, relx=0.2, rely=0.9, width=150, height=50)
        '''
        

    def add_to_gallery(self):
        # new_name = 'cat'
        name = self.name_entry.get()
        print(name)
        self.new_name = name
            
        self.name_entry.place_forget()
        self.confirm_button.place_forget()

        if self.other_ids:
            new_emb, new_face = self.other_ids[0]

        #print(len(self.FaceID.gal_embs[0]))
        #print(len(new_emb))
        #print(self.FaceID.gal_embs)
        #self.FaceID.gal_embs.append(new_emb)
        #new_emb = new_emb.reshape(-1, 1)
        #self.FaceID.gal_embs = np.append(self.FaceID.gal_embs,list(new_emb))
        self.FaceID.gal_embs = np.vstack([self.FaceID.gal_embs,new_emb])
        self.FaceID.gal_names.append(self.new_name)
        self.FaceID.gal_faces.append(new_face)

        """ 
        self.FaceID.gal_embs.append(new_emb)
        self.FaceID.gal_names.append(self.new_name)
        self.FaceID.gal_faces.append(new_face)
        """
        print('Add successful!!!')

        print(len(self.FaceID.gal_embs))

        """        
        self.stop = False
        self.button1.place_forget()
        self.button2.place_forget()

        #self.mainWindow.bind('<Return>', self.add_person)
        self.mainWindow.bind('<Return>', self.show_entry)
        self.button.place(bordermode=tk.INSIDE, relx=0.5, rely=0.9, anchor=tk.CENTER, width=300, height=50)
        self.lmain.after(10, self.show_frame)
        """
    '''
    def resume(self):
        self.stop = False

        self.button1.place_forget()
        self.button2.place_forget()
        self.name_entry.place_forget()
        self.confirm_button.place_forget()

        self.mainWindow.bind('<Return>', self.add_person)
        self.button.place(bordermode=tk.INSIDE, relx=0.5, rely=0.9, anchor=tk.CENTER, width=300, height=50)
        self.lmain.after(10, self.show_frame)
        '''
