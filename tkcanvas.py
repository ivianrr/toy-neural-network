# https://github.com/nj-AllAboutCode/Python-Tkinter-Projects/blob/master/DrawingApp.py
from tkinter import *
from tkinter import ttk, colorchooser, filedialog
import PIL
from PIL import ImageGrab
from PIL import Image, ImageTk
import pickle
import numpy as np
from toynetwork import Network, Layer,Functions
import matplotlib.pyplot as plt
import win32gui


class main:
    def __init__(self,master,model):
        self.master = master
        self.color_fg = 'white'
        self.color_bg = 'black'
        self.old_x = None
        self.old_y = None
        self.penwidth = 25
        self.drawWidgets()
        self.c.bind('<B1-Motion>',self.paint)
        self.c.bind('<ButtonRelease-1>',self.reset)
        self.clear()
        self.model=model
        self.preview_id=-1
        plt.plot(0,0)
        plt.show()
        plt.close()

    def paint(self,e):
        if self.preview_id !=-1:
            self.c.delete(self.preview_id)
            self.preview_id=-1

        if self.old_x and self.old_y:
            self.c.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True)

        self.old_x = e.x
        self.old_y = e.y


    def reset(self,e):
        self.old_x = None
        self.old_y = None   
        
        self.updateimg()

        self.tkimg=ImageTk.PhotoImage(image=Image.fromarray(self.img*255).resize((564,564),Image.BOX))
        self.preview_id=self.c.create_image(0, 0,anchor=NW,image=self.tkimg)
        # self.c.itemconfig(-1, image=self.tkimg)
        # plt.imshow(self.img.reshape(28,28), cmap='gray')
        # plt.show()
        # Image.fromarray(self.img*255).resize((500,500)).show()

    def changeW(self,e):
        self.penwidth = e

    def save(self):
        file = filedialog.asksaveasfilename(filetypes=[('Portable Network Graphics','*.png')])
        if file:
            x = self.master.winfo_rootx() + self.c.winfo_x()
            y = self.master.winfo_rooty() + self.c.winfo_y()
            x1 = x + self.c.winfo_width()
            y1 = y + self.c.winfo_height()

            PIL.ImageGrab.grab().crop((x,y,x1,y1)).save(file + '.png')

    def updateimg(self):
        HWND = self.c.winfo_id()  # get the handle of the canvas

        rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas

        # im = ImageGrab.grab(rect)  
        x = self.c.winfo_rootx() #+ self.c.winfo_x()
        y = self.c.winfo_rooty() #+ self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()
        # X=np.array(PIL.ImageGrab.grab(bbox=(x,y,x1,y1)).resize((28,28),resample=Image.BILINEAR)  ) [:,:,0]#
        # X=np.array(PIL.ImageGrab.grab().crop((x,y,x1,y1))  ) [:,:,0]#.resize((28,28),resample=Image.BILINEAR)
        X=np.array(PIL.ImageGrab.grab(rect).resize((28,28),resample=Image.BILINEAR)) [:,:,0]#
        X=X/X.max()
        # print("___________")
        txt=""
        for line in model.get_prediction_prob(X.reshape((-1,1))):
            s="" if line[1]>10 else "  "
            txt+=f"\n{line[0]} \t{s}{line[1]:.1f} %"
        self.l.config(text=txt)
        self.img=X
        

    def clear(self):
        self.c.delete(ALL)

    def change_fg(self):
        self.color_fg=colorchooser.askcolor(color=self.color_fg)[1]

    def change_bg(self):
        self.color_bg=colorchooser.askcolor(color=self.color_bg)[1]
        self.c['bg'] = self.color_bg

    def drawWidgets(self):
        self.controls = Frame(self.master,padx = 5,pady = 5)
        Label(self.controls, text='Pen Width: ',font=('',15)).grid(row=0,column=0)
        self.slider = ttk.Scale(self.controls,from_= 5, to = 100, command=self.changeW,orient=HORIZONTAL)
        self.slider.set(self.penwidth)
        self.slider.grid(row=0,column=1,ipadx=30)
        
        self.c = Canvas(self.controls,width=500,height=500,bg=self.color_bg)
        self.c.grid(row=1,column=0)
        # self.c.pack(fill=None,expand=False)

        self.l=Label(self.controls, text='',font=('',15))
        self.l.grid(row=1,column=1)

        self.controls.pack()

        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        menu.add_cascade(label='File..',menu=filemenu)
        filemenu.add_command(label='Export..',command=self.save)
        colormenu = Menu(menu)
        menu.add_cascade(label='Colors',menu=colormenu)
        colormenu.add_command(label='Brush Color',command=self.change_fg)
        colormenu.add_command(label='Background Color',command=self.change_bg)
        optionmenu = Menu(menu)
        menu.add_cascade(label='Options',menu=optionmenu)
        optionmenu.add_command(label='Clear Canvas',command=self.clear)
        optionmenu.add_command(label='Exit',command=self.master.destroy) 
        
        

if __name__ == '__main__':
    root = Tk()
    with open('model_deep.pkl', 'rb') as inp:
        model: Network = pickle.load(inp)
    main(root, model)
    root.title('DrawingApp')
    root.mainloop()

    













    
