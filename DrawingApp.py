import tkinter as tk
from tkinter import font
import numpy as np
from CNN import get_output_labels, train_model;
import torch
import sys
import os
cnn = None

class DrawingApp:
    def __init__(self, master):
        background_color = "black"
        self.master = master
        self.master.title("Drawing")
        self.master.resizable(False,False)
        self.master.configure(background=background_color)

        # Increase canvas size and pixel size
        self.canvas_width = 720
        self.canvas_height = 720
        self.pixel_size = 25  # Adjust this value to set the size of each pixel

        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.grid(row = 0, column=1)

        self.setup_bindings()
        # Add an export button
        #self.export_button = tk.Button(self.master, text="Export to Numpy Bitmap", command=self.export_to_numpy)
        #self.export_button.grid(row = 3, column=0 )
        
        self.bigger_font = font.Font(size=36)
        self.smaller_font = font.Font(size=18)
        
        text_label = tk.Label(root, text="Left Mouse - Draw \t Right Mouse - Erase \t C - Clear",font=self.smaller_font,bg=background_color,fg='white')
        text_label.grid(row=1, column=0,padx=30)
        
        classes_label = tk.Label(root, text="Classes I can recognize: ant, bucket, cow, crab, dragon, fork, lollipop, moon, pizza, zigzag",font=self.smaller_font,bg=background_color,fg='white')
        classes_label.grid(row = 2, column = 0,columnspan=2,pady=30)
        
        self.predicted_label = tk.Label(root, text="",font=self.bigger_font,bg=background_color,fg='white')
        self.predicted_label.grid(row = 1, column=1)
        
        #   Prediction Probabilities
        self.predictions_probabilities = tk.Label(root, text="Draw to predict!",font=self.bigger_font,background=background_color,fg='white')
        self.predictions_probabilities.grid(row = 0, column=0)


    def set_prediction_label(self,label: str):
        self.predicted_label.config(text="My prediction is: " + label)

    def clear_prediction_label(self):
        self.predicted_label.config(text="")
        
    """
    input - (label, probabilies) from the model
    """
    def set_probability_labels(self,labels: list):
        output_text = ""
        for label,probability in labels:
            output_text += str(label) + ": " + '{:.1%}'.format(probability) + "\n" 
        model_prediction = labels[0][0]
        self.set_prediction_label(model_prediction)
        self.predictions_probabilities.config(text=output_text)
        
    def clear_probability_labels(self):
        self.predictions_probabilities.config(text="")
        
    def setup_bindings(self):
        self.canvas.bind("<B1-Motion>", lambda event:self.draw(event=event,color="black"))
        self.canvas.bind("<B2-Motion>", self.erase)
        self.canvas.bind("<B3-Motion>", self.erase)
        self.master.bind("<KeyPress-c>", self.clear_canvas)

    """
    Predict current canvas with our model
    """
    def predict(self):
        #   print(x,y)
        #   converting canvas to tensor to let cnn predict label
        data = self.export_to_numpy()
        data = np.reshape(data, (1, 1, 28, 28))
        data = torch.from_numpy(data)

        labels = get_output_labels(cnn(data))
        self.set_probability_labels(labels)

        
    def draw(self, event,color):
        x = event.x
        y = event.y
        # Map to the nearest pixel coordinates
        x = (x // self.pixel_size) * self.pixel_size
        y = (y // self.pixel_size) * self.pixel_size

        # Draw a pixel on the canvas
        x1, y1 = x, y
        x2, y2 = x + self.pixel_size, y + self.pixel_size
        
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
        self.predict()
        
    def erase(self,event):
        x = (event. x // self.pixel_size) * self.pixel_size
        y = (event. y // self.pixel_size) * self.pixel_size

        # Identify items at the cursor position
        overlapping_items = self.canvas.find_overlapping(x, y, x + self.pixel_size, y + self.pixel_size)

        # Erase by deleting identified items
        for item_id in overlapping_items:
            self.canvas.delete(item_id)
        self.predict()

    # Numpy bitmap files in the actual dataset are flattened numpy array.
    def export_to_numpy(self):
        # Create a numpy array to store pixel values
        pixel_array = np.zeros((28, 28), dtype=np.float32)

        # Iterate through the canvas and update the pixel array
        for y in range(0, 28):
            for x in range(0, 28):
                raw_x = int(x * self.pixel_size + (self.pixel_size/2))
                raw_y = int(y * self.pixel_size + (self.pixel_size/2))
                bounding_box = (raw_x,raw_y,raw_x+1,raw_y+1)
                overlapping_items = self.canvas.find_overlapping(*bounding_box)
                
                pixel_color = self.get_pixel_color(raw_x,raw_y)

                #   Because of the data the model is trained on, it expects the background to be black
                #   and strokes to be white
                #   technically the drawing app we use has a white background and black strokes,
                #   but internally we represent the 'black' pixels as 'white' and vice versa
                #   so that the model can better understand the input
                if(overlapping_items):  #   black turns to white (1)
                    pixel_array[y][x] = 1
                else:                   #   white turns to black (-1)
                    pixel_array[y][x] = -1
                
                #pixel_array[y][x] = 1 if pixel_filled else 0

        # Print or save the numpy array as needed
        #   print("Exported Numpy Array:")
        #   print(pixel_array)
        return pixel_array
        
    def get_pixel_color(self, x, y):
        # Check if the pixel at the specified coordinates is filled
        return self.canvas.itemcget(self.canvas.find_closest(x, y), "fill")
    
    def clear_canvas(self,event):
        self.canvas.delete("all")
        self.clear_prediction_label()
        self.clear_probability_labels()
 
 
if __name__ == "__main__":

    # allow pyinstaller to get the model when building executable
    file_path=""
    if getattr(sys, 'frozen', False):
        file_path=os.path.join(sys._MEIPASS, "./model.pth")
    else:
        file_path = "./model.pth"
        
    #   Only training on 3% of data right now
    cnn = train_model(file_path=file_path, max_iterations=20, batch_size=4, classes=10)
    cnn.eval()
    
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
