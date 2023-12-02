import tkinter as tk
import numpy as np

from CNN import train_model;

class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Drawing")
        self.master.resizable(False,False)
        # Increase canvas size and pixel size
        self.canvas_width = 720
        self.canvas_height = 720
        self.pixel_size = 25  # Adjust this value to set the size of each pixel

        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

        self.setup_bindings()

        # Add an export button
        self.export_button = tk.Button(self.master, text="Export to Numpy Bitmap", command=self.export_to_numpy)
        self.export_button.pack()
        
        text_label = tk.Label(root, text="Left Mouse - Draw \t Right Mouse - Erase \t C - Clear")
        text_label.pack()

    def setup_bindings(self):
        self.canvas.bind("<B1-Motion>", lambda event:self.draw(event=event,color="black"))
        self.canvas.bind("<B2-Motion>", self.erase)
        self.master.bind("<KeyPress-c>", self.clear_canvas)

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
        
        print(x,y)
    def erase(self,event):
        
        x = (event. x // self.pixel_size) * self.pixel_size
        y = (event. y // self.pixel_size) * self.pixel_size

        # Identify items at the cursor position
        overlapping_items = self.canvas.find_overlapping(x, y, x + self.pixel_size, y + self.pixel_size)

        # Erase by deleting identified items
        for item_id in overlapping_items:
            self.canvas.delete(item_id)

    def export_to_numpy(self):
        # Create a numpy array to store pixel values
        pixel_array = np.zeros((28, 28), dtype=np.uint8)

        # Iterate through the canvas and update the pixel array
        for y in range(0, 28):
            for x in range(0, 28):
                raw_x = int(x * self.pixel_size + (self.pixel_size/2))
                raw_y = int(y * self.pixel_size + (self.pixel_size/2))
                bounding_box = (raw_x,raw_y,raw_x+1,raw_y+1)
                overlapping_items = self.canvas.find_overlapping(*bounding_box)
                
                pixel_color = self.get_pixel_color(raw_x,raw_y)
                if(overlapping_items):
                    pixel_array[y][x] = 1
                else:
                    pixel_array[y][x] = 0
                
                #pixel_array[y][x] = 1 if pixel_filled else 0

        # Print or save the numpy array as needed
        print("Exported Numpy Array:")
        print(pixel_array)
        
    def get_pixel_color(self, x, y):
        # Check if the pixel at the specified coordinates is filled
        return self.canvas.itemcget(self.canvas.find_closest(x, y), "fill")
    
    def clear_canvas(self,event):
        self.canvas.delete("all")
 
if __name__ == "__main__":
    #   Only training of 3% of data right now
    train_model(max_iterations=10, batch_size=4, classes=10)
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
