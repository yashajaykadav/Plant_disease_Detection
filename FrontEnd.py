from tkinter import ttk, Tk, filedialog, messagebox ,Text
from PIL import ImageTk, Image
import EX1
import Diesease_info

# Initialize the Tkinter window
root = Tk()
root.geometry("1536x864")
root.minsize(1024, 600)
root.title("Crop Disease Detection System and Solutions")
root.config(bg="lightgreen")

# Global variable to hold the image path
ImgLocation =r"default.jpeg"

# Function to get the image path and update the displayed image
def GetImage():
    global ImgLocation
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if file_path:
        ImgLocation = file_path
        img = Image.open(file_path)
        img = img.resize((500, 500))  # Resize the image
        img_tk = ImageTk.PhotoImage(img)
        ImgButton.config(image=img_tk)
        ImgButton.image = img_tk  # Keep a reference to the image

# Function to check the crop and display results
def CheckCrop():
    disease_name = EX1.predict_image(ImgLocation)
    disease_info = Diesease_info.getInfo(disease_name)

    # Update Disease Description TextBox
    description_text.config(state="normal")
    description_text.delete("1.0", "end")
    description_text.insert("end","Disease Name: " + disease_name + "\n\n"+disease_info["Description"])
    description_text.config(state="disabled")

    # Update Solution TextBox
    solution_text.config(state="normal")
    solution_text.delete("1.0", "end")
    solution_text.insert("end", disease_info["Solution"] )
    solution_text.config(state="disabled")

style = ttk.Style()
style.configure("Custom.TFrame", background="lightgreen")
# Heading Frame with logo and title
Heading_frame = ttk.Frame(root, padding=10, relief="flat")
Heading_frame.grid(row=0, column=0, columnspan=3)
Heading_frame.config(style='Custom.TFrame')

logo_image = Image.open('cropLogo.jpeg')
logo_image = logo_image.resize((50, 50))
logo = ImageTk.PhotoImage(logo_image)
Heading = ttk.Label(Heading_frame, image=logo)
Heading.grid(row=0, column=0)
Headingtext = ttk.Label(Heading_frame, text="Crop Disease Detection System", font=("times new roman", 40, "bold"),background='lightgreen')
Headingtext.grid(row=0, column=1, padx=20)

# Image Selection Frame
img_frame = ttk.Frame(root, padding=10)
img_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
img_frame.columnconfigure(0, weight=1)
img_frame.config(style='Custom.TFrame')

Imgtext = ttk.Label(img_frame, text="Select Image", font=("times new roman", 25, "bold"),background='lightgreen')
Imgtext.grid(row=0, column=0, pady=10)

defaultImg = Image.open("default.jpeg")
defaultImg = defaultImg.resize((500, 500))
defaultImg_tk = ImageTk.PhotoImage(defaultImg)

ImgButton = ttk.Button(img_frame, image=defaultImg_tk, command=GetImage)
ImgButton.grid(row=1, column=0, sticky="nsew", pady=10)

Scan = ttk.Button(img_frame, text="Check Crop", command=CheckCrop)
Scan.grid(row=2, column=0, pady=10)

# Vertical Separator
separator = ttk.Separator(root, orient='vertical')
separator.grid(row=1, column=1, sticky="ns", padx=0, pady=0)

# Disease Description and Solution Frame
info_frame = ttk.Frame(root, padding=10)
info_frame.grid(row=1, column=2, sticky="nsew", padx=10, pady=10)
info_frame.columnconfigure(0, weight=1)

description_label = ttk.Label(info_frame, text="Disease Description", font=("times new roman", 25, "bold"))
description_label.grid(row=0, column=0, sticky="w")
description_text = Text(info_frame, font=("times new roman", 15), wrap="word", height=10)
description_text.grid(row=1, column=0, sticky="nsew")
description_text.insert("end", "Description will appear here...")
description_text.config(state="disabled")

solution_label = ttk.Label(info_frame, text="Solution", font=("times new roman", 25, "bold"))
solution_label.grid(row=2, column=0, sticky="w", pady=10)
solution_text = Text(info_frame, font=("times new roman", 15), wrap="word", height=10)
solution_text.grid(row=3, column=0, sticky="nsew")
solution_text.insert("end", "Solutions will appear here...")
solution_text.config(state="disabled")

root.mainloop()