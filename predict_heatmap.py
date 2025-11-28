import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

MODEL_PATH = 'cataract_classifier.h5' 
IMG_SIZE = (150, 150) 
CLASS_NAMES = ["Cataract", "Normal Eye"]
POSITIVE_CLASS_INDEX = 1 
GRAD_CAM_TARGET_LAYER = 'conv2d_2'

DISPLAY_BOX_SIZE = (350, 350) 

# --- Global Variables for Tkinter (will be initialized later) ---
model = None 
image_label = None
heatmap_ax = None
heatmap_canvas = None
result_label = None
heatmap_frame = None # New global variable for the right side frame

# --- Grad-CAM Functions ---

def make_gradcam_heatmap(img_array, model, target_layer_name, pred_index):
    """
    Computes the Grad-CAM heatmap using a single, robust GradientTape block 
    by separating the model into a feature extractor and a classifier.
    """
    try:
        target_layer = model.get_layer(target_layer_name)
    except ValueError:
        print(f"Error: Target layer '{target_layer_name}' not found.")
        return np.zeros(IMG_SIZE)
    
    # separate model into feature extractor and classifier
    feature_extractor = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=target_layer.output
    )

    temp_conv_output = feature_extractor(img_array)
    target_layer_shape = temp_conv_output.shape[1:]
    
    layer_index = -1
    for i, layer in enumerate(model.layers):
        if layer.name == target_layer_name:
            layer_index = i
            break
            
    classifier_input = tf.keras.Input(shape=target_layer_shape)
    x = classifier_input
    for layer in model.layers[layer_index + 1:]:
        x = layer(x)
        
    classifier_model = tf.keras.models.Model(classifier_input, x)

    # compute gradients
    with tf.GradientTape() as tape:
        conv_output = feature_extractor(img_array)
        tape.watch(conv_output) 
        predictions = classifier_model(conv_output)
       
        if pred_index == POSITIVE_CLASS_INDEX:
            loss_target = predictions[0][0]
        else:
            loss_target = 1.0 - predictions[0][0] 
        
    # get gradients of the target prediction score
    grads = tape.gradient(loss_target, conv_output)[0]

    # check for dead gradients
    if tf.reduce_sum(tf.abs(grads)).numpy() < 1e-6:
        print("Warning: Gradients are essentially zero.")
        return np.zeros(IMG_SIZE)

    weights = tf.reduce_mean(grads, axis=(0, 1))
    conv_output_tensor = conv_output[0]
    heatmap = conv_output_tensor * weights
    heatmap = tf.reduce_sum(heatmap, axis=-1)
    
    # ReLU to the heatmap (only positive contributions)
    heatmap = tf.maximum(heatmap, 0)
    
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros(IMG_SIZE) 
        
    heatmap = heatmap / max_val
    
    return heatmap.numpy()


def display_gradcam_in_gui(img_path, heatmap, ax, alpha=0.4):
    """
    Overlays the heatmap on the original image and displays it in the Matplotlib axes.
    Ensures the image maintains its aspect ratio (non-distorted fit) and removes the title.
    """
    ax.clear() 
    
    original_img_pil = Image.open(img_path).convert('RGB')
    
    # Convert the heatmap (0-1 float) to a PIL image (0-255 uint8) and resize it
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap)).resize(original_img_pil.size, Image.LANCZOS)
    heatmap_resized = np.array(heatmap_pil)
    
    # Display the original image on the Matplotlib axes
    ax.imshow(original_img_pil)
    
    # Overlay the heatmap.
    ax.imshow(
        heatmap_resized, 
        cmap="jet", 
        alpha=alpha, 
        extent=[0, original_img_pil.width, original_img_pil.height, 0] 
    )
    
    ax.set_title("") # No title
    ax.axis('off') # Hide axes ticks and labels
    ax.set_aspect('equal', adjustable='box') # Key for non-distorted fit
    
    # Redraw the canvas
    ax.figure.canvas.draw_idle()



def load_and_preprocess_image(img_path, target_size):
    """Loads an image, resizes it, crops it, and converts it to a normalized array."""
    print(f"Loading image from: {img_path}")
    
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_tensor = tf.cast(img_array, tf.float32) 
    
    crop_size = 112 
    offset = (IMG_SIZE[0] - crop_size) // 2 
    
    cropped_tensor = tf.image.crop_to_bounding_box(
        img_tensor, 
        offset, 
        offset, 
        crop_size, 
        crop_size 
    )
    
    resized_tensor = tf.image.resize(cropped_tensor, target_size)
    final_image = resized_tensor / 255.0
    final_image = tf.expand_dims(final_image, axis=0) 
    
    return final_image


# --- Prediction and GUI Logic ---

def predict_cataract_gui(image_file_path):
    """Handles the model prediction, Grad-CAM generation, and updates the GUI."""
    global model, image_label, result_label, heatmap_ax

    if model is None:
        result_label.config(text="Model not loaded!")
        return
        
    # --- 1. Load and Preprocess Image ---
    preprocessed_image = load_and_preprocess_image(image_file_path, IMG_SIZE)
    
    try:
        _ = model(preprocessed_image) 
    except Exception as e:
        print(f"Warning: Model forward pass failed. Error: {e}")
    
    # --- 2. Make Prediction ---
    probability_of_normal = model.predict(preprocessed_image)[0][0]
    
    # --- 3. Interpretation ---
    threshold = 0.5
    
    if probability_of_normal >= threshold:
        classification = CLASS_NAMES[1]
        certainty = probability_of_normal * 100
        target_pred_index = 1
    else:
        classification = CLASS_NAMES[0]
        certainty = (1 - probability_of_normal) * 100
        target_pred_index = 0
    
    # --- 4. Generate Heatmap ---
    heatmap = make_gradcam_heatmap(preprocessed_image, model, GRAD_CAM_TARGET_LAYER, target_pred_index)
    
    # --- 5. Update GUI ---
    
    original_img_pil = Image.open(image_file_path)
    
    box_w, box_h = DISPLAY_BOX_SIZE
    img_w, img_h = original_img_pil.size
    
    ratio = min(box_w / img_w, box_h / img_h)
    
    # Calculate new dimensions
    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)

    # Resize the image using the calculated dimensions
    scaled_img_pil = original_img_pil.resize((new_w, new_h), Image.LANCZOS)
    
    # Create a new blank image (the fixed size of the box)
    display_img = Image.new('RGB', DISPLAY_BOX_SIZE, color='white') 
    
    # Calculate offset to center the scaled image
    offset_w = (box_w - new_w) // 2
    offset_h = (box_h - new_h) // 2
    
    # Paste the scaled image onto the center of the display canvas
    display_img.paste(scaled_img_pil, (offset_w, offset_h))
    
    tk_img = ImageTk.PhotoImage(display_img)
    image_label.config(image=tk_img)
    image_label.image = tk_img # Keep a reference
    
    if np.allclose(heatmap, 0): 
        print("\n*** GRAD-CAM FAILED: Could not generate heatmap (all zeros). ***")
        heatmap_ax.clear()
        heatmap_ax.text(0.5, 0.5, "Grad-CAM Failed", ha='center', va='center', fontsize=16)
        heatmap_ax.axis('off')
        heatmap_ax.figure.canvas.draw_idle()
    else:
        display_gradcam_in_gui(image_file_path, heatmap, heatmap_ax)
    
    final_result_text = (
        f"--- CLASSIFICATION RESULT ---\n\n"
        f"Classification: {classification}\n"
        f"Certainty: {certainty:.2f}%\n\n"
        f"Raw Score (Prob. Normal): {probability_of_normal:.4f}\n\n"
        f"Areas highlighted in red/yellow contributed most to the model's decision."
    )
    
    result_label.config(text=final_result_text)


def load_model_once():
    """Load the model and store it globally."""
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("Model loaded and recompiled successfully for GUI.")
        return True
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        print("Please ensure the model file is present.")
        return False


def browse_file_and_predict():
    """Opens a file dialog and starts the prediction process."""
    file_path = filedialog.askopenfilename(
        title="Select an Eye Image",
        filetypes=(("PNG files", "*.png"), ("JPG files", "*.jpg"), ("all files", "*.*"))
    )
    if file_path:
        if model is None:
            result_label.config(text="ERROR: Model failed to load!")
            return
            
        predict_cataract_gui(file_path)

# --- Main GUI Setup ---

def setup_gui():
    global image_label, heatmap_ax, heatmap_canvas, result_label, heatmap_frame
    
    root = tk.Tk()
    root.title("Cataract Classifier with Grad-CAM")
    
    if not load_model_once():
        error_label = tk.Label(root, text=f"ERROR: Model file '{MODEL_PATH}' not found or invalid.", fg="red", font=("Helvetica", 12, "bold"))
        error_label.pack(pady=20)
        ttk.Button(root, text="Upload Image (Model Error)", state='disabled').pack(pady=10)
        root.mainloop()
        return

    # --- Layout Frames ---
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill='both', expand=True)

    image_panel = ttk.Frame(main_frame)
    image_panel.grid(row=0, column=0, padx=10, pady=10, sticky="n")

    heatmap_panel = ttk.Frame(main_frame)
    heatmap_panel.grid(row=0, column=1, padx=10, pady=10, sticky="n")

    # --- Original Image (Left Panel) ---
    ttk.Label(image_panel, text="Original Image", font=("Helvetica", 12, "bold")).pack(pady=5)
    
    placeholder_img = Image.new('RGB', DISPLAY_BOX_SIZE, color='white')
    tk_placeholder = ImageTk.PhotoImage(placeholder_img)
    
    image_label = tk.Label(
        image_panel, 
        image=tk_placeholder, 
        relief="groove",
        width=DISPLAY_BOX_SIZE[0], 
        height=DISPLAY_BOX_SIZE[1]
    )
    image_label.image = tk_placeholder
    image_label.pack()

    # --- Heatmap Image (Right Panel) ---
    ttk.Label(heatmap_panel, text="Grad-CAM Visualization", font=("Helvetica", 12, "bold")).pack(pady=5)
    
    heatmap_frame = tk.Frame(
        heatmap_panel, 
        relief="groove", 
        borderwidth=2, 
        width=DISPLAY_BOX_SIZE[0], 
        height=DISPLAY_BOX_SIZE[1],
        bg="white" # Set a background color for padding around the plot if needed
    )
    heatmap_frame.pack(fill="both", expand=True)
    heatmap_frame.pack_propagate(False) # Prevent frame from resizing to content
    
    # Matplotlib Figure: Set size to match the frame size
    fig, heatmap_ax = plt.subplots(figsize=(DISPLAY_BOX_SIZE[0]/100, DISPLAY_BOX_SIZE[1]/100), dpi=100) 
    
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    heatmap_ax.axis('off')
    heatmap_ax.set_title("") # No title
    
    # Embed Matplotlib figure into the new heatmap_frame
    heatmap_canvas = FigureCanvasTkAgg(fig, master=heatmap_frame)
    heatmap_widget = heatmap_canvas.get_tk_widget()
    heatmap_widget.pack(fill=tk.BOTH, expand=True) 

    # --- Result and Upload (Below Heatmap) ---
    result_frame = ttk.Frame(main_frame)
    result_frame.grid(row=1, column=0, columnspan=2, pady=10)

    upload_button = ttk.Button(result_frame, text="Upload Image & Predict", command=browse_file_and_predict)
    upload_button.pack(pady=10)

    result_label = tk.Label(
        result_frame, 
        text="Click 'Upload Image & Predict' to begin.", 
        font=("Courier", 11), 
        justify=tk.LEFT,
        bg="#f0f0f0", 
        padx=10, 
        pady=10,
        relief="groove"
    )
    result_label.pack(pady=10, padx=10)

    root.mainloop()

if __name__ == "__main__":
    setup_gui()
