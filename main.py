import cv2
import numpy as np
import mimetypes
import os

def colorize_image(image_path):
    """
    Colorizes a black and white image using a pre-trained neural network.

    Args:
        image_path: Path to the black and white image.

    Returns:
        A colorized image.
    """
      
    # Load the pre-trained models
    model_folder_path = os.path.join(app_path, 'models')
    net = cv2.dnn.readNetFromCaffe(os.path.join(model_folder_path , 'colorization_deploy_v2.prototxt'), os.path.join(model_folder_path , 'colorization_release_v2.caffemodel'))
    pts = np.load(os.path.join(model_folder_path , 'pts_in_hull.npy'))

    class8_ab = net.getLayerId('class8_ab')
    conv8_313_rh = net.getLayerId('conv8_313_rh')
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8_ab).blobs = [pts.astype("float32")]
    net.getLayer(conv8_313_rh).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    image = cv2.imread(image_path)
    scaled = image.astype("float32") / 255.0
    # L (Lightness) , A (Green-Red), B (Blue-Yellow)
    lab_color_space = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab_color_space, (224, 224))
    rescaled_l  = cv2.split(resized)[0] - 45

    net.setInput(cv2.dnn.blobFromImage(rescaled_l))
    a_b_channels = net.forward()[0, :, :, :].transpose((1, 2, 0))

    a_b_channels = cv2.resize(a_b_channels, (image.shape[1], image.shape[0]))

    rescaled_l  = cv2.split(lab_color_space)[0]
    colorized = np.concatenate((rescaled_l [:, :, np.newaxis], a_b_channels), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    return (255 * colorized).astype("uint8")


def is_image(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type and mime_type.startswith("image")

def process_images_files(folder_path):
    """Processes all mp3 files in the given folder."""
    for filename in os.listdir(folder_path):
        if is_image(filename):
            file_name, file_extension = os.path.splitext(os.path.basename(filename))
            image_file = os.path.join(folder_path, filename)
            converted_image = os.path.join(folder_path, "colored",  file_name  + "_colorized" + file_extension)
            if not os.path.exists(converted_image):
                colorized = colorize_image(image_file)
                cv2.imwrite(converted_image, (np.clip(colorized, 0, 255)).astype(np.uint8))
                print(f"Converted {image_file} to {converted_image}")

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

app_path = os.getcwd()
images_path = os.path.join(app_path, 'images')

if __name__ == "__main__":
    try:
        check_folder(images_path)
        check_folder(os.path.join(images_path, 'colored'))
        process_images_files(images_path)
    except Exception as e:
        print(f"Error: {e}")
