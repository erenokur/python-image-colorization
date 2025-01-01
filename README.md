# python-image-colorization

This project provides a script to colorize black-and-white images using a pre-trained neural network. The program processes grayscale images in a specified folder, colorizes them, and saves the results in a subfolder. It leverages OpenCV's DNN module and a Caffe model for image colorization.

### Installing

Model files are too large you need to use LFS to download them. You can install LFS using the following command:

```bash
git lfs install
```

Or you install manually on Git LFS, please visit [Git LFS website](https://git-lfs.github.com/).

Ensure the following Python libraries are installed:

- opencv-python
- numpy

You can install them using pip:

```bash
pip install -r requirements.txt
```

Note: My Python version is 3.8.4. You may need to install the required libraries with the appropriate version.

### Resources

he following files are required and should be placed in a models folder within the root directory:

1. colorization_deploy_v2.prototxt - Model architecture file.

2. colorization_release_v2.caffemodel - Pre-trained weights.

3. pts_in_hull.npy - Cluster center data for colorization.

### Feedback

If you have any feedback about the project, please let me know. I am always looking for ways to improve the user experience.
