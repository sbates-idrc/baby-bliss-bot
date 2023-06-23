# Baby Bliss Bot

An exploratory research project to generate new Bliss vocabulary using machine learning techniques.

[The Bliss language](https://www.blissymbolics.org/) is an Augmentative and Alternative Communication (AAC) language
used by individuals with severe speech and physical impairments around the world, but also by others for language
learning and support, or just for the fascination and joy of this unique language representation. It is a semantic
graphical language that is currently composed of more than 5000 authorized symbols - Bliss-characters and Bliss-words.
It is a generative language that allows its users to create new Bliss-words as needed.

We are exploring the generation of new Bliss vocabulary using emerging AI techniques, including Large Language Models
(LLM), OCR, and other models for text generation and completion.

## Local Installation

### Prerequisites

* [Python 3](https://www.python.org/downloads/)
  * Version 3.9+. On Mac, Homebrew is the easiest way to install.

### Clone the Repository

* Clone the project from GitHub. [Create a fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo)
with your GitHub account, then run the following in your command line (make sure to replace `your-username` with
your username):

```bash
git clone https://github.com/your-username/baby-bliss-bot
cd baby-bliss-bot
```

### Create/Activitate Virtual Environment
Always activate and use the python virtual environment to maintain an isolated environment for project's dependencies.

* [Create the virtual environment](https://docs.python.org/3/library/venv.html)
  (one time setup): 
  - `python -m venv .venv` 

* Activate (every command-line session):
  - Windows: `.\.venv\Scripts\activate`
  - Mac/Linux: `source .venv/bin/activate`

### Install Python Dependencies

Run in the baby-bliss-bot directory:
* `pip install -r requirements.txt`

## Linting

Run the following command to lint all python scripts:

* `flake8`

## Model Experiments

We performed experiments with a number of existing models listed below to understand how useful they are in helping
with generating new Bliss symbols etc.

### StyleGAN3

Conclusion: not useful

Refer to the [documentation](./docs/TrainStyleGAN3MOdel.md) about how to train this model, training results and
the conclusion about how useful it is.

### Texture Inversion

Concolusion: not useful 

Refer to the [documentation](./notebooks/README.md) for details.

## Utility Functions

All utility functions are in `utils` directory.

### Get Bliss single characters (utils/get_bliss_single_chars.py)

This script filters out all Bliss single characters from a directory with all Bliss symbols.

**Usage**: python script_name.py [tsv_file_path] [all_bliss_symbol_dir] [target_dir]

* *tsv_file_path*: The path to the .tsv file to be read. This file contains single characters by BCI IDs
* *all_bliss_symbol_dir*: The path to the directory where all Bliss symbol images are located.
* *target_dir*: The path to the directory where matched symbol images will be copied to.

**Example**: python get_bliss_single_chars.py ~/Downloads/BCI_single_characters.tsv ~/Downloads/h264-0.666-nogrid-transparent-384dpi-bciid ~/Downloads/bliss_single_chars

**Return**: None

### Resize all images to a same height (utils/resize_images_to_same_height.py)

This script resizes all images in a directory to the same height. The resized images are saved into a target directory.

**Usage**: python resize_images_to_same_height.py [image_dir] [target_height] [target_dir]

* *image_dir*: The directory with all images
* *target_height*: The target height to resize all images to
* *target_dir*: The target directory to save resized images

**Example**: python resize_images_to_same_height.py ~/Downloads/bliss_single_chars 216 ~/Downloads/bliss_single_chars_in_height_216

**Return**: None

### Get max image dimensions (utils/get_max_dimensions.py)

This script finds the maximum width and maximum height of all PNG and JPG images in a directory,
along with a list of image filenames that have the maximum width and maximum height.
It also returns the second maximum width and second maximum height, along with their respective
lists of image filenames.

**Usage**: python get_max_dimensions.py [image_directory]

* *image_directory*: The path to the directory containing the images.

**Example**: python get_max_dimensions.py images/

**Return**: tuple: A tuple containing:
* the maximum width (int)
* maximum height (int)
* a list of filenames of images with maximum width (list)
* a list of filenames of images with maximum height (list)
* the second maximum width (int), the second maximum height (int)
* a list of filenames of images with the second maximum width (list)
* a list of filenames of images with the second maximum height (list)

### Scale down images (utils/scale_down_images.py)

This script scales down JPG and PNG images in a directory to a specified size while maintaining their aspect ratios. 
The output images are saved in a new directory. If the output directory doesn't exist, it will be created.

**Usage**: python scale_down_images.py [input_dir] [output_dir] [new_size]

* *input_dir*: The directory where the original images are located.
* *output_dir*: The directory where the output images will be saved.
* *new_size*: The desired size of the scaled down images, in the format "widthxheight".

**Example**: python scale_down_images.py images/ scaled_down_images/ 128x128

**Return**: None

### Sync up image sizes (utils/image_size_sync.py)

This script synchronizes the size of all PNG and JPG files in the input directory.
It first finds the maximum dimension (either width or height) among all the input images.
Then it loops through the image directory to perform these operations for every image:
1. Transform the image to grayscale and find the background color of this image using the color code at the pixel
(1, 1);
2. Create a square canvas with the maximum dimension as its width and height. The color of the canvas is the background
color observed at the previous step;
3. Copy each input image onto the center of the canvas, without changing the size of the input image. This ensures that
each output image has the same maximum dimension and is centered in the canvas. 
Finally, all output images are saved in the specified output directory.

**Usage**: python image_size_sync.py [input_dir] [output_dir]

* *input_dir*: The directory where the original images are located.
* *output_dir*: The directory where the output images will be saved.

**Example**: python image_size_sync.py images/ output/

**Return**: None

## Notebooks

[`/notebooks`](./notebooks/) directory contains all notebooks used for training or fine-tuning various models.
Each notebook usually comes with a accompanying `dockerfile.yml` to elaborate the environment that the notebook was
running in.

## Jobs
[`/jobs`](./jobs/) directory contains all jobs used for training or fine-tuning various models.
