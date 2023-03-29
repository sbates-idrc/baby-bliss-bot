# Baby Bliss Bot

An exploratory research to generate new Bliss vocabulary.

[Bliss language](https://www.blissymbolics.org/) is an Augmentative and Alternative Communication (AAC) language
used by individuals with severe speech and physical impairments around the world, but also by others for language
learning and support, or just for the fascination and joy of this unique language representation. It is a semantic
graphical language that is currently composed of more than 5000 authorized symbols - Bliss-characters and Bliss-words.
It is a generative language that allows its users to create new Bliss-words as needed.

The generation of new vocabulary situates in the context of an exploratory research that could leverage from emerging
trends in AI methods and technologies, including the use of Large Language Models (LLM) such as GPT and other models
for text generation and completion.

## Local Installation

### Prerequisites

* [Python 3](https://www.python.org/downloads/)
  * Version 3.9+. On Mac, Homebrew is the easiest way to install.
* [virtualenv](https://virtualenv.pypa.io/en/latest/) 
  * Not required, but highly recommended for maintaining an isolated environment for Python and Python packages.

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
  - Or do this via your IDE, e.g. [Intellij's support for virtualenv](https://www.jetbrains.com/help/idea/creating-virtual-environment.html)

* Activate (every command-line session):
  - Windows: `.\.venv\Scripts\activate`
  - Mac/Linux: `source .venv/bin/activate`

### Install Python Dependencies

Run in the baby-bliss-bot directory:
* `pip install -r requirements.txt`

## Linting

Run the following command to lint all python scripts:

* `flake8`

## Utility Functions

All utility functions are in `utils` directory.

### Resize images (utils/resize_images.py)

This script resizes JPG and PNG images in a directory to a specified size while maintaining their aspect ratios. 
The resized images are saved in a new directory. If the output directory doesn't exist, it will be created.

**Usage**: python resize_images.py [input_dir] [output_dir] [new_size]

*input_dir*: The directory where the original images are located.
*output_dir*: The directory where the resized images will be saved.
*new_size*: The desired size of the resized images, in the format "widthxheight".

**Example**: python resize_images.py images/ resized_images/ 128x128

**Returns**: None
