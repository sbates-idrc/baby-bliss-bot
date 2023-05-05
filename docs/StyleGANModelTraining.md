# Train StyleGAN model

[StyleGAN model](https://machinelearningmastery.com/introduction-to-style-generative-adversarial-network-stylegan/) 
offers control over the style of the generated image. Bliss has 1217 single characters that are used to compose other
Bliss words. It is interesting to train a StyleGAN model with these single characters to find out if it is useful to
give us some new bliss shapes, and to give the whole team a feel for what these systems can and cannot do.

## Prepare the image set

Goal: The image set contains all Bliss single characters. All images need to be transformed to grayscale. They should
also be cropped or padded out to make a square of 256 x 256 with the baseline always in the same position, and the
symbol centered horizontally.

Note: The Bliss single character with BCI ID 25600 is missing from the final image set. According to the information
from [Xelify](https://www.xelify.se/blissfiles/), this character is missing on purpose because it has been decided as
part of the work with encoding Blissymbolics into Unicode. This particular character will not be part of Unicode. It's
because it doesn't stand for a concept, it's just a description of a graphical shape used in the Bliss character for
squirrel.

### Steps

1. At [the Xelify Bliss File page](https://www.xelify.se/blissfiles/), download the png package with height 344px,
transparent background, 384dpi resolution and the naming of BCI ID.

2. Export [the Bliss single character](https://docs.google.com/spreadsheets/d/1t1x1UFuJC1hpjrxdXKi19Tk_Tv-9GVQWSA4sN2FScv4/edit#gid=138588066) spreadsheet as tab separated file.

3. Filter out all Bliss single characters from the downloaded png package into a directory.
```
cd utils

// Filter out all Bliss single characters
python get_bliss_single_chars.py ~/Downloads/BCI_single_characters.tsv ~/Downloads/h264-0.666-nogrid-transparent-384dpi-bciid ~/Downloads/bliss_single_chars
Error: 25600.png not found in /Users/cindyli/Downloads/h264-0.666-nogrid-transparent-384dpi-bciid
```

4. Scan through all single characters to find the maximum dimension.
```
// Find the maximum dimensions
python get_max_dimensions.py ~/Downloads/bliss_single_chars

Results:
The max width is:  313
The list of images with the max width is:  ['26057.png', '14958.png', '24281.png', '22625.png', '17999.png', '26049.png']
The max height is:  264
The second max width is:  289
The list of images with the second max width is:  ['13090.png']
The second max height is:  0
The list of images with the second max height is:  []
```

5. Resize images with the max width 313px to a width of 256px. Since all resized images need to be in the same height 
with the max dimension of 256px, it results in the calculation of the height:
```
max_height = 256 * 264 / 313 = 215.92
```

6. Resize all single character images to a height of 216px.
```
// Resize
python resize_images_to_same_height.py ~/Downloads/bliss_single_chars 216 ~/Downloads/bliss_single_chars_in_height_216

// Check the max dimension of resized images
python get_max_dimensions.py ~/Downloads/bliss_single_chars_in_height_216

Results:
The max width is:  256
The list of images with the max width is:  ['26057.png', '14958.png', '24281.png', '22625.png', '17999.png', '26049.png']
The max height is:  216
The second max width is:  236
The list of images with the second max width is:  ['13090.png']
The second max height is:  0
The list of images with the second max height is:  []

The verification shows the resizing is correct.
```

7. Transform all images to grayscale. Pad out all images with the background in the same background color as the
grayscaled image to make a square of 256X256. All images are centred horizontally.
```
// Pad out all images
python image_size_sync.py ~/Downloads/bliss_single_chars_in_height_216 ~/Downloads/bliss_single_chars_final

// Verify the max dimension of final images
python get_max_dimensions.py ~/Downloads/bliss_single_chars_final
Results:
The max width is:  256
The second max width is:  0
The list of images with the second max width is:  []
The second max height is:  0
The list of images with the second max height is:  []

The verification shows the resizing is correct.
```

## Train the styleGAN3 model

The styleGAN3 model is trained on [the Cedar platform](https://docs.alliancecan.ca/wiki/Cedar).

### Start the training job

Step 1: Use [`rsync`](https://linuxhandbook.com/transfer-files-ssh/) or other commands to transfer transformed Bliss
images to Cedar

Step 2. Login to the Cedar and fetch stylegan3 source code
```
mkdir stylegan3
cd stylegan3
git clone https://github.com/NVlabs/stylegan3
```

Step 3. Creating a zip archive of Bliss images will lead to a better performance.
```
cd stylegan3
python dataset_tool.py --source=../bliss_single_chars_final --dest=../datasets/bliss-256x256.zip
```

Step 4: Submit Job

* Copy [requirements.txt](../jobs/stylegan3/requirements.txt) to stylegan3 source code root directory.

* Copy [job_stylegan3.sh](../jobs/stylegan3/job_stylegan3.sh) to the `scratch/' directory in your home directory

* Submit the job

```
cd ~/scratch
sbatch job_stylegan3.sh
```

Use `sq` to check the status of the job. Use `scancel` to cancel a running job.

### The training result

The Bliss images are trained using `stylegan3-r` model (translation and rotation equiv.). The training result can
be found at [this repository](https://github.com/cindyli/bliss-data/tree/main/styleGAN/styleGAN-training-results/stylegan3-r).

`reals.png` is a collection of real Bliss symobles

`fakes*.png` are random image grids exported from the training loop at regular intervals.

`training_options.json` contains training options used for this round of training.

`metric-fid50k_full.jsonl` logs the result and records` FID evaluated by the training loop for every export.
