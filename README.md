# SlideLink: Matching Slides with Recordings

The COVID-19 pandemic has brought significant changes to the education landscape, leading to the adoption of alternative teaching methods, including recorded lectures and live streams. However, a common challenge faced by students and viewers is the poor quality of some recordings, making it difficult to see the displayed slides. This limitation affects the learning experience and hampers students' ability to stay focused.

To address this issue, we present SlideLink, a tool designed to match slides with recordings and enhance the viewer's experience. By automatically pairing the blurry images of slides from the recording with clear slides from the slide deck, SlideLink improves the clarity and accessibility of educational content.

This project tackles a relatively unexplored area, with limited previous work on slide-to-recording synchronization. It involves the utilization of multiple Deep Learning models and solutions to effectively identify the relevant slide information and infer the currently displayed slide.

Existing research in the field of enhancing lectures has primarily focused on improving the quality from the student's perspective. Techniques such as observing student activities and tracking facial expressions have been employed in virtual environments. However, our approach differs as we aim to enhance the user experience from the lecturer's standpoint.

Other methods have explored non-verbal behavior prediction and Optical Character Recognition (OCR) for extracting text from slides. However, OCR faces challenges with low-quality images and non-textual content, which are common in slide decks. Our project seeks to overcome these limitations and accommodate various types of footage captured in a video classroom setting.

The proposed solution consists of two main stages utilizing Computer Vision techniques. In the first stage, we employ the YOLOv8 model for object detection to identify the projection screen in the lecture recordings. In the second stage, we use the LoFTR model to match the recorded frames with corresponding slides from the slide deck. The combination of these two stages allows us to synchronize the slides with the recordings effectively.

## Installation

Before running the project with its tools e.g. CLI, you must follow the instructions below.

## Python setup with the virtual environment

To create a Python virtual environment and install all the dependencies from a requirements.txt file, follow these instructions:

1. Open your command-line interface (e.g., Terminal, Command Prompt).
2. Navigate to the directory where you want to create your virtual environment. Use the `cd` command to change directories. For example:

```shell
$ cd /path/to/project/directory
```

3. Create a new virtual environment using the `venv` module. Run the following command:
   ```
   python3 -m venv myenv
   ```
   Replace `myenv` with the desired name for your virtual environment.

4. Activate the virtual environment. The commands to activate the virtual environment depend on your operating system:

   - For Windows (Command Prompt):

```shell
# Activate the virtual environment
$ myenv\Scripts\activate.bat
```

   - For Windows (PowerShell):

```shell
# Activate the virtual environment
$ myenv\Scripts\Activate.ps1
```

   - For Linux/macOS:
  
```shell
# Activate the virtual environment
$ source myenv/bin/activate
```

Once activated, you should see the virtual environment's name in your command-line prompt.
5. Now that you are inside the virtual environment, you can install the dependencies from the requirements.txt file. Run the following command:

```shell
# Install all dependencies
$ pip install -r requirements.txt
```

This command will install all the packages listed in the requirements.txt file into your virtual environment.

6. Wait for the installation process to complete. Once finished, all the required packages should be installed in your virtual environment.
7. You can now start working with your Python project using the installed dependencies within the virtual environment.
8. When you're done working in the virtual environment, you can deactivate it. Run the following command:

```shell
# Deactivate virtual environment
$ deactivate
```

This command will return you to your system's default Python environment.

Remember to activate the virtual environment every time you work on your project, as it ensures that you are using the correct dependencies and prevents conflicts with other Python projects on your system.

### Chromium Web Driver

You need to install Chrome and [Chromium Web Drivers](https://skolo.online/documents/webscrapping/#step-2-install-chromedriver).


## Usage

Here you can use these commands to run the project tools. Make sure you are using the virtual environment.

### Web Scrapping Collegerama

To scrape the data from Collegerama website, make sure you create two files in `data_fetch` directory:

- `secrets.txt`
- `urls.txt`

The `secrets.txt` must contain your TU Delft login credentials in the following format:

```text
<netid>
<password>
```

The `urls.txt` file must contain URLs to Collegerama lecture, XPath index of slides view (the last number in the XPath) and XPath index of presenter view (the last number in the XPath) in the following format:

```text
<url> <slide_index> <presenter_index>
............<more resources>............
<url> <slide_index> <presenter_index>
```

For example:

```text
https://collegerama.tudelft.nl/Mediasite/Channel/eemcs-msc-cs/watch/26e9f17037c74a07a16d2c647db4508b1d 1 2
```

The XPath is for slides is `//*[@id="vjs_video_3"]/div[7]/div[10]/div[1]/div[1]` and XPath is for slides is `//*[@id="vjs_video_3"]/div[7]/div[10]/div[1]/div[2]`.

After that, you can use the scraper by running:

```shell
# activate scraping from Collegerama
$ cd data_fetch
$ python main.py
```

After scraping is done, use [VIA tool](https://www.robots.ox.ac.uk/~vgg/software/via/) to label where the projected slides are on the image.

## CLI

On top of the scraper, you can use the CLI program. To understand the CLI options use:

```shell
# Help options
$ python main.py --help
```

### Running models

Firstly, you need to prepare the scrapped data so that they can be used by the models. To do that use:

```shell
# Prepare data for models
$ python main.py prepare-data
```

Then you can train your models by using:

```shell
# Train a model
$ python main.py train {yolo|maskrcnn|all} --epochs {epoch number} --weights [coco|imagenet|none]
```

The `weights` option is ONLY for maskrcnn model.

After that, you can either evaluate the models on the test set or crop out the detected slides projected slides using:

```shell
# Evaluate the models
$ python main.py evaluate {yolo|maskrcnn|none} --model-path {path-to-the-trained-and-saved-model}

# Crop out slides
$ python main.py evaluate {yolo|maskrcnn|none} --model-path {path-to-the-trained-and-saved-model}
```

Once cropped-out slides are generated, you can match them using LoFTR by running the following command:

```shell
# Match crops
$ python main.py match-crops {yolo|maskrcnn|non} {indoor|outdoor}
```
