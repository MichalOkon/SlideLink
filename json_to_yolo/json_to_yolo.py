import json
import os

from PIL import Image


def segmentation_conversion(shape_attributes, img_width, img_height):
    scaled_width = shape_attributes['width'] / img_width
    scaled_height = shape_attributes['height'] / img_height
    scaled_x = shape_attributes['x'] / img_width
    scaled_y = shape_attributes['y'] / img_height

    x1 = scaled_x
    y1 = scaled_y
    x2 = scaled_x + scaled_width
    y2 = scaled_y
    x3 = scaled_x + scaled_width
    y3 = scaled_y + scaled_height
    x4 = scaled_x
    y4 = scaled_y + scaled_height

    return x1, y1, x2, y2, x3, y3, x4, y4


def detection_conversion(shape_attributes, img_width, img_height):
    x_center = (shape_attributes['x'] + shape_attributes['width'] / 2) / img_width
    y_center = (shape_attributes['y'] + shape_attributes['height'] / 2) / img_height
    width = shape_attributes['width'] / img_width
    height = shape_attributes['height'] / img_height

    return x_center, y_center, width, height


def json_to_yolo_format(json_data, image_dir):
    yolo_data = {}

    for key, value in json_data.items():
        # Get the filename without extension
        file_id = value['filename'].split('.')[0]

        # Open the image to get its dimensions
        with Image.open(f'{image_dir}/{value["filename"]}') as img:
            img_width, img_height = img.size

        for region in value['regions']:
            shape_attributes = region['shape_attributes']
            if shape_attributes['name'] == 'rect':

                x_center, y_center, width, height = detection_conversion(shape_attributes, img_width, img_height)

                # Save the converted data
                if file_id not in yolo_data:
                    yolo_data[file_id] = []

                # Append to list of rectangles for this image
                # Assuming a class id of 0 for all rectangles
                yolo_data[file_id].append(f'0 {x_center} {y_center} {width} {height}')
    return yolo_data


if __name__ == '__main__':
    slides = "2ffd6cd6c2634641ba161d8efac204eb1d"
    with open(f'unprocessed_data/{slides}/{slides}_labels_json.json', 'r') as f:
        json_data = json.load(f)

    image_dir = f'unprocessed_data/{slides}/presenter'
    yolo_data = json_to_yolo_format(json_data, image_dir)

    # Write YOLO data to txt files
    if not os.path.exists(f'processed_data/{slides}'):
        os.makedirs(f'processed_data/{slides}')

    for file_id, lines in yolo_data.items():
        with open(f'processed_data/{slides}/{file_id}.txt', 'w') as f:
            f.write('\n'.join(lines))
