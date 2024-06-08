import os
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define the paths
data_dir = 'data'
csv_file = 'merged_dataset.csv'

# Function to load data from CSV file
def load_data(csv_file):
    data = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            data.append(row)
    return data

# Function to display an image by its number
def display_image_by_number(image_number, data, base_dir):
    image_number_str = f"{image_number:05d}.jpg"
    for row in data:
        image_name, bbox_x1, bbox_y1, bbox_x2, bbox_y2, make, make_id, class_name, class_id = row
        if image_name == image_number_str:
            image_path = os.path.join(base_dir, image_name)
            if os.path.isfile(image_path):
                img = mpimg.imread(image_path)
                plt.imshow(img)
                plt.title(
                    f"Image: {image_name}\n"
                    f"Make: {make} (ID: {make_id})"
                )
                plt.axis('off')
                plt.show()
                return
    print(f"Image number {image_number_str} not found.")

# Load data from CSV
data = load_data(csv_file)

# Prompt user for image number to display
while True:
    try:
        user_input = input("Enter image number (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        image_number = int(user_input)
        display_image_by_number(image_number, data, data_dir)
    except ValueError:
        print("Invalid input. Please enter a valid image number or type 'exit' to quit.")
