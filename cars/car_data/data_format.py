import os
import csv
import shutil

# Define the paths to the test and train directories
test_dir = 'test'
train_dir = 'train'
anno_test_file = 'anno_test.csv'
anno_train_file = 'anno_train.csv'

# Define the output directory and CSV file
data_dir = 'data'
output_csv = 'merged_dataset.csv'

# Create the data directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Function to gather image data from a directory and return the highest image number
def gather_data(directory, offset=0):
    data = []
    highest_number = offset
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith('.jpg'):
                    picture_number = int(filename.split('.')[0])
                    new_number = picture_number + offset
                    new_filename = f"{new_number:05d}.jpg"
                    data.append([new_filename, class_name, picture_number, filename, class_path])
                    if new_number > highest_number:
                        highest_number = new_number
    return data, highest_number

# Function to load annotation data
def load_annotations(file_path):
    annotations = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            filename, bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_id = row
            annotations[filename] = [bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_id]
    return annotations

# Load annotation data
anno_train = load_annotations(anno_train_file)
anno_test = load_annotations(anno_test_file)

# Gather data from the train directory
train_data, highest_train_number = gather_data(train_dir)

# Sort train data
train_data.sort(key=lambda x: x[2])  # Sort by original picture number

# Gather data from the test directory, offsetting the image numbers
test_data, _ = gather_data(test_dir, highest_train_number)

# Merge the data
all_data = train_data + test_data

# Sort the merged data by image_name
all_data.sort(key=lambda x: int(x[0].split('.')[0]))

# Extract makes and assign unique numbers
make_dict = {}
make_counter = 0

for row in all_data:
    new_filename, class_name, original_number, original_filename, original_path = row
    make = class_name.split()[0]
    if make not in make_dict:
        make_dict[make] = make_counter
        make_counter += 1

# Copy images to the data directory and write the CSV file
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_name', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'make', 'make_id', 'class_name', 'class_id'])  # Write the header
    
    for data in all_data:
        new_filename, class_name, original_number, original_filename, original_path = data
        original_image_name = f"{original_number:05d}.jpg"
        src_path = os.path.join(original_path, original_filename)
        dest_path = os.path.join(data_dir, new_filename)
        
        # Copy the image to the data directory only if the destination file doesn't exist
        if not os.path.exists(dest_path):
            shutil.copy(src_path, dest_path)
        
        # Get the annotation data
        if original_image_name in anno_train:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_id = anno_train[original_image_name]
        elif original_image_name in anno_test:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_id = anno_test[original_image_name]
        else:
            continue
        
        # Write the data to the CSV file
        make = class_name.split()[0]
        make_id = make_dict[make]
        writer.writerow([new_filename, bbox_x1, bbox_y1, bbox_x2, bbox_y2, make, make_id, class_name, class_id])

print(f"CSV file '{output_csv}' created successfully, and images copied to '{data_dir}'.")
