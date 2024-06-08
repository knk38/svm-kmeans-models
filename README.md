The instructions for setting up, running, and testing the different algorithms are separated by dataset. 


# Dataset 1: Spotify Song Popularity Classifier
## TASK: Given features about each song predict the number of streams within a certain range.



# Dataset 2: Car Make Classifier
## TASK: Given images of cars classify them into the make (manufacturer)

## Requirements

- Python 3.6+
- pandas
- torch
- torchvision
- Pillow
- matplotlib
- scikit-learn

## Setup and Usage

1. **Prepare Data:**
   - This is only required when the data is being downloaded for the first time
        the box folder should contain this data already prepared. 
   - Ensure your data is in the following structure:
     - `anno_train.csv` and `train/` for training data.
     - `anno_test.csv` and `test/` for test data.
   - Merge and label the data by running:
     ```sh
     python data_format.py
     ```
   - This will create `merged_dataset.csv` and a folder named `train`.

2. **Verify Data:**
   - Run the viewer script from the terminal to verify the merged data:
     ```sh
     python viewr.py
     ```

3. **Train the Model:**
   - Train the model and generate plots by running:
     ```sh
     python train.py
     ```
   - Select the tests to be run in the `main()` function of the `train.py` script. 
   - Te corresponding plots will be generated once training is completed.