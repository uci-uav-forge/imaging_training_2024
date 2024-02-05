from collections import defaultdict
import tensorflow as tf
import pandas as pd
import keras

LABEL_TO_COLOR_DICT = {
    0: "red",
    1: "orange",
    2: "yellow",
    3: "green",
    4: "blue",
    5: "purple",
    6: "white",
    7: "black",
    8: "brown",
    9: "gray",
}


# change test dataset directory here
test_directory = './test/dataset'
# change file name here
test_df = pd.read_csv(test_directory + '/labels.txt')

def test_read_image(image_file):
    image = tf.io.read_file(test_directory + image_file)
    image = tf.image.decode_image(image, channels=3, dtype=tf.float32)  #channel=1 means grayscale
    return image

test_file_paths = test_df['file'].values
# test_labels = test_df[' label'].values
ds_test = tf.data.Dataset.from_tensor_slices(test_file_paths)
ds_test = ds_test.map(test_read_image).batch(10)

model = tf.keras.models.load_model('trained_model')
model.summary()
# model.evaluate(ds_test)

result = model.predict(ds_test)

shape_predict = result[0]
letter_predict = result[1]


predict_shape_color = tf.argmax(shape_predict, axis=1).numpy()
print(predict_shape_color)
print(predict_shape_color.shape)
predict_letter_color = tf.argmax(letter_predict, axis=1).numpy()

total = len(predict_shape_color)
total_correct_shape_color = sum(predict_shape_color == test_df[' shape_color'].values)
total_correct_letter_color = sum(predict_letter_color == test_df[' letter_color'].values)
def evaluate_colors():
    global total, total_correct_shape_color, total_correct_letter_color
    
    # Create a dictionary to store accuracies
    accuracies = defaultdict(list)
    
    # Calculate accuracies for each shape-letter color combination
    for index, row in test_df.iterrows():
        shape_color = row[' shape_color']
        letter_color = row[' letter_color']
        if len(accuracies[f"{LABEL_TO_COLOR_DICT[shape_color]}_{LABEL_TO_COLOR_DICT[letter_color]}"]) == 0:
            accuracies[f"{LABEL_TO_COLOR_DICT[shape_color]}_{LABEL_TO_COLOR_DICT[letter_color]}"] = [0,0]
        # Calculate shape color accuracy
        shape_color_correct = (predict_shape_color[index] == shape_color)
        if shape_color_correct:
            accuracies[f"{LABEL_TO_COLOR_DICT[shape_color]}_{LABEL_TO_COLOR_DICT[letter_color]}"][0] += 1
        accuracies[f"{LABEL_TO_COLOR_DICT[shape_color]}_{LABEL_TO_COLOR_DICT[letter_color]}"][1] += 1

        # Calculate letter color accuracy
        letter_color_correct = (predict_letter_color[index] == letter_color)
        if letter_color_correct:
            accuracies[f"{LABEL_TO_COLOR_DICT[shape_color]}_{LABEL_TO_COLOR_DICT[letter_color]}"][0] += 1
        accuracies[f"{LABEL_TO_COLOR_DICT[shape_color]}_{LABEL_TO_COLOR_DICT[letter_color]}"][1]+=1

    # Print and return the accuracies
    for key, value in accuracies.items():
        accuracy = value[0] / value[1]
        print(f"{key} Accuracy:", accuracy)
    
    total = len(predict_shape_color)
    total_correct_shape_color = sum(predict_shape_color == test_df[' shape_color'].values)
    total_correct_letter_color = sum(predict_letter_color == test_df[' letter_color'].values)
    print(f"Shape Accuracy: {total_correct_shape_color/total}")
    print(f"Letter Accuracy: {total_correct_letter_color/total}")
    return accuracies

accuracies = evaluate_colors()
print("Accuracies:", accuracies)