#! /usr/bin/env python

file_path = "Data/ocr_fold1_sm_train.txt"

test_file_path = "Data/ocr_fold1_sm_test.txt"

VECTOR_SIZE = 128 

LEARNING_RATE = 1

correct_predictions = 0
wrong_predictions = 0

vowels = ['a', 'e', 'i', 'o' ,'u']

weight_vector = [0] * VECTOR_SIZE

def dot_product(vector1, vector2):
    return sum([x * y for x,y in zip(vector1, vector2)])
    
def adjust_weight_vector(train_vector, value):
    for i in xrange(VECTOR_SIZE):
        weight_vector[i] += LEARNING_RATE * (value * train_vector[i])

def train(train_vector, value):
    if value in vowels:
        yt = 1
    else:
        yt = -1

    prediction = dot_product(weight_vector, train_vector)
    print prediction, yt
    if (prediction * yt) <= 0:
        adjust_weight_vector(train_vector, yt)

def test(train_vector, value):
    global correct_predictions, wrong_predictions
    if value in vowels:
        yt = 1
    else:
        yt = -1

    prediction = dot_product(weight_vector, train_vector)
    print prediction, yt
    if (prediction * yt) <= 0:
        wrong_predictions += 1
    else:
        correct_predictions += 1


def convert_string_to_int_list(pixel_values):
    pixel_values = pixel_values[2:]
    pixel_vector = []
    for x in pixel_values:
        pixel_vector.append(int(x))
    return pixel_vector    

f = open(file_path, 'r')

for line in f:
    if len(line) > 4:
        line_split = line.strip().split('\t')
        pixel_values = line_split[1]
        pixel_vector = convert_string_to_int_list(pixel_values)
        train(pixel_vector, line_split[2])

f.close()

print "TRAINING DONE"

f = open(test_file_path, 'r')

for line in f:
    if len(line) > 4:
        line_split = line.strip().split('\t')
        pixel_values = line_split[1]
        pixel_vector = convert_string_to_int_list(pixel_values)
        test(pixel_vector, line_split[2])

f.close()

print weight_vector
print wrong_predictions, correct_predictions
