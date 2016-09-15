#! /usr/bin/env python

import sys
import math

training_files = ["Data/ocr_fold0_sm_train.txt", "Data/ocr_fold1_sm_train.txt", "Data/ocr_fold2_sm_train.txt", "Data/ocr_fold3_sm_train.txt", "Data/ocr_fold4_sm_train.txt", "Data/ocr_fold5_sm_train.txt", "Data/ocr_fold6_sm_train.txt", "Data/ocr_fold7_sm_train.txt", "Data/ocr_fold8_sm_train.txt", "Data/ocr_fold9_sm_train.txt"]

test_files = ["Data/ocr_fold0_sm_test.txt", "Data/ocr_fold1_sm_test.txt", "Data/ocr_fold2_sm_test.txt", "Data/ocr_fold3_sm_test.txt", "Data/ocr_fold4_sm_test.txt", "Data/ocr_fold5_sm_test.txt", "Data/ocr_fold6_sm_test.txt", "Data/ocr_fold7_sm_test.txt", "Data/ocr_fold8_sm_test.txt", "Data/ocr_fold9_sm_test.txt"]

VECTOR_SIZE = 128 
TRAINING_ITERATIONS = 1
if len(sys.argv) > 1:
    TRAINING_ITERATIONS = int(sys.argv[1])

learning_rate = 1

correct_predictions = 0
wrong_predictions = 0

vowels = ['a', 'e', 'i', 'o' ,'u']

weight_vector = [0] * VECTOR_SIZE

def dot_product(vector1, vector2):
    result = 0
    for i in xrange(VECTOR_SIZE):
        result += vector1[1] * vector2[i]
    return result


def modulus(vector):
    sum_of_squares = 0
    for x in vector:
        sum_of_squares += x*x
    
    return math.sqrt(sum_of_squares) 
    
def perceptron_train(train_vector, value):
    prediction = dot_product(weight_vector, train_vector)
    if (prediction * value) <= 0:
        for i in xrange(VECTOR_SIZE):
            weight_vector[i] += learning_rate * (value * train_vector[i])

def passive_aggressive_train(train_vector, value):
    prediction = dot_product(weight_vector, train_vector)
    learning_rate = (1 - value * (dot_product(weight_vector, train_vector))) / (modulus(train_vector) ** 2)
    if (prediction * value) <= 0:
        for i in xrange(VECTOR_SIZE):
            weight_vector[i] += learning_rate * (value * train_vector[i])

def test(train_vector, value):
    global correct_predictions, wrong_predictions
    prediction = dot_product(weight_vector, train_vector)
#    print prediction, value
    if (prediction * value) <= 0:
        wrong_predictions += 1
    else:
        correct_predictions += 1


def convert_string_to_int_list(pixel_values):
    pixel_values = pixel_values[2:]
    pixel_vector = []
    for x in pixel_values:
        pixel_vector.append(int(x))
    return pixel_vector    

#Train the classifier
for x in xrange(TRAINING_ITERATIONS):
	for file_path in training_files:
	    f = open(file_path, 'r')

	    for line in f:
		if len(line) > 4:
		    line_split = line.strip().split('\t')
		    pixel_values = line_split[1]
		    pixel_vector = convert_string_to_int_list(pixel_values)
                    value = -1
                    if line_split[2] in vowels:
                        value = 1
		    passive_aggressive_train(pixel_vector, value)

	    f.close()

print "TRAINING DONE"

#Test the classifier
for test_file_path in training_files:
    f = open(test_file_path, 'r')

    for line in f:
        if len(line) > 4:
            line_split = line.strip().split('\t')
            pixel_values = line_split[1]
            pixel_vector = convert_string_to_int_list(pixel_values)
            value = -1
            if line_split[2] in vowels:
                value = 1
            test(pixel_vector, value)

    f.close()

print weight_vector
print wrong_predictions, correct_predictions
