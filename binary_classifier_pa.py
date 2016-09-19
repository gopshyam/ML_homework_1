#! /usr/bin/env python

import sys
import math
import matplotlib.pyplot as plt

training_files = ["Data/ocr_fold0_sm_train.txt", "Data/ocr_fold1_sm_train.txt", "Data/ocr_fold2_sm_train.txt", "Data/ocr_fold3_sm_train.txt", "Data/ocr_fold4_sm_train.txt", "Data/ocr_fold5_sm_train.txt", "Data/ocr_fold6_sm_train.txt", "Data/ocr_fold7_sm_train.txt", "Data/ocr_fold8_sm_train.txt", "Data/ocr_fold9_sm_train.txt"]

test_files = ["Data/ocr_fold0_sm_test.txt", "Data/ocr_fold1_sm_test.txt", "Data/ocr_fold2_sm_test.txt", "Data/ocr_fold3_sm_test.txt", "Data/ocr_fold4_sm_test.txt", "Data/ocr_fold5_sm_test.txt", "Data/ocr_fold6_sm_test.txt", "Data/ocr_fold7_sm_test.txt", "Data/ocr_fold8_sm_test.txt", "Data/ocr_fold9_sm_test.txt"]

VECTOR_SIZE = 128 
vowels = ['a', 'e', 'i', 'o' ,'u']

TRAINING_ITERATIONS = 50
if len(sys.argv) > 1:
    TRAINING_ITERATIONS = int(sys.argv[1])

learning_mistakes_per_fold = []
learning_successes_per_fold = []
training_mistakes_per_fold = []
training_successes_per_fold = []
training_accuracy_per_fold = []
testing_mistakes_per_fold = []
testing_successes_per_fold = []
testing_accuracy_per_fold = []

def parse_file_line(line):
    if len(line) < 4:
        return None
    line_split = line.strip().split('\t')
    pixel_values = line_split[1]
    pixel_vector = convert_string_to_int_list(pixel_values)
    y_hat = -1
    if line_split[2] in vowels:
        y_hat = 1
    return pixel_vector, y_hat


def dot_product(vector1, vector2):
    result = 0
    for i in xrange(VECTOR_SIZE):
        result += vector1[i] * vector2[i]
    return result


def modulus(vector):
    sum_of_squares = 0
    for x in vector:
        sum_of_squares += x*x
    
    return math.sqrt(sum_of_squares) 
    
def passive_aggressive_train(train_vector, y_hat):
    global learning_mistakes, learning_successes
    prediction = dot_product(weight_vector, train_vector)
    learning_rate = (1 - y_hat * (dot_product(weight_vector, train_vector))) / (modulus(train_vector) ** 2)
    if (prediction * y_hat) < 1:
        learning_mistakes += 1
        for i in xrange(VECTOR_SIZE):
            weight_vector[i] += learning_rate * (y_hat * train_vector[i])
    else:
        learning_successes += 1

def test(train_vector, y_hat):
    global testing_successes, testing_mistakes
    prediction = dot_product(weight_vector, train_vector)
    if (prediction * y_hat) <= 0:
        testing_mistakes += 1
    else:
        testing_successes += 1


def convert_string_to_int_list(pixel_values):
    pixel_values = pixel_values[2:]
    pixel_vector = []
    for x in pixel_values:
        pixel_vector.append(int(x))
    return pixel_vector    

def average_per_fold(fold_list):
    #accepts a list of lists containing y_hats of different interations per fold, and returns a list of averages
    average_array = []
    for i in range(len(fold_list[0])):
        sum_of_elements = 0
        for j in range(len(fold_list)):
            sum_of_elements += fold_list[j][i]
        average_array.append(sum_of_elements / len(fold_list))
    return average_array
        

for fold in range(len(training_files)):
    weight_vector = [0] * VECTOR_SIZE
    cached_weight_vector = [0] * VECTOR_SIZE
    weight_vector_array = []
    cached_weight_vector_array = []
    learning_mistakes_array = []
    learning_successes_array = []
    training_mistakes_array = []
    training_successes_array = []
    training_accuracy_array = []
    testing_mistakes_array = []
    testing_successes_array = []
    testing_accuracy_array = []

    for x in range(TRAINING_ITERATIONS):
        learning_mistakes = 0
        learning_successes = 0
        training_mistakes = 0
        training_successes = 0
        testing_mistakes = 0
        testing_successes = 0

        with open(training_files[fold], 'r') as f:
            for line in f:
                if len(line) > 4:
                    pixel_vector, y_hat = parse_file_line(line)
                    passive_aggressive_train(pixel_vector, y_hat)

        weight_vector_array.append(weight_vector[:])
        learning_mistakes_array.append(learning_mistakes)
        learning_successes_array.append(learning_successes)

        #Calculate accuracy on training data
        final_weight_vector = weight_vector_array[-1]
        with open(training_files[fold], 'r') as f:
            for line in f:
                if len(line) > 4:
                    pixel_vector, y_hat = parse_file_line(line)
                    if dot_product(final_weight_vector, pixel_vector) < 0:
                        training_mistakes += 1
                    else:
                        training_successes += 1

        training_mistakes_array.append(training_mistakes)
        training_successes_array.append(training_successes)
        training_accuracy_array.append((training_successes * 100) / float(training_mistakes + training_successes))

        with open(test_files[fold], 'r') as f:
            for line in f:
                if len(line) > 4:
                    pixel_vector, y_hat = parse_file_line(line)
                    test(pixel_vector, y_hat)

        testing_mistakes_array.append(testing_mistakes)
        testing_successes_array.append(testing_successes)
        testing_accuracy_array.append((testing_successes * 100) / float(testing_mistakes + testing_successes))

    learning_mistakes_per_fold.append(learning_mistakes_array)
    learning_successes_per_fold.append(learning_successes_array)
    training_mistakes_per_fold.append(training_mistakes_array)
    training_successes_per_fold.append(training_successes_array)
    training_accuracy_per_fold.append(training_accuracy_array)
    testing_mistakes_per_fold.append(testing_mistakes_array)
    testing_successes_per_fold.append(testing_successes_array)
    testing_accuracy_per_fold.append(testing_accuracy_array)


print average_per_fold(learning_mistakes_per_fold)
print average_per_fold(training_accuracy_per_fold)
print average_per_fold(testing_accuracy_per_fold)
plt.plot(average_per_fold(learning_mistakes_per_fold))
plt.ylabel('Mistakes')
plt.xlabel('Iterations')
plt.title('Learning Curve for Passive Aggressive Algorithm')
plt.show()
plt.plot(average_per_fold(training_accuracy_per_fold), label = "Training Accuracy")
plt.plot(average_per_fold(testing_accuracy_per_fold), label = "Testing Accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title('Accuracy Curve for Passive Aggressive algorithm')
plt.legend(loc = 0)
plt.show()
