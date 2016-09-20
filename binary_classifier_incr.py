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
    
def perceptron_train(train_vector, y_hat):
    global learning_mistakes, learning_successes
    prediction = dot_product(weight_vector, train_vector)
    if (prediction * y_hat) <= 0:
        learning_mistakes += 1
        for i in xrange(VECTOR_SIZE):
            weight_vector[i] += y_hat * train_vector[i]
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
        
testing_mistakes_per_increment = []    
testing_successes_per_increment = []   
testing_accuracy_per_increment = []    

for num_examples in range(1000, 7000, 1000):
 
    for fold in range(len(training_files)):
        weight_vector = [0] * VECTOR_SIZE
        weight_vector_array = []
        learning_mistakes_array = []
        learning_successes_array = []
        testing_mistakes = 0
        testing_successes = 0

        for x in range(TRAINING_ITERATIONS):
            learning_mistakes = 0
            learning_successes = 0

            with open(training_files[fold], 'r') as f:
                count = 0
                for line in f:
                    if len(line) > 4:
                        count += 1
                        pixel_vector, y_hat = parse_file_line(line)
                        perceptron_train(pixel_vector, y_hat)
                        if count >= num_examples :
                            break;

            weight_vector_array.append(weight_vector[:])
            learning_mistakes_array.append(learning_mistakes)
            learning_successes_array.append(learning_successes)

        with open(test_files[fold], 'r') as f:
            for line in f:
                if len(line) > 4:
                    pixel_vector, y_hat = parse_file_line(line)
                    test(pixel_vector, y_hat)

        testing_mistakes_per_fold.append(testing_mistakes)
        testing_successes_per_fold.append(testing_successes)
        testing_accuracy_per_fold.append((testing_successes * 100) / float(testing_mistakes + testing_successes))

    testing_mistakes_per_increment.append(float(sum(testing_mistakes_per_fold)/len(testing_mistakes_per_fold)))
    testing_successes_per_increment.append(float(sum(testing_successes_per_fold)/len(testing_successes_per_fold)))
    testing_accuracy_per_increment.append(float(sum(testing_accuracy_per_fold))/len(testing_accuracy_per_fold))

print testing_accuracy_per_increment
plt.plot(range(1000, 7000, 1000), testing_accuracy_per_increment)
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title('Accuracy per increment for perceptron')
plt.show()
