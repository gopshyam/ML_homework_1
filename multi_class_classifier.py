#! /usr/bin/env python

import numpy as np

training_files = ["Data/ocr_fold0_sm_train.txt", "Data/ocr_fold1_sm_train.txt", "Data/ocr_fold2_sm_train.txt", "Data/ocr_fold3_sm_train.txt", "Data/ocr_fold4_sm_train.txt", "Data/ocr_fold5_sm_train.txt", "Data/ocr_fold6_sm_train.txt", "Data/ocr_fold7_sm_train.txt", "Data/ocr_fold8_sm_train.txt", "Data/ocr_fold9_sm_train.txt"]

test_files = ["Data/ocr_fold0_sm_test.txt", "Data/ocr_fold1_sm_test.txt", "Data/ocr_fold2_sm_test.txt", "Data/ocr_fold3_sm_test.txt", "Data/ocr_fold4_sm_test.txt", "Data/ocr_fold5_sm_test.txt", "Data/ocr_fold6_sm_test.txt", "Data/ocr_fold7_sm_test.txt", "Data/ocr_fold8_sm_test.txt", "Data/ocr_fold9_sm_test.txt"]

learning_mistakes = 0
learning_successes = 0
testing_mistakes = 0
testing_successes = 0

def convert_string_to_int_list(pixel_values):
    pixel_values = pixel_values[2:]
    pixel_vector = []
    for x in pixel_values:
        pixel_vector.append(int(x))
    return pixel_vector

def parse_line(line):
    if len(line) < 2:
        return None
    line_split = line.strip().split('\t')
    pixel_values = line_split[1]
    pixel_vector = convert_string_to_int_list(pixel_values)
    y_hat = ord(line_split[2]) - ord('a')
    return np.asarray(pixel_vector), y_hat

def f_x_y(x, y):
    #Returns the vector representation
    F = [0] * VECTOR_SIZE * NUM_CLASSES
    start_index = y * VECTOR_SIZE
    end_index = start_index + VECTOR_SIZE
    F[start_index : end_index] = x[:]
    return np.asarray(F) 

def arg_max(x_t, weight_vector):
    #Returns argmax w.F(x_t,y)
    class_label_list = range(NUM_CLASSES)
    return max(class_label_list, key = lambda y: np.dot(weight_vector, f_x_y(x_t, y)))

def best_bad(x_t, weight_vector):
    #Returns the second best value of w.F(x_t,y)
    class_label_list = range(NUM_CLASSES)
    best = arg_max(x_t)
    class_label_list.remove(best)
    return max(class_label_list, key = lambda y: np.dot(weight_vector, f_x_y(x_t, y)))

def perceptron_learn(weight_vector, x_t, y_t):
    #Updates weight vector based on the training vector x_t
    global learning_mistakes, learning_successes
    y_hat = arg_max(x_t, weight_vector)
    if y_t != y_hat:
        #mistake, update weight vector
        learning_mistakes += 1
        weight_vector = np.add(weight_vector, np.subtract(f_x_y(x_t, y_t), f_x_y(x_t, y_hat)))
    else:
        learning_successes += 1
    return weight_vector

def perceptron_test(weight_vector, x_t, y_t):
    #Tests and updates counters based on mistakes
    global testing_mistakes, testing_successes
    y_hat = arg_max(x_t, weight_vector)
    if y_t != y_hat:
        testing_mistakes += 1
    else:
        testing_successes += 1
        

TRAINING_ITERATIONS = 5
VECTOR_SIZE = 128
NUM_CLASSES = 26

for fold in range(1):
    weight_vector = np.asarray([0] * VECTOR_SIZE * NUM_CLASSES)
    weight_vector_list = []
    learning_mistakes_list = []
    testing_mistakes_list = []
    
    for iteration in range(TRAINING_ITERATIONS):

        testing_mistakes = 0
        testing_successes = 0
        learning_mistakes = 0
        trainins_successes = 0

        with open(training_files[fold], 'r') as f:
            for line in f:
                if len(line) > 4:
                    x_t, y_t = parse_line(line)
                    weight_vector = perceptron_learn(weight_vector, x_t, y_t)

        weight_vector_list.append(weight_vector)
        learning_mistakes_list.append(learning_mistakes)
       

        with open(test_files[fold], 'r') as f:
            for line in f:
                if len(line) > 4:
                    x_t, y_t = parse_line(line)
                    perceptron_test(weight_vector, x_t, y_t)

        testing_mistakes_list.append(testing_mistakes)

print learning_mistakes_list, testing_mistakes_list
