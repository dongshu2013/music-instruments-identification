# -*- coding: utf-8 -*-

import numpy as np

def load_train():
    data = np.load('../feature/nmfcc/training/Piano.npz')
    feature1 = np.array(data['arr_0'])
    label1 = 3 * np.ones((feature1.shape[0], 1))
    feature = np.concatenate((label1, feature1), axis = 1)

    data = np.load('../feature/nmfcc/training/Flute.npz')
    feature2 = np.array(data['arr_0'])
    label2 = np.ones((feature2.shape[0], 1))
    tfeature = np.concatenate((label2, feature2), axis = 1)
    feature = np.concatenate((feature, tfeature))

    data = np.load('../feature/nmfcc/training/Guitar.npz')
    feature3 = np.array(data['arr_0'])
    label3 = 4 * np.ones((feature3.shape[0], 1))
    tfeature = np.concatenate((label3, feature3), axis = 1)
    feature = np.concatenate((feature, tfeature))

    data = np.load('../feature/nmfcc/training/Violin.npz')
    feature4 = np.array(data['arr_0'])
    label4 = 2 * np.ones((feature4.shape[0], 1))
    tfeature = np.concatenate((label4, feature4), axis = 1)
    feature = np.concatenate((feature, tfeature))

    data = np.load('../feature/nmfcc/training/AltoSax.npz')
    feature5 = np.array(data['arr_0'])
    label5 = np.zeros((feature5.shape[0], 1))
    tfeature = np.concatenate((label5, feature5), axis = 1)
    feature = np.concatenate((feature, tfeature))

    return  feature

def load_test():
    data = np.load('../feature/nmfcc/testing/Piano_test.npz')
    feature1 = np.array(data['arr_0'])
    label1 = 3 * np.ones((feature1.shape[0], 1))
    feature = np.concatenate((label1, feature1), axis = 1)

    data = np.load('../feature/nmfcc/testing/Flute_test.npz')
    feature2 = np.array(data['arr_0'])
    label2 = np.ones((feature2.shape[0], 1))
    tfeature = np.concatenate((label2, feature2), axis = 1)
    feature = np.concatenate((feature, tfeature))

    data = np.load('../feature/nmfcc/testing/Guitar_test.npz')
    feature3 = np.array(data['arr_0'])
    label3 = 4 * np.ones((feature3.shape[0], 1))
    tfeature = np.concatenate((label3, feature3), axis = 1)
    feature = np.concatenate((feature, tfeature))

    data = np.load('../feature/nmfcc/testing/Violin_test.npz')
    feature4 = np.array(data['arr_0'])
    label4 = 2 * np.ones((feature4.shape[0], 1))
    tfeature = np.concatenate((label4, feature4), axis = 1)
    feature = np.concatenate((feature, tfeature))

    data = np.load('../feature/nmfcc/testing/AltoSax_test.npz')
    feature5 = np.array(data['arr_0'])
    label5 = np.zeros((feature5.shape[0], 1))
    tfeature = np.concatenate((label5, feature5), axis = 1)
    feature = np.concatenate((feature, tfeature))
    return  feature

