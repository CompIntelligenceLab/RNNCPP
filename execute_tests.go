#!/bin/bash

export H=../build/tests
#${H}/activations/activations 
#$H/copy_constructors/copy_constructors 
#$H/weights/weights 
#$H/feedforward/feed_forward
#$H/input/input
$H/test_recurrent_model1/test_recurrent_model1
$H/test_recurrent_model2/test_recurrent_model2
$H/test_recurrent_model3/test_recurrent_model3
$H/test_recurrent_model4/test_recurrent_model4 # same as test_recurrent_model2
$H/test_recurrent_model5/test_recurrent_model5
#$H/test_recurrent_model_bias1/test_recurrent_model_bias1 # same as model1
$H/test_recurrent_model_bias2/test_recurrent_model_bias2
$H/test_recurrent_model_bias5/test_recurrent_model_bias5
