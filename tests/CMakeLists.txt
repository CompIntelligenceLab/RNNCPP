
add_subdirectory(activations)
#add_subdirectory(copy_constructors)
#add_subdirectory(weights)
#add_subdirectory(input)

#add_subdirectory(test_recurrent_model1)

#add_subdirectory(test_recurrent_model2)

#add_subdirectory(test_recurrent_model3)

#add_subdirectory(test_recurrent_model4)
#add_subdirectory(test_recurrent_model5)
add_subdirectory(test_recurrent_model_bias1)
#add_subdirectory(test_recurrent_model_bias2)
#add_subdirectory(test_recurrent_model_bias5)

#add_subdirectory(softmax)
#add_subdirectory(softmax_rec1)
#add_subdirectory(debug_bias1)

# calculate differential equation from signal (works)
#add_subdirectory(test_diff_eq1)

# solve for differential equation parameters
#add_subdirectory(test_diff_eq2)

# solve for simplest equation dy/dy = -alpha y with simplest network (recurrent, seq=1)
add_subdirectory(test_diff_eq3)

# solve for simplest equation dy/dy = -alpha y (recurrent, seq=1)
# Experiment with multiple layers and layers with multiple nodes. 
add_subdirectory(test_diff_eq4)

# return to tanh recurrent node following the first dense node with forward Euler activation. 
add_subdirectory(test_diff_eq5)

