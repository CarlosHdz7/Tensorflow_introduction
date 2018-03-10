import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 

#Data generation
observations = 1000
xs = np.random.uniform(low=-10,high=10,size=(observations,1))
zs = np.random.uniform(-10,10,(observations,1))

generated_inputs = np.column_stack((xs,zs))
noise = np.random.uniform(-1,1,(observations,1))
generated_targets = 2*xs - 3*zs + 5 + noise

np.savez('TZ_intro',inputs=generated_inputs,targets=generated_targets)

#Solving with tensorflow
#input size = 2 output size = 1
inputs_size = 2
output_size = 1

#outlining the model
#--------------------------------Data set
#We feed the inputs and targets
inputs = tf.placeholder(tf.float32,[None,inputs_size])
targets = tf.placeholder(tf.float32,[None,output_size])
#nothing has happened yet

weights = tf.Variable(tf.random_uniform([inputs_size,output_size],minval=-0.1,maxval=0.1))
biases = tf.Variable(tf.random_uniform([output_size],minval=-0.1,maxval = 0.1))

#outputs
#XW + B = Y
outputs = tf.matmul(inputs,weights) + biases
#At this point our algorithm do nothing :V

#loss function
#l2-norm loss / obhservations entre 2
#only define the objective function that we use
mean_loss = tf.losses.mean_squared_error(labels=targets,predictions=outputs) / 2.

#Gradient descent
optimize = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(mean_loss)

#nothing has been executed yet
#Prepare it for execution
#sess = session
#The training happens in these called sessions
sess = tf.InteractiveSession()

#It is time to train the algorithm

#initialization
initializer = tf.global_variables_initializer()
sess.run(initializer)

#Loading training data
training_data = np.load('TZ_intro.npz')


#learning
for e in range(100):
    _, curr_loss = sess.run([optimize,mean_loss],feed_dict = {inputs: training_data['inputs'], targets: training_data['targets']})
    print(curr_loss)

#plotting data
out = sess.run([outputs],feed_dict={inputs:training_data['inputs']})
plt.plot(np.squeeze(out), np.squeeze(training_data['targets']))
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()