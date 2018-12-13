import tensorflow as tf
#Create a constant that contains a string.
hello = tf.constant('Hello, TensorFlow!')
#Create a TensorFlow session.
sess = tf.Session()
#You can ignore the warnings that the TensorFlow library wasn't compiled to use certain instructions.
#Display the value of hello.
print(sess.run(hello))
