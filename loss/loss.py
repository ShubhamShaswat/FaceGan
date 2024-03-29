import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

tf.compat.v1.disable_eager_execution() #for tf.gradients 

####### Tenforflow loss function for WGAN #######
#Loss = tf.keras.losses.Loss(reduction=losses_utils.ReductionV2.AUTO, name=None)

#Critic Loss
class CriticLoss(object):
    """ Criric Loss """
    def __init__(self, gp_lambda=10):
        self.gp_lambda = gp_lambda

    @tf.function
    def __call__(self,discriminator, Dx, Dx_hat,x_interpolated):
        #orgnal critic loss
        d_loss = tf.reduce_mean(Dx_hat) - tf.reduce_mean(Dx)
        #calculte gradinet penalty
        gradients = tf.gradients(discriminator(x_interpolated, training=True), [x_interpolated,])[0]
        grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        grad_penalty = tf.reduce_mean(tf.square(grad_l2 - 1.0))
        #final discriminator loss
        d_loss += self.gp_lambda * grad_penalty
        return d_loss

#Generator loss, Test Cases Passed
class GeneratorLoss(object):
    """ Generator Loss """

    def __call__(self,Dx_hat):
        return tf.reduce_mean(-Dx_hat)
        
 
 
 

##Test Loss---------------------------------------------------------------------------------------
gen_loss = GeneratorLoss()
critic_loss = CriticLoss()

# UNIT TEST
tf.debugging.assert_near(
    gen_loss(tf.constant(1.)),
    tf.constant(-1.0)
)
print("Success!")



def gradient_penalty(gradients):
	#gradients = tf.gradients(y, [x_interpolated,])[0]
	grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
	grad_penalty = tf.reduce_mean(tf.square(grad_l2 - 1.0))
	return grad_penalty 




x=tf.zeros([16,1,28,28])
y = tf.random.normal((16,1))

gradients = tf.gradients(y, [x])[0]
print(gradients)
# UNIT TEST
tf.debugging.assert_near(
	gradient_penalty(x),
    	tf.constant(1.0)
)
print("Success!")


x=tf.ones([16,1,28,28]) / tf.sqrt(784.)
tf.debugging.assert_near(
	gradient_penalty(x),
    	tf.constant(0.)
)
print("Success!")










