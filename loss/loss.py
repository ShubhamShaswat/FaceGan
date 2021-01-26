import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

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

#Generator loss
class GeneratorLoss(object):
    """ Generator Loss """

    def __call__(self,Dx_hat):
        return tf.reduce_mean(-Dx_hat)
