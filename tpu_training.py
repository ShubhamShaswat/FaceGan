import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Flatten, Dropout, LayerNormalization
from tensorflow.keras.layers import Dense, LeakyReLU, Conv2D, Reshape

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)

#get GSC path
from kaggle_datasets import KaggleDatasets
GCS_DS_PATH = KaggleDatasets().get_gcs_path()

'''
from google.cloud import storage

STORAGE_CLIENT = storage.Client(project='burnished-edge-278511')
'''

#GCS_DS_PATH

#!gsutil ls $GCS_DS_PATH

# detect and init the TPU


tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    # instantiate a distribution strategy
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print("Number of accelerators: ", tpu_strategy.num_replicas_in_sync)

IMAGE_SIZE = [64,64]


# Create a dictionary describing the features.
image_feature_description = {

    'image': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  parsed_example = tf.io.parse_single_example(example_proto, image_feature_description)
  image = decode_image(parsed_example['image'])

  return image

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.image.resize(image,[*IMAGE_SIZE])
    return image

def view_image(ds):
    image = next(iter(ds)) # extract 1 batch from the dataset
    image = image.numpy()

    fig = plt.figure(figsize=(20, 20))
    for i in range(20):
        ax = fig.add_subplot(4, 5, i+1, xticks=[], yticks=[])
        ax.imshow(image[i])

if not tpu:
    AUTO = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 128

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    # On Kaggle you can also use KaggleDatasets().get_gcs_path() to obtain the GCS path of a Kaggle dataset
    #filenames = tf.io.gfile.glob("gs://celeba_bucket/*.tfrecord")
    data_dir = GCS_DS_PATH + '/*.tfrecord'
    filenames = tf.io.gfile.glob(data_dir)
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)


    dataset = dataset.map(_parse_image_function, num_parallel_calls=AUTO)
    #train_dataset = train_dataset.map(process_data, num_parallel_calls=AUTO)
    dataset = dataset.repeat().shuffle(1024).batch(BATCH_SIZE)
    #dataset = dataset.map(...) # TFRecord decoding here...
    
    #show images 
    view_image(dataset)

"""
#create a discriminator model with LayerNormalization
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dense, LeakyReLU, Conv2D, Reshape, Flatten, Dropout, LayerNormalization


class make_disc_block(tf.keras.Model):
    def __init__(self,filters,kernel_size,strides):
        super(make_disc_block,self).__init__()
        
        self.conv2d = Conv2D(filters,
                             kernel_size,
                             strides,
                             padding='same')
        
        #self.layer_norm = LayerNormalization(axis=[1,2,3])
        
        self.leaky_relu = LeakyReLU(0.2)
        
    def call(self,x):
        
        x = self.conv2d(x)
        #x = self.layer_norm(x)
        x = self.leaky_relu(x)
        return x
    

class Discriminator(tf.keras.Model):

  def __init__(self,nodes=256*4*4):
    super(Discriminator,self).__init__()
    
    self.disc_block_1 = make_disc_block(filters=64,kernel_size=(5,5),strides=(2,2))
    self.disc_block_2 = make_disc_block(filters=128,kernel_size=(5,5),strides=(2,2))
    self.disc_block_3 = make_disc_block(filters=256,kernel_size=(5,5),strides=(2,2))
    self.disc_block_4 = make_disc_block(filters=512,kernel_size=(5,5),strides=(2,2))
    self.disc_block_5 = make_disc_block(filters=1024,kernel_size=(5,5),strides=(2,2))
   
    self.dense1 = Dense(1)
    self.flatten = Flatten()
    self.dropout = Dropout(.4)
   
  def call(self,x):

    x = self.disc_block_1(x)
    x = self.disc_block_2(x)
    x = self.disc_block_3(x)
    x = self.disc_block_4(x)
    x = self.disc_block_5(x)

    x = self.flatten(x)
    x = self.dropout(x)
    x = self.dense1(x)

    return x

#create a discriminator model with BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dense, LeakyReLU, Conv2D, Reshape, Flatten, Dropout, LayerNormalization

"""
#------------------------------MODELS--------------------------------------------------------------
#define Discriminator
class make_disc_block(tf.keras.Model):
    def __init__(self,filters,kernel_size,strides):
        super(make_disc_block,self).__init__()
        
        self.conv2d = Conv2D(filters,
                             kernel_size,
                             strides,
                             padding='same')
        
        self.batch_norm = BatchNormalization()
        
        self.leaky_relu = LeakyReLU(0.2)
        
    def call(self,x):
        
        x = self.conv2d(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return x

class Discriminator(tf.keras.Model):
  def __init__(self,nodes=256*4*4):
    super(Discriminator,self).__init__()
    
    self.disc_block_1 = make_disc_block(filters=64,kernel_size=(5,5),strides=(2,2))
    self.disc_block_2 = make_disc_block(filters=128,kernel_size=(5,5),strides=(2,2))
    self.disc_block_3 = make_disc_block(filters=256,kernel_size=(5,5),strides=(2,2))
    self.disc_block_4 = make_disc_block(filters=512,kernel_size=(5,5),strides=(2,2))
    self.disc_block_5 = make_disc_block(filters=1024,kernel_size=(5,5),strides=(2,2))
   
    self.dense1 = Dense(1)
    self.flatten = Flatten()
    self.dropout = Dropout(.4)
   
  def call(self,x):

    x = self.disc_block_1(x)
    x = self.disc_block_2(x)
    x = self.disc_block_3(x)
    x = self.disc_block_4(x)
    x = self.disc_block_5(x)

    x = self.flatten(x)
    x = self.dropout(x)
    x = self.dense1(x)

    return x

#define Generator
class make_generator_block(tf.keras.Model):
    
  def __init__(self,filters,kernel_size=(5,5),strides=(2,2)):
    super(make_generator_block,self).__init__()

    self.conv2d_transpose = Conv2DTranspose(filters,
                                            kernel_size,
                                            strides,
                                            padding='same',
                                            )
    self.batch_norm = BatchNormalization()
    self.leaky_relu = LeakyReLU(0.2)
        
  def call(self,x):
    x = self.conv2d_transpose(x)
    x = self.batch_norm(x)
    x = self.leaky_relu(x)
    return x

class Generator(tf.keras.Model):
  def __init__(self,nodes=512*4*4):
    super(Generator,self).__init__()

    self.dense1 = Dense(nodes)
    self.leaky_relu = LeakyReLU(0.2)
    self.gen_block_1 = make_generator_block(filters=512)
    self.gen_block_2 = make_generator_block(filters=256)
    self.gen_block_3 = make_generator_block(filters=128)
    self.gen_block_4 = make_generator_block(filters=64)
    self.conv2d_transpose_final = Conv2DTranspose(3,(5,5),strides=(1,1),activation='tanh',padding='same') #experiment with tanh
    self.reshape = Reshape((4,4,512))
  
  @tf.function
  def call(self,z):

    #input layer
    z = self.dense1(z)
    z = self.reshape(z)
    z = self.leaky_relu(z)
    
    z = self.gen_block_1(z)
    z = self.gen_block_2(z)
    z = self.gen_block_3(z)
    z = self.gen_block_4(z)

    #final layer
    out = self.conv2d_transpose_final(z)
    return out


#------------------------------------------------------------DATASET---------------------------------
AUTO = tf.data.experimental.AUTOTUNE
#BATCH_SIZE = 4096
def get_dataset(batch_size):
  

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    # On Kaggle you can also use KaggleDatasets().get_gcs_path() to obtain the GCS path of a Kaggle dataset
    #filenames = tf.io.gfile.glob('gs://celeba_bucket/*.tfrecord')
    filenames = tf.io.gfile.glob(GCS_DS_PATH + '/celeba.tfrecord')
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    dataset = dataset.map(_parse_image_function, num_parallel_calls=AUTO)
    dataset = dataset.repeat().shuffle(200000).batch(batch_size,drop_remainder=True).prefetch(AUTO)
    
    return dataset

checkpoint_dir = 'gs://celeba_bucket/training_checkpoints_april_2'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

#---------------------------------------------LOSS-------------------------------------------------------
#define custom loss function

#let's write a Wasserstein loss

"""
def critic_loss(real_output,fake_output):
    loss = real_output - fake_output
    return tf.nn.compute_average_loss(loss, global_batch_size=BATCH_SIZE)
    

def gen_loss(fake_output):
    loss = -1 * fake_output
    return tf.nn.compute_average_loss(loss, global_batch_size=BATCH_SIZE)

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
"""


#Critic Loss
class CriticLoss(object):
    """ Criric Loss """
    def __init__(self, gp_lambda=10):
        self.gp_lambda = gp_lambda

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


#----------------------------------TRAINING STEP-------------------------------------------------


# instantiating the model in the strategy scope creates the model on the TPU
######FOR WGAN #######

EPOCHS = 100
BATCH_SIZE = 4096
STEPS_PER_TPU_CALL = 202599 // BATCH_SIZE
STEPS_PER_EPOCH = 202599 // BATCH_SIZE
penality_coeff = 10 #from paper

with tpu_strategy.scope():
    
    #define model
    generator = Generator()
    discriminator = Discriminator()
    
    #define optimizers
    g_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5) #0.0001
    d_opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5) #0.0001
    
    
    #define loss 
    gen_loss = tf.keras.metrics.Mean(name='gen_loss')
    disc_loss = tf.keras.metrics.Mean(name='disc_loss')
    
     
    critic_loss = CriticLoss()
    generator_loss = GeneratorLoss()
    
    """
    #let's write a Wasserstein loss
    def critic_loss(real_output,fake_output,interpolated):
        
        gradients = tf.gradients(ys=discriminator(interpolated,training=True), xs=[interpolated,])[0]
        grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients),axis=[1,2,3]))
        gradient_penality = tf.reduce_mean(tf.square(grad_l2 - 1.0))
        loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        loss += penality_coeff * gradient_penality
    
        return loss / BATCH_SIZE


    def generator_loss(fake_output):
        loss = -1 * fake_output
        #loss = tf.reduce_mean(loss)
        return tf.nn.compute_average_loss(loss, global_batch_size=BATCH_SIZE)
    """
    #define checkpoints
    checkpoint = tf.train.Checkpoint(generator_optimizer=g_opt,
                                 discriminator_optimizer=d_opt,
                                 generator=generator,
                                 discriminator=discriminator)
    

per_replica_batch_size =  BATCH_SIZE // tpu_strategy.num_replicas_in_sync

train_dataset = tpu_strategy.experimental_distribute_datasets_from_function(lambda _:get_dataset(per_replica_batch_size))

#define a train step

@tf.function
def train_step(iterator):
    
    def step_fn(x):
        """The computation to run on each TPU device."""
        z = tf.random.normal([per_replica_batch_size,100])
        epsilon = tf.random.uniform([per_replica_batch_size,1,1,1]) 
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            Dx = discriminator(x,training=True)
            Gz = generator(z,training=True)
            
            #interpolated 
            interpolated = epsilon * x + (1 - epsilon) * Gz
            
            DGz = discriminator(Gz,training=True)
            
            #get generator loss
            g_loss = generator_loss(DGz) / BATCH_SIZE # In WGAN replace this gen_loss

            #get discriminator loss
            d_loss = critic_loss(discriminator,Dx,DGz,interpolated) / BATCH_SIZE  #In WGAN replace this with ciritic_loss

        gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)

        g_opt.apply_gradients(zip(gradients_of_generator,generator.trainable_variables))
        d_opt.apply_gradients(zip(gradients_of_discriminator,discriminator.trainable_variables))
        
        gen_loss.update_state(g_loss * tpu_strategy.num_replicas_in_sync) # * tpu_strategy.num_replicas_in_sync 
        disc_loss.update_state(d_loss * tpu_strategy.num_replicas_in_sync )
        
   
    tpu_strategy.run(step_fn, args = (next(iterator),))

"""
ckpt_path = 'gs://celeba_bucket/training_checkpoints_april_2'
manager = tf.train.CheckpointManager(checkpoint, ckpt_path, max_to_keep=3)
checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")
"""

#-------------------------------------------------TRAINING-----------------------------------
#define train
import IPython.display as display
import time

gen_loss_hist = []
disc_loss_hist = []

steps_per_epoch = 202599 // BATCH_SIZE
train_iterator = iter(train_dataset)

def train(dataset):
 
  for epoch in range(EPOCHS):

    #start time 
    start = time.time()
    
    for step in range(steps_per_epoch):

      train_step(train_iterator)
        
      print('Epoch : {} Batch : {} G Loss: {} D Loss: {}'.format(epoch,g_opt.iterations.numpy(),gen_loss.result(),disc_loss.result()))
      display.clear_output(wait=True)
        
     #append losses
      gen_loss_hist.append(gen_loss.result())
      disc_loss_hist.append(disc_loss.result())
    
    print('Time : {}'.format(time.time()-start))
    
    # Save the model every 15 epochs
   
    if (epoch + 1) % 100 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    
    ##append losses
    #gen_loss_hist.append(gen_loss.result())
    #disc_loss_hist.append(disc_loss.result())
    
    #reset loss states
    gen_loss.reset_states()
    disc_loss.reset_states()
#---------------------------------------------------------------------------------------------------------------
train(train_dataset)

plt.plot(gen_loss_hist,label="Gen")
plt.plot(disc_loss_hist,label="Disc")
plt.legend(loc='upper right')
plt.show()
"""
tf.saved_model.save(
    generator, 'gs://celeba_bucket/saved_model_april_2/my_model',
    signatures=generator.call.get_concrete_function(
        tf.TensorSpec(shape=[None, 100], dtype=tf.float32, name="inp")))

sda

n=16
seed = tf.random.normal([n,100])
generated_samples = generator(seed)
fig = plt.figure(figsize=(10,10))

for i in range(n):
    
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_samples[i])

plt.show()

n=4

fname = ['gs://celeba_bucket/training_checkpoints_april/ckpt-1',
         'gs://celeba_bucket/training_checkpoints_april/ckpt-2',
         'gs://celeba_bucket/training_checkpoints_april/ckpt-3',
         'gs://celeba_bucket/training_checkpoints_april/ckpt-4',
         'gs://celeba_bucket/training_checkpoints_april/ckpt-5',
         'gs://celeba_bucket/training_checkpoints_april/ckpt-6',
         'gs://celeba_bucket/training_checkpoints_april/ckpt-7',
         'gs://celeba_bucket/training_checkpoints_april/ckpt-8',
         'gs://celeba_bucket/training_checkpoints_april/ckpt-9',
        ]

for c in range(len(fname)):
    
    checkpoint.restore(fname[c])
    seed = tf.random.normal([n,100])
    generated_samples = generator(seed)
    fig = plt.figure(figsize=(10,10))

    for i in range(n):
    
        plt.subplot(1, 4, i+1)
        plt.imshow(generated_samples[i])

    plt.show()

#custom training loop

import IPython.display as display
import time


train_iterator = iter(train_dataset)

gen_loss_hist = []
disc_loss_hist = []
step = 0
epoch_steps = 0
epoch = 0

while True:
    
    # run training step
    train_step(train_iterator)
    epoch_steps += STEPS_PER_TPU_CALL
    step += STEPS_PER_TPU_CALL

    
    print('Epoch : {} steps : {} G Loss: {} D Loss: {}'.format(epoch+1,epoch_steps,gen_loss.result(),disc_loss.result()))
    display.clear_output(wait=True)
    
    if (step // STEPS_PER_EPOCH) > epoch:
        
        gen_loss_hist.append(gen_loss.result())
        disc_loss_hist.append(disc_loss.result())
    
        # set up next epoch
        epoch = step // STEPS_PER_EPOCH
        epoch_steps = 0
        #reset loss states
        gen_loss.reset_states()
        disc_loss.reset_states()
        
    
    if epoch >= EPOCHS:
        break

train(train_dataset)

!ls

!zip -r filename.zip saved_model

from IPython.display import FileLink
FileLink(r'filename.zip')
"""