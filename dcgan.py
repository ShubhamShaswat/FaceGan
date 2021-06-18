from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dense, LeakyReLU, Conv2D, Reshape, Flatten, Dropout, LayerNormalization

########create a discriminator model with LayerNormalization#########
class make_disc_block(tf.keras.Model):
    """ Discriminator Block """
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
    """ Discriminator """

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

######create a Generator model#####
class make_generator_block(tf.keras.Model):
  """ Generator Block """

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

#define generator
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
