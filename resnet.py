import tensorflow as tf
import tensorflow.keras.layers as layers

class Resnet(tf.keras.Model):
    '''Resnet model.\n
    Args:
        CLASSES: number of classes.
        conv2_x: conv2_x layer list.
        conv3_x: conv3_x layer list.
        conv4_x: conv4_x layer list.
        conv5_x: conv5_x layer list.
        (Note: conv2_x, conv3_x, conv4_x, conv5_x can be obtained via function net_cfg.)
        initializer: kernel initializers (default: HeUniform).
    
    Return:
        The resnet model
    '''

    def __init__(self, CLASSES, conv2_x, conv3_x, conv4_x, conv5_x, initializer):
        super(Resnet, self).__init__()
        self.head_conv = layers.Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), padding = "same",\
                                  kernel_initializer = initializer, name = "stage1_conv")
        self.head_pool = layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), name = "stage2_pool")
        self.conv2_x = tf.keras.Sequential(conv2_x, name = "stage2")
        self.conv3_x = tf.keras.Sequential(conv3_x, name = "stage3")
        self.conv4_x = tf.keras.Sequential(conv4_x, name = "stage4")
        self.conv5_x = tf.keras.Sequential(conv5_x, name = "stage5")
        self.avg_pool = layers.GlobalAveragePooling2D(name="avg_pool")
        self.dense = layers.Dense(CLASSES, kernel_initializer = initializer, name = "dense")

    def call(self, x, training = False):
        x = self.head_conv(x)
        x = self.head_pool(x)
        x = self.conv2_x(x, training)
        x = self.conv3_x(x, training)
        x = self.conv4_x(x, training)
        x = self.conv5_x(x, training)
        x = self.avg_pool(x)
        x = self.dense(x)
        return x
        
class IdentityBlock(tf.keras.Model):
    '''Identity Block of renet.\n
    Args:
        filters: number of filters for first layer.
        name: name for IdentityBlock.
        initializer: kernel_initializer (default: HeUniform).

    Return:
        The identity block.

    '''
    def __init__(self, filters, name, initializer):
        super(IdentityBlock, self).__init__()
        self.conv_1 = layers.Conv2D(filters = filters, kernel_size = (1, 1), strides = (1, 1),\
                                    padding = "same", kernel_initializer = initializer, name = name + "_conv_1")
        self.BN_1 = layers.BatchNormalization(name = name + "_BN_1")
        self.conv_2 = layers.Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1),\
                                    padding = "same", kernel_initializer = initializer, name = name + "_conv_2")
        self.BN_2 = layers.BatchNormalization(name = name + "_BN_2")
        self.conv_3 = layers.Conv2D(filters = filters * 4, kernel_size = (1, 1), strides = (1, 1),\
                                    padding = "same", kernel_initializer = initializer, name = name + "_conv_3")
        self.BN_3 = layers.BatchNormalization(name = name + "_BN_3")

    def call(self, inputs, training = False):
        x = self.conv_1(inputs)
        x = self.BN_1(x, training = training)
        x = tf.keras.activations.relu(x)
        x = self.conv_2(x)
        x = self.BN_2(x, training = training)
        x = tf.keras.activations.relu(x)
        x = self.conv_3(x)
        x = self.BN_3(x, training = training)
        x += inputs
        return tf.keras.activations.relu(x)

class ConvBlock(tf.keras.Model):
    '''Conv Block of renet.\n
    Args:
        filters: number of filters for first layer.
        stride: stride for first conv layer and short cut.
        name: name for convBlock.
        initializer: kernel_initializer (default: HeUniform).

    Return:
        The conv block.

    '''
    def __init__(self, filters, stride, name, initializer):
        super(ConvBlock, self).__init__()
        self.conv_1 = layers.Conv2D(filters = filters, kernel_size = (1, 1), strides = (stride, stride),\
                                    padding = "same", kernel_initializer = initializer, name = name + "_conv_1")
        self.BN_1 = layers.BatchNormalization(name = name + "_BN_1")
        self.conv_2 = layers.Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1),\
                                    padding = "same", kernel_initializer = initializer, name = name + "_conv_2")
        self.BN_2 = layers.BatchNormalization(name = name + "_BN_2")
        self.conv_3 = layers.Conv2D(filters = filters * 4, kernel_size = (1, 1), strides = (1, 1),\
                                    padding = "same", kernel_initializer = initializer, name = name + "_conv_3")
        self.BN_3 = layers.BatchNormalization(name = name + "_BN_3")
        self.short_cut = layers.Conv2D(filters = filters * 4, kernel_size = (1, 1), strides = (stride, stride),\
                                    padding = "same", kernel_initializer = initializer, name = name + "_conv_short")
        self.BN_short = layers.BatchNormalization(name = name + "_BN_short")
    
    def call(self, inputs, training = False):
        x = self.conv_1(inputs)
        x = self.BN_1(x, training = training)
        x = tf.keras.activations.relu(x)
        x = self.conv_2(x)
        x = self.BN_2(x, training = training)
        x = tf.keras.activations.relu(x)
        x = self.conv_3(x)
        x = self.BN_3(x, training = training)
        short = self.short_cut(inputs)
        short = self.BN_short(short, training = training)
        x += short
        return tf.keras.activations.relu(x)

class SmallIdentityBlock(tf.keras.Model):
    '''Identity Block of renet18, 34.\n
    Args:
        filters: number of filters for first layer.
        name: name for IdentityBlock.
        initializer: kernel_initializer (default: HeUniform).

    Return:
        The identity block.

    '''
    def __init__(self, filters, name, initializer):
        super(SmallIdentityBlock, self).__init__()
        self.conv_1 = layers.Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1),\
                                    padding = "same", kernel_initializer = initializer, name = name + "_conv_1")
        self.BN_1 = layers.BatchNormalization(name = name + "_BN_1")
        self.conv_2 = layers.Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1),\
                                    padding = "same", kernel_initializer = initializer, name = name + "_conv_2")
        self.BN_2 = layers.BatchNormalization(name = name + "_BN_2")

    def call(self, inputs, training = False):
        x = self.conv_1(inputs)
        x = self.BN_1(x, training = training)
        x = tf.keras.activations.relu(x)
        x = self.conv_2(x)
        x = self.BN_2(x, training = training)
        x += inputs
        return tf.keras.activations.relu(x)

class SmallConvBlock(tf.keras.Model):
    '''Conv Block of renet18, 34.\n
    Args:
        filters: number of filters for first layer.
        stride: stride for first conv layer and short cut.
        name: name for convBlock.
        initializer: kernel_initializer (default: HeUniform).

    Return:
        The conv block.

    '''
    def __init__(self, filters, stride, name, initializer):
        super(SmallConvBlock, self).__init__()
        self.conv_1 = layers.Conv2D(filters = filters, kernel_size = (3, 3), strides = (stride, stride),\
                                    padding = "same", kernel_initializer = initializer, name = name + "_conv_1")
        self.BN_1 = layers.BatchNormalization(name = name + "_BN_1")
        self.conv_2 = layers.Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1),\
                                    padding = "same", kernel_initializer = initializer, name = name + "_conv_2")
        self.BN_2 = layers.BatchNormalization(name = name + "_BN_2")
        self.short_cut = layers.Conv2D(filters = filters, kernel_size = (3, 3), strides = (stride, stride),\
                                    padding = "same", kernel_initializer = initializer, name = name + "_conv_short")
        self.BN_short = layers.BatchNormalization(name = name + "_BN_short")
    
    def call(self, inputs, training = False):
        x = self.conv_1(inputs)
        x = self.BN_1(x, training = training)
        x = tf.keras.activations.relu(x)
        x = self.conv_2(x)
        x = self.BN_2(x, training = training)
        short = self.short_cut(inputs)
        short = self.BN_short(short, training = training)
        x += short
        return tf.keras.activations.relu(x)
