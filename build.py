import tensorflow as tf
import tensorflow.keras.layers as layers
from resnet import *

def build_net(cfg, CLASSES, initializer = tf.keras.initializers.he_uniform()):
    '''Build resnet.\n
    Args:
        cfg: string, type of net (options: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152').
        CLASSES: number of classes.
        initializer: kernel initializers (default: HeUniform).

    Return:
        resnet model you selected.
    '''
    conv2_x, conv3_x, conv4_x, conv5_x = net_cfg(cfg, initializer)
    net = Resnet(CLASSES, conv2_x, conv3_x, conv4_x, conv5_x, initializer)
    net._name = cfg
    return net


def net_cfg(cfg, initializer):
    '''Select type of resnet.\n
    Arg:
        cfg: type of net (options: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152').

    Return:
        conv2_x, conv3_x, conv4_x, conv5_x (layer list for each stage).
    '''
    conv2_x = []
    conv3_x = []
    conv4_x = []
    conv5_x = []
    small = False
    if cfg == 'resnet18':
        layer_list = [2, 2, 2, 2]
        small = True
    elif cfg == 'resnet34':
        layer_list = [3, 4, 6, 3]
        small = True
    elif cfg == 'resnet50':
        layer_list = [3, 4, 6, 3]
    elif cfg == 'resnet101':
        layer_list = [3, 4, 23, 3]
    elif cfg == 'resnet152':
        layer_list = [3, 8, 36, 3]
    else:
        raise Exception("No such option!")
    for i in range(layer_list[0]):
        if i == 0:
            if small:
                conv2_x.append(SmallConvBlock(64, 1, "stage2_{}".format(i + 1), initializer))
            else:
                conv2_x.append(ConvBlock(64, 1, "stage2_{}".format(i + 1), initializer))
        else:
            if small:
                conv2_x.append(SmallIdentityBlock(64, "stage2_{}".format(i + 1), initializer))
            else:
                conv2_x.append(IdentityBlock(64, "stage2_{}".format(i + 1), initializer))
    #conv3_x
    for i in range(layer_list[1]):
        if i == 0:
            if small:
                conv3_x.append(SmallConvBlock(128, 2, "stage3_{}".format(i + 1), initializer))
            else:
                conv3_x.append(ConvBlock(128, 2, "stage3_{}".format(i + 1), initializer))
        else:
            if small:
                conv3_x.append(SmallIdentityBlock(128, "stage3_{}".format(i + 1), initializer))
            else:
                conv3_x.append(IdentityBlock(128, "stage3_{}".format(i + 1), initializer))
    #conv4_x
    for i in range(layer_list[2]):
        if i == 0:
            if small:
                conv4_x.append(SmallConvBlock(256, 2, "stage4_{}".format(i + 1), initializer))
            else:
                conv4_x.append(ConvBlock(256, 2, "stage4_{}".format(i + 1), initializer))
        else:
            if small:
                conv4_x.append(SmallIdentityBlock(256, "stage4_{}".format(i + 1), initializer))
            else:
                conv4_x.append(IdentityBlock(256, "stage4_{}".format(i + 1), initializer))
    #conv5_x
    for i in range(layer_list[3]):
        if i == 0:
            if small:
                conv5_x.append(SmallConvBlock(512, 2, "stage5_{}".format(i + 1), initializer))
            else:
                conv5_x.append(ConvBlock(512, 2, "stage5_{}".format(i + 1), initializer))
        else:
            if small:
                conv5_x.append(SmallIdentityBlock(512, "stage5_{}".format(i + 1), initializer))
            else:
                conv5_x.append(IdentityBlock(512, "stage5_{}".format(i + 1), initializer))

    return conv2_x, conv3_x, conv4_x, conv5_x
