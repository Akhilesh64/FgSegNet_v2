import keras
from keras import layers
from keras.models import Model
from keras.layers import Input, Dropout, Activation, SpatialDropout2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, Cropping2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import concatenate, add, multiply
from my_upsampling_2d import MyUpSampling2D
from instance_normalization import InstanceNormalization
import keras.backend as K
import tensorflow as tf


def loss(y_true, y_pred):
    void_label = -1.
    y_pred = K.reshape(y_pred, [-1])
    y_true = K.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def acc(y_true, y_pred):
    void_label = -1.
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def loss2(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def acc2(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

class FgSegNet_v2_module(object):

    def __init__(self, lr, img_shape, scene, resnet50_weights_path):
        self.lr = lr
        self.img_shape = img_shape
        self.scene = scene
        self.resnet50_weights_path = resnet50_weights_path
        self.method_name = 'FgSegNet_v2'

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):                 #Identity Block for ResNet architecture

        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x


    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):      #Convolutional Block for ResNet architecture
        
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def resnet50(self, x):
        a = Conv2D(64, (3, 3), strides=(1, 1), name='custom_conv1')(x)                          #Low-level feature extraction for decoder 
        x = ZeroPadding2D((2, 2))(x)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=3, name='bn_conv1')(x)
        x = Activation('relu')(x)
        b = Conv2D(128, (3, 3), strides=(1, 1), name='custom_conv2')(x)                         #Low-level feature extraction for decoder

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        return x, a, b

    def decoder(self, x, a, b):
        a = GlobalAveragePooling2D()(a)
        b = Conv2D(64, (1, 1), strides=1, padding='same')(b)
        b = GlobalAveragePooling2D()(b)
        
        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x1 = multiply([x, b])
        x = add([x, x1])
        x = UpSampling2D(size=(2, 2))(x)
        
        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x2 = multiply([x, a])
        x = add([x, x2])
        x = UpSampling2D(size=(2, 2))(x)
        
        x = Conv2D(128, (3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
            
        x = Conv2D(1, 1, padding='same', activation='sigmoid')(x)
        return x
    
    def M_FPM(self, x):
        
        pool = MaxPooling2D((2, 2), strides=(1,1), padding='same')(x)
        pool = Conv2D(64, (1, 1), padding='same')(pool)
        
        d1 = Conv2D(64, (3, 3), padding='same')(x)
        
        y = concatenate([x, d1], axis=-1, name='cat4')
        y = Activation('relu')(y)
        d4 = Conv2D(64, (3, 3), padding='same', dilation_rate=4)(y)
        
        y = concatenate([x, d4], axis=-1, name='cat8')
        y = Activation('relu')(y)
        d8 = Conv2D(64, (3, 3), padding='same', dilation_rate=8)(y)
        
        y = concatenate([x, d8], axis=-1, name='cat16')
        y = Activation('relu')(y)
        d16 = Conv2D(64, (3, 3), padding='same', dilation_rate=16)(y)
        
        x = concatenate([pool, d1, d4, d8, d16], axis=-1)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout2D(0.25)(x)
        return x


    def initModel(self, dataset_name):
        assert dataset_name in ['CDnet'], 'dataset_name must ["CDnet]'
        assert len(self.img_shape)==3
        h, w, d = self.img_shape
        
        net_input = Input(shape=(h, w, d), name='net_input')
        resnet50_output = self.resnet50(net_input)
        model = Model(inputs=net_input, outputs=resnet50_output, name='model')
        model.load_weights(self.resnet50_weights_path, by_name=True)
        
        for layer in model.layers:                                        #Freezing all the layers and blocks except the last(third) block used in the encoder model
            if('bn3' in layer.name or 'res3' in layer.name):
               layer.trainable = True
            else:
                layer.trainable = False
                
        x, a, b = model.output                   
        x = self.M_FPM(x)
        x = self.decoder(x, a, b)
        
        # pad in case of CDnet2014
        if dataset_name=='CDnet':
            if(self.scene=='tramCrossroad_1fps'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0), method_name=self.method_name)(x)
            elif(self.scene=='bridgeEntry'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,2), method_name=self.method_name)(x)
            elif(self.scene=='fluidHighway'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0), method_name=self.method_name)(x)
            elif(self.scene=='streetCornerAtNight'): 
                x = MyUpSampling2D(size=(1,1), num_pixels=(1,0), method_name=self.method_name)(x)
                x = Cropping2D(cropping=((0, 0),(0, 1)))(x)
            elif(self.scene=='tramStation'):  
                x = Cropping2D(cropping=((1, 0),(0, 0)))(x)
            elif(self.scene=='twoPositionPTZCam'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(0,2), method_name=self.method_name)(x)
            elif(self.scene=='turbulence2'):
                x = Cropping2D(cropping=((1, 0),(0, 0)))(x)
                x = MyUpSampling2D(size=(1,1), num_pixels=(0,1), method_name=self.method_name)(x)
            elif(self.scene=='turbulence3'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0), method_name=self.method_name)(x)

        vision_model = Model(inputs=net_input, outputs=x, name='vision_model')
        opt = keras.optimizers.RMSprop(lr = self.lr, rho=0.9, epsilon=1e-08, decay=0.)
                
        vision_model.compile(loss=loss, optimizer=opt, metrics=[acc])
        return vision_model
 
