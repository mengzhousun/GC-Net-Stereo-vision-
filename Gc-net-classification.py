from tensorflow import keras as K
from Aufrufe1 import *
import numpy as np
BatchNormalization = K.layers.BatchNormalization
Conv2D=K.layers.Conv2D
Conv3D=K.layers.Conv3D
Conv3Dtransposed=K.layers.Conv3DTranspose
Aktivation = K.layers.Activation
Conv2Dtransposed=tf.keras.layers.Conv2DTranspose
def creat_model(shape_tuple):
    img_left=  K.Input(shape=shape_tuple.shape)
    img_right= K.Input(shape=shape_tuple.shape)
#model

    layer1=Conv2D(32, kernel_size=(5, 5),
        padding='same',
        strides=2,
        activation='relu',
             )
    layer2=Conv2D(32,kernel_size=(3,3),
              padding='same',
              activation='relu')
    layer3=Conv2D(32,kernel_size=(3,3),
              padding='same',
              activation='relu')
    layer4=Conv2D(32,kernel_size=(3,3),
              padding='same',
              activation='relu')
    layer5=Conv2D(32,kernel_size=(3,3),
              padding='same',
              activation='relu')
    layer6=Conv2D(32,kernel_size=(3,3),
              padding='same',
              activation='relu')
    layer7=Conv2D(32,kernel_size=(3,3),
              padding='same',
              activation='relu')
    layer8=Conv2D(32,kernel_size=(3,3),
              padding='same',
              activation='relu')
    layer9=Conv2D(32,kernel_size=(3,3),
              padding='same',
              activation='relu')
    layer10=Conv2D(32,kernel_size=(3,3),
              padding='same',
              activation='relu')
    layer11=Conv2D(32,kernel_size=(3,3),
              padding='same',
              activation='relu')
    layer12=Conv2D(32,kernel_size=(3,3),
              padding='same',
              activation='relu')
    layer13=Conv2D(32,kernel_size=(3,3),
              padding='same',
              activation='relu')
    layer14=Conv2D(32,kernel_size=(3,3),
              padding='same',
              activation='relu')
    layer15=Conv2D(32,kernel_size=(3,3),
              padding='same',
              activation='relu')
    layer16=Conv2D(32,kernel_size=(3,3),
              padding='same',
              activation='relu')
    layer17=Conv2D(32,kernel_size=(3,3),
              padding='same',
              activation='relu')
    layer18=Conv2D(32,kernel_size=(3,3),
              padding='same',
              )

    with tf.name_scope("img_left"):
     x_l1=layer1(img_left)
     x_l1 = BatchNormalization()(x_l1)
     x_l2=layer2(x_l1)
     x_l2=BatchNormalization()(x_l2)
     x_l3 = layer3(x_l2)
     x_l3 = BatchNormalization()(x_l3)
     x_l3=tf.keras.layers.add([x_l1,x_l3])
     x_l4 = layer4(x_l3)
     x_l4 = BatchNormalization()(x_l4)
     x_l5 = layer5(x_l4)
     x_l5 = BatchNormalization()(x_l5)
     x_l5 = tf.keras.layers.add([x_l3, x_l5])
     x_l6 = layer6(x_l5)
     x_l6 = BatchNormalization()(x_l6)
     x_l7 = layer7(x_l6)
     x_l7 = BatchNormalization()(x_l7)
     x_l7 = tf.keras.layers.add([x_l5, x_l7])
     x_l8 = layer8(x_l7)
     x_l8 = BatchNormalization()(x_l8)
     x_l9 = layer9(x_l8)
     x_l9 = BatchNormalization()(x_l9)
     x_l9=tf.keras.layers.add([x_l9,x_l7])
     x_l10 = layer10(x_l9)
     x_l10 = BatchNormalization()(x_l10)
     x_l11 = layer11(x_l10)
     x_l11 = BatchNormalization()(x_l11)
     x_l11 = tf.keras.layers.add([x_l9, x_l11])
     x_l12 = layer12(x_l11)
     x_l12 = BatchNormalization()(x_l12)
     x_l13 = layer13(x_l12)
     x_l13 = BatchNormalization()(x_l13)
     x_l13 = tf.keras.layers.add([x_l11, x_l13])
     x_l14 = layer14(x_l13)
     x_l14 = BatchNormalization()(x_l14)
     x_l15 = layer15(x_l14)
     x_l15 = BatchNormalization()(x_l15)
     x_l15 =tf.keras.layers.add([x_l13,x_l15])
     x_l16 = layer16(x_l15)
     x_l16 = BatchNormalization()(x_l16)
     x_l17 = layer17(x_l16)
     x_l17 = BatchNormalization()(x_l17)
     x_l17 = tf.keras.layers.add([x_l17, x_l15])
     x_l18 =  layer18(x_l17)
     #x_l18 =tf.keras.layers.add([x_l18,x_l16])

    with tf.name_scope("img_right"):
     x_r1 = layer1(img_right)
     x_r1 = BatchNormalization()(x_r1)
     x_r2 = layer2(x_r1)
     x_r2 = BatchNormalization()(x_r2)
     x_r3 = layer3(x_r2)
     x_r3 = BatchNormalization()(x_r3)
     x_r3 = tf.keras.layers.add([x_r1, x_r3])
     x_r4 = layer4(x_r3)
     x_r4 = BatchNormalization()(x_r4)
     x_r5 = layer5(x_r4)
     x_r5 = BatchNormalization()(x_r5)
     x_r5 = tf.keras.layers.add([x_r5, x_r3])
     x_r6 = layer6(x_r5)
     x_r6 = BatchNormalization()(x_r6)
     x_r7 = layer7(x_r6)
     x_r7 = BatchNormalization()(x_r7)
     x_r7 = tf.keras.layers.add([x_r7, x_r5])
     x_r8 = layer8(x_r7)
     x_r8 = BatchNormalization()(x_r8)
     x_r9 = layer9(x_r8)
     x_r9 = BatchNormalization()(x_r9)
     x_r9 = tf.keras.layers.add([x_r9, x_r7])
     x_r10 = layer10(x_r9)
     x_r10 = BatchNormalization()(x_r10)
     x_r11 = layer11(x_r10)
     x_r11 = BatchNormalization()(x_r11)
     x_r11 = tf.keras.layers.add([x_r11, x_r9])
     x_r12 = layer12(x_r11)
     x_r12 = BatchNormalization()(x_r12)
     x_r13 = layer13(x_r12)
     x_r13 = BatchNormalization()(x_r13)
     x_r13 = tf.keras.layers.add([x_r11, x_r13])
     x_r14 = layer14(x_r13)
     x_r14 = BatchNormalization()(x_r14)
     x_r15 = layer15(x_r14)
     x_r15 = BatchNormalization()(x_r15)
     x_r15 = tf.keras.layers.add([x_r13, x_r15])
     x_r16 = layer16(x_r15)
     x_r16 = BatchNormalization()(x_r16)
     x_r17 = layer17(x_r16)
     x_r17 = BatchNormalization()(x_r17)
     x_r17 = tf.keras.layers.add([x_r17, x_r15])
     x_r18 = layer18(x_r17)


     left_cost=tf.keras.layers.Lambda(expand)(x_l18)
     right_cost=tf.keras.layers.Lambda(expand)(x_r18)
     cost_volum=tf.keras.layers.concatenate([left_cost,right_cost],axis=4)
     total=K.layers.Lambda(cost5)(cost_volum)


     x19 = Conv3D(32, kernel_size=(3, 3,3),
                  padding='same',
                  activation='relu'
                  )(total)
     x19 = BatchNormalization()(x19)
     x20 = Conv3D(32, kernel_size=(3, 3,3), padding='same',
                  activation='relu')(x19)
     x20 = BatchNormalization()(x20)

     x21 = Conv3D(64, kernel_size=(3, 3,3), padding='same',
                  strides=2,
                  activation='relu')(total)
     x21 = BatchNormalization()(x21)
     x22 = Conv3D(64, kernel_size=(3, 3,3),
                  padding='same',
                  activation='relu'
                  )(x21)
     x22 = BatchNormalization()(x22)
     x23 = Conv3D(64, kernel_size=(3, 3,3),
                  padding='same',
                  activation='relu'
                  )(x22)
     x23 = BatchNormalization()(x23)
     x24 = Conv3D(64, kernel_size=(3, 3,3), padding='same',
                  strides=2,
                  activation='relu')(x21)
     x24 = BatchNormalization()(x24)
     x25 = Conv3D(64, kernel_size=(3,3,3),
                  padding='same',
                  activation='relu'
                  )(x24)
     x25 = BatchNormalization()(x25)
     x26 = Conv3D(64, kernel_size=(3, 3,3),
                  padding='same',
                  activation='relu'
                  )(x25)
     x26 = BatchNormalization()(x26)
     x27 = Conv3D(64, kernel_size=(3,3,3), padding='same',
                  strides=2,
                  activation='relu')(x24)
     x27 = BatchNormalization()(x27)
     x28 = Conv3D(64, kernel_size=(3,3,3),
                  padding='same',
                  activation='relu'
                  )(x27)
     x28 = BatchNormalization()(x28)
     x29 = Conv3D(64, kernel_size=(3,3,3),
                  padding='same',
                  activation='relu'
                  )(x28)
     x29 = BatchNormalization()(x29)
     x30 = Conv3D(128, kernel_size=(3,3,3),
                  padding='same',
                  strides=2,
                  activation='relu'
                  )(x27)
     x30 = BatchNormalization()(x30)
     x31 = Conv3D(128, kernel_size=(3,3,3), padding='same',
                  activation='relu')(x30)
     x31 = BatchNormalization()(x31)
     x32 = Conv3D(128, kernel_size=(3, 3,3),
                  padding='same',
                  activation='relu'
                  )(x31)
     x32 = BatchNormalization()(x32)
     x33 = Conv3Dtransposed(64, kernel_size=(3,3,3),
                            padding='same',
                            strides=2,
                            activation='relu'
                            )(x32)
     x33 = BatchNormalization()(x33)
     x33 = tf.keras.layers.add([x29, x33])
     x34 = Conv3Dtransposed(64, kernel_size=(3,3,3),
                            activation='relu',
                            strides=2,
                            padding='same'
                            )(x33)
     x34 = BatchNormalization()(x34)
     x34 = tf.keras.layers.add([x26, x34])
     x35 = Conv3Dtransposed(64, kernel_size=(3,3 ,3), strides=2,
                            activation='relu',
                            padding='same')(x34)
     x35 = BatchNormalization()(x35)
     x35 = tf.keras.layers.add([x35, x23])
     x36 = Conv3Dtransposed(32, kernel_size=(3,3,3), strides=2,
                            activation='relu',
                            padding='same')(x35)
     x36 = BatchNormalization()(x36)
     x36 = tf.keras.layers.add([x20, x36])
     x37 = Conv3Dtransposed(1, kernel_size=(3,3,3),
                            strides=2,
                            padding='same')(x36)
     P=tf.keras.layers.Lambda(softmax)(x37)

     return K.Model(inputs=[img_left, img_right], outputs=P)
