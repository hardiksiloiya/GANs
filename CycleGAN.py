import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.layers import Activation,Concatenate,Conv2D,UpSampling2D,Dense,Flatten,Input
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model


def batch(temp, n=1):
    l = len(temp)
    for ndx in range(0, l, n):
        yield temp[ndx:min(ndx + n, l)]

def data_loader(x1,x2):
    random.shuffle(x1)
    random.shuffle(x2)
    
    app=batch(x1,10)
    bnn=batch(x2,10)
    res=zip(app,bnn)
    return list(res)

def train():
  valid=np.ones((10,8,8,1))
  fake=np.zeros((10,8,8,1))

  for i in range(10):
    print("epoch ",i)
    for [imgea,imgeb] in data_loader(apples,bananas):
        tempa=[]
        tempb=[]
        for j in range(min(len(imgea),len(imgeb))):
            sample1=cv2.imread(imgea[j])
            sample2=cv2.imread(imgeb[j])
            sample1=cv2.resize(sample1,dsize=(256,256),interpolation=cv2.INTER_CUBIC)
            sample2=cv2.resize(sample1,dsize=(256,256),interpolation=cv2.INTER_CUBIC)
            sample1=cv2.cvtColor(sample1,cv2.COLOR_BGR2RGB)
            sample2=cv2.cvtColor(sample2,cv2.COLOR_BGR2RGB)
            tempa.append(sample1)
            tempb.append(sample2)
        imga=np.array(tempa,dtype=np.float32)
        imgb=np.array(tempb,dtype=np.float32)

        fakeb=generator_ab(imga)
        fakea=generator_ba(imgb)

        daloss1=discriminator_a.train_on_batch(imga,valid)
        daloss2=discriminator_a.train_on_batch(fakea,fake)

        dbloss1=discriminator_b.train_on_batch(imgb,valid)
        dbloss2=discriminator_b.train_on_batch(fakeb,fake)
        dloss=0.5*np.add((0.5*np.add(daloss1,daloss2)),(0.5*np.add(dbloss1,dbloss2)))
        gloss=combined.train_on_batch([imga,imgb],[valid,valid,imga,imgb,imga,imgb]) 
        #print(dloss)
        #print(gloss)

    testimg=cv2.imread('C:/Users/hardi/Downloads/fruits-360/Training/apple/14_100.jpg')
    testimg=cv2.resize(testimg,dsize=(256,256),interpolation=cv2.INTER_CUBIC)

    testimg=[cv2.cvtColor(testimg,cv2.COLOR_BGR2RGB)]
    testimg=np.array(testimg,np.float32)

    testout=generator_ab.predict(testimg)
    testout=np.reshape(testout,(256,256,3))
    plt.imshow(testout)
    plt.show()


if __name__ == '__main__':
    #gpu_options = tf.GPUOptions(allow_growth=True)
    #session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    path='C:/Users/hardi/Downloads/fruits-360/Training/'
    apples=glob.glob(path+'apple/*')
    bananas=glob.glob(path+'banana/*')
    dat=data_loader(apples,bananas)

    #generator A -> B
    a1=Input((256,256,3))
    generator_ab_input=a1
    a2=Conv2D(32,5,2,"same")(a1)
    a2=Activation('relu')(a2)
    a2=InstanceNormalization()(a2)

    a3=Conv2D(32,5,2,"same")(a2)
    a3=Activation('relu')(a3)
    a3=InstanceNormalization()(a3)

    a4=Conv2D(32,5,2,"same")(a3)
    a4=Activation('relu')(a4)
    a4=InstanceNormalization()(a4)

    a5=UpSampling2D(2)(a4)
    a5=Conv2D(32,5,1,"same")(a5)
    a5=Activation('relu')(a5)
    a5=Concatenate()([a5,a3])

    a5=UpSampling2D(2)(a5)
    a5=Conv2D(32,5,1,"same")(a5)
    a5=Activation('relu')(a5)
    a5=Concatenate()([a5,a2])

    a5=UpSampling2D(2)(a5)
    a5=Conv2D(3,5,1,"same")(a5)
    a5=Activation('relu')(a5)
    generator_ab_output=a5

    generator_ab=Model(generator_ab_input,generator_ab_output)

    #generator_ab.summary()

    #####

    #generator B -> A
    b1=Input((256,256,3))
    generator_ba_input=b1
    b2=Conv2D(32,5,2,"same")(b1)
    b2=Activation('relu')(b2)
    b2=InstanceNormalization()(b2)

    b3=Conv2D(32,5,2,"same")(b2)
    b3=Activation('relu')(b3)
    b3=InstanceNormalization()(b3)

    b4=Conv2D(32,5,2,"same")(b3)
    b4=Activation('relu')(b4)
    b4=InstanceNormalization()(b4)

    b5=UpSampling2D(2)(b4)
    b5=Conv2D(32,5,1,"same")(b5)
    b5=Activation('relu')(b5)
    b5=Concatenate()([b5,b3])

    b5=UpSampling2D(2)(b5)
    b5=Conv2D(32,5,1,"same")(b5)
    b5=Activation('relu')(b5)
    b5=Concatenate()([b5,b2])

    b5=UpSampling2D(2)(b5)
    b5=Conv2D(3,5,1,"same")(b5)
    b5=Activation('relu')(b5)
    generator_ba_output=b5

    generator_ba=Model(generator_ba_input,generator_ba_output)

    #generator_ba.summary()

    #####

    d1=Input((256,256,3))
    discriminator_a_input=d1
    d1=Conv2D(32,5,2,'same')(d1)
    d1=Activation('relu')(d1)

    d1=Conv2D(64,5,2,'same')(d1)
    d1=Activation('relu')(d1)
    d1=InstanceNormalization()(d1)

    d1=Conv2D(64,5,2,'same')(d1)
    d1=Activation('relu')(d1)
    d1=InstanceNormalization()(d1)

    d1=Conv2D(32,5,2,'same')(d1)
    d1=Activation('relu')(d1)
    d1=InstanceNormalization()(d1)

    d1=Conv2D(1,5,2,'same')(d1)
    discriminator_a_output=d1
    discriminator_a=Model(discriminator_a_input,discriminator_a_output)
    #discriminator_ab.summary()

    ######

    d2=Input((256,256,3))
    discriminator_b_input=d2
    d2=Conv2D(32,5,2,'same')(d2)
    d2=Activation('relu')(d2)

    d2=Conv2D(64,5,2,'same')(d2)
    d2=Activation('relu')(d2)
    d2=InstanceNormalization()(d2)

    d2=Conv2D(64,5,2,'same')(d2)
    d2=Activation('relu')(d2)
    d2=InstanceNormalization()(d2)

    d2=Conv2D(32,5,2,'same')(d2)
    d2=Activation('relu')(d2)
    d2=InstanceNormalization()(d2)

    d2=Conv2D(1,5,2,'same')(d2)
    discriminator_b_output=d2
    discriminator_b=Model(discriminator_b_input,discriminator_b_output)
    #discriminator_b.summary()


    ################

    discriminator_a.compile(optimizer=Adam(0.00012,0.5),loss='mse',metrics=['accuracy'])
    discriminator_b.compile(optimizer=Adam(0.00012,0.5),loss='mse',metrics=['accuracy'])

    discriminator_a.trainable=False
    discriminator_b.trainable=False
    im_a=Input((256,256,3))
    im_b=Input((256,256,3))
    fake_a=generator_ba(im_b)
    fake_b=generator_ab(im_a)
    valid_a=discriminator_a(fake_a)
    valid_b=discriminator_b(fake_b)

    recons_a=generator_ba(fake_b)
    recons_b=generator_ab(fake_a)

    id_a=generator_ba(im_a)
    id_b=generator_ab(im_b)

    combined=Model(inputs=[im_a,im_b],outputs=[valid_a,valid_b,recons_a,recons_b,id_a,id_b])
    combined.compile(loss=['mse','mse','mae','mae','mae','mae'])
    discriminator_a.trainable=True
    discriminator_b.trainable=True

    testimg=cv2.imread('C:/Users/hardi/Downloads/fruits-360/Training/apple/14_100.jpg')
    testimg=cv2.resize(testimg,dsize=(256,256),interpolation=cv2.INTER_CUBIC)

    testimg=[cv2.cvtColor(testimg,cv2.COLOR_BGR2RGB)]
    testimg=np.array(testimg,np.float32)
    generator_ab.summary()
    print(np.shape(testimg))

    testout=generator_ab.predict(testimg)
    testout=np.reshape(testout,(256,256,3))
    plt.imshow(testout)
    plt.show()

    train()
    #combined.save()


   



