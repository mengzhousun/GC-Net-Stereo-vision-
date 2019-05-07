from preprocessing  import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from Gc-net classi import creat_model
from function import *
batch_size = 1
dataset = input_from_dataset(buffer=1024)
dataset_test = input_from_dataset(filename="/daten/b/sceneflow_disp_test.tfrecord", buffer=128)
dataset_test = dataset_test.batch(batch_size)
dataset = dataset.repeat()
dataset = dataset.batch(batch_size)
interator=dataset.make_one_shot_iterator()
next_batch=interator.get_next()
next_batch_test = dataset_test.repeat().make_one_shot_iterator().get_next()
img_left_test = next_batch_test[0][:,0]
img_right_test = next_batch_test[0][:,1]
disp_left_test = next_batch_test[1][:,0]

images=next_batch[0]
disp=next_batch[1]
img_left=images[:,0]
img_right=images[:,1]
disp_left=disp[:,0]
disp_right=disp[:,1]

print(disp)
print('left_shape',img_left)
print('left_disp',disp_left)
# limit gpu-memory consumption
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    for i in range(1):
        tensor1,tensor2=sess.run([next_batch[0],next_batch[1]])
l=tensor1[:,0]
l=l[0,:]


r=tensor1[:,1]
r=r[0,:]


b=tensor2[:,0]
b=b[0,:]
labe1=np.reshape(b,(256,512))


a=tensor2[:,1]
a=a[0,:]
labe2=np.reshape(a,(256,512))


shape_tuple=images[0][1]
Model=creat_model(shape_tuple)
# Model=tf.keras.models.load_model('Gc-Net3.h5')
Model.load_weights('Gc-Net3.h5')
pre=Model.predict_on_batch([img_left, img_right])
b=np.max(pre,axis=3)
b=b[0,:,:]
#pre1=np.argmax(pre,axis=-1)
#b=tf.where(x=,y=)

#b=Model.predict([img_left,img_right],steps=1,verbose=0)
plt.figure()
plt.subplot(2,2,1)

plt.imshow(l)


plt.subplot(2,2,2)
plt.imshow(r)

plt.subplot(2,2,3)
plt.imshow(labe1)

plt.subplot(2,2,4)
plt.imshow(b)

plt.show()
