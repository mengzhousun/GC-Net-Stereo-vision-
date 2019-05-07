from Aufrufe import creat_model
import matplotlib.pyplot as plt
from function import *
from preprocessing import *
batch_size = 1
dataset = input_from_dataset(buffer=1024)
dataset_test = input_from_dataset(filename="/daten/b/sceneflow_disp_train.tfrecord", buffer=128)
dataset_test = dataset_test.batch(batch_size)
dataset_test = dataset_test.filter(lambda x, y: tf.less(tf.reduce_max(y), 192))
dataset = dataset.filter(lambda x, y: tf.less(tf.reduce_max(y), 192))
dataset = dataset.repeat()
dataset = dataset.batch(batch_size).prefetch(32)
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

# test images


# limit gpu-mdmory consumption
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))
print(disp_left.shape)
shape_tuple=images[0][1]
Model=creat_model(shape_tuple)
print(Model.output_shape)

optimizer=tf.keras.optimizers.RMSprop(lr=0.0001)
#loss=tf.reduce_mean(tf.keras.backend.sparse_categorical_crossentropy(target=c,output=Model.output))
Model.compile(optimizer=optimizer,loss=def_loss,metrics=[disp_accuracy])
callback = tf.keras.callbacks.TensorBoard(log_dir="./TB/")

Model.fit([img_left, img_right], disp_left,
         epochs=30,steps_per_epoch=5000, callbacks=[callback], validation_steps=int(4000/batch_size), validation_data=([img_left_test, img_right_test], disp_left_test)
        ,verbose=2 )
Model.save('Gc-Net3.h5')
