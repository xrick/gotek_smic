#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
import os


# In[ ]:


trained_checkpoint_prefix = "model_512x32_lr_0_008_20200110"
saved_model_path = '512x32_to_tflite_test2'
export_dir = "export_dir"

print(export_dir)

graph = tf.Graph()
with tf.compat.v1.Session(graph=graph) as sess:
    # Restore from checkpoint
    loader = tf.compat.v1.train.import_meta_graph(saved_model_path + "\\"+ trained_checkpoint_prefix + '.meta')
    loader.restore(sess, saved_model_path)

    # Export checkpoint to SavedModel
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.TRAINING, tf.saved_model.SERVING],
                                         strip_default_attrs=True)
    builder.save()            


# In[ ]:




