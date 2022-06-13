import tensorflow._api.v2.compat.v1 as tf  #tf_gpu version 2.3
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from tensorflow.python.framework import ops

model_path = r'saved_model.pb'
with tf.Session(graph=ops.Graph()) as sess:
  with tf.gfile.GFile(model_path, "rb") as f:
    data = compat.as_bytes(f.read())
    sm = saved_model_pb2.SavedModel()
    sm.ParseFromString(data)
    g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
    train_writer = tf.summary.FileWriter("../log")
    train_writer.add_graph(sess.graph)
    train_writer.flush()
    train_writer.close()

