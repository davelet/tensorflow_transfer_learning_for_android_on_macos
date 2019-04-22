import tensorflow as tf

NODE_OPS = ['Placeholder','Identity']
MODEL_FILE = '/tmp/output_graph.pb'

gf = tf.GraphDef()
gf.ParseFromString(open(MODEL_FILE,'rb').read())

print([n.name + '=>' +  n.op for n in gf.node if n.op in (NODE_OPS)])