import os, argparse

import tensorflow as tf
import data

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

dir = os.path.dirname(os.path.realpath(__file__))


def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


def load_graph():
    frozen_graph_filename = data.MODEL_DIR + '/frozen_model.pb'
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def)
    return graph


def test():
    # We use our "load_graph" function
    graph = load_graph()

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions

    # We access the input and output nodes

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants
        # y_out = sess.run(y, feed_dict={
        #     x: [[3, 5, 7, 4, 5, 1, 1, 1, 1, 1]]  # < 45
        # })
        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case
        # print(y_out)  # [[ False ]] Yay, it works!
        pass


if __name__ == '__main__':
    arch.yolo_arch_fast_020(tf.placeholder(tf.float32, shape=(1, data.IMAGE_WIDTH, data.IMAGE_HEIGHT, 1), name='Input'), True, 0.1)
    ops = []
    for op in tf.get_default_graph().get_operations():
        print(op.name + ',', end='')
    freeze_graph(data.MODEL_DIR,
                 'ManibusConv/conv1/conv1_1/weights/Initializer/random_uniform/shape,ManibusConv/conv1/conv1_1/weights/Initializer/random_uniform/min,ManibusConv/conv1/conv1_1/weights/Initializer/random_uniform/max,ManibusConv/conv1/conv1_1/weights/Initializer/random_uniform/RandomUniform,ManibusConv/conv1/conv1_1/weights/Initializer/random_uniform/sub,ManibusConv/conv1/conv1_1/weights/Initializer/random_uniform/mul,ManibusConv/conv1/conv1_1/weights/Initializer/random_uniform,ManibusConv/conv1/conv1_1/weights,ManibusConv/conv1/conv1_1/weights/Assign,ManibusConv/conv1/conv1_1/weights/read,ManibusConv/conv1/conv1_1/biases/Initializer/zeros,ManibusConv/conv1/conv1_1/biases,ManibusConv/conv1/conv1_1/biases/Assign,ManibusConv/conv1/conv1_1/biases/read,ManibusConv/conv1/conv1_1/dilation_rate,ManibusConv/conv1/conv1_1/Conv2D,ManibusConv/conv1/conv1_1/BiasAdd,ManibusConv/conv1/conv1_1/Relu,ManibusConv/conv1/conv1_2/weights/Initializer/random_uniform/shape,ManibusConv/conv1/conv1_2/weights/Initializer/random_uniform/min,ManibusConv/conv1/conv1_2/weights/Initializer/random_uniform/max,ManibusConv/conv1/conv1_2/weights/Initializer/random_uniform/RandomUniform,ManibusConv/conv1/conv1_2/weights/Initializer/random_uniform/sub,ManibusConv/conv1/conv1_2/weights/Initializer/random_uniform/mul,ManibusConv/conv1/conv1_2/weights/Initializer/random_uniform,ManibusConv/conv1/conv1_2/weights,ManibusConv/conv1/conv1_2/weights/Assign,ManibusConv/conv1/conv1_2/weights/read,ManibusConv/conv1/conv1_2/biases/Initializer/zeros,ManibusConv/conv1/conv1_2/biases,ManibusConv/conv1/conv1_2/biases/Assign,ManibusConv/conv1/conv1_2/biases/read,ManibusConv/conv1/conv1_2/dilation_rate,ManibusConv/conv1/conv1_2/Conv2D,ManibusConv/conv1/conv1_2/BiasAdd,ManibusConv/conv1/conv1_2/Relu,ManibusConv/pool1/MaxPool,ManibusConv/dropout1/dropout/keep_prob,ManibusConv/dropout1/dropout/Shape,ManibusConv/dropout1/dropout/random_uniform/min,ManibusConv/dropout1/dropout/random_uniform/max,ManibusConv/dropout1/dropout/random_uniform/RandomUniform,ManibusConv/dropout1/dropout/random_uniform/sub,ManibusConv/dropout1/dropout/random_uniform/mul,ManibusConv/dropout1/dropout/random_uniform,ManibusConv/dropout1/dropout/add,ManibusConv/dropout1/dropout/Floor,ManibusConv/dropout1/dropout/div,ManibusConv/dropout1/dropout/mul,ManibusConv/conv2/conv2_1/weights/Initializer/random_uniform/shape,ManibusConv/conv2/conv2_1/weights/Initializer/random_uniform/min,ManibusConv/conv2/conv2_1/weights/Initializer/random_uniform/max,ManibusConv/conv2/conv2_1/weights/Initializer/random_uniform/RandomUniform,ManibusConv/conv2/conv2_1/weights/Initializer/random_uniform/sub,ManibusConv/conv2/conv2_1/weights/Initializer/random_uniform/mul,ManibusConv/conv2/conv2_1/weights/Initializer/random_uniform,ManibusConv/conv2/conv2_1/weights,ManibusConv/conv2/conv2_1/weights/Assign,ManibusConv/conv2/conv2_1/weights/read,ManibusConv/conv2/conv2_1/biases/Initializer/zeros,ManibusConv/conv2/conv2_1/biases,ManibusConv/conv2/conv2_1/biases/Assign,ManibusConv/conv2/conv2_1/biases/read,ManibusConv/conv2/conv2_1/dilation_rate,ManibusConv/conv2/conv2_1/Conv2D,ManibusConv/conv2/conv2_1/BiasAdd,ManibusConv/conv2/conv2_1/Relu,ManibusConv/conv2/conv2_2/weights/Initializer/random_uniform/shape,ManibusConv/conv2/conv2_2/weights/Initializer/random_uniform/min,ManibusConv/conv2/conv2_2/weights/Initializer/random_uniform/max,ManibusConv/conv2/conv2_2/weights/Initializer/random_uniform/RandomUniform,ManibusConv/conv2/conv2_2/weights/Initializer/random_uniform/sub,ManibusConv/conv2/conv2_2/weights/Initializer/random_uniform/mul,ManibusConv/conv2/conv2_2/weights/Initializer/random_uniform,ManibusConv/conv2/conv2_2/weights,ManibusConv/conv2/conv2_2/weights/Assign,ManibusConv/conv2/conv2_2/weights/read,ManibusConv/conv2/conv2_2/biases/Initializer/zeros,ManibusConv/conv2/conv2_2/biases,ManibusConv/conv2/conv2_2/biases/Assign,ManibusConv/conv2/conv2_2/biases/read,ManibusConv/conv2/conv2_2/dilation_rate,ManibusConv/conv2/conv2_2/Conv2D,ManibusConv/conv2/conv2_2/BiasAdd,ManibusConv/conv2/conv2_2/Relu,ManibusConv/conv2/conv2_3/weights/Initializer/random_uniform/shape,ManibusConv/conv2/conv2_3/weights/Initializer/random_uniform/min,ManibusConv/conv2/conv2_3/weights/Initializer/random_uniform/max,ManibusConv/conv2/conv2_3/weights/Initializer/random_uniform/RandomUniform,ManibusConv/conv2/conv2_3/weights/Initializer/random_uniform/sub,ManibusConv/conv2/conv2_3/weights/Initializer/random_uniform/mul,ManibusConv/conv2/conv2_3/weights/Initializer/random_uniform,ManibusConv/conv2/conv2_3/weights,ManibusConv/conv2/conv2_3/weights/Assign,ManibusConv/conv2/conv2_3/weights/read,ManibusConv/conv2/conv2_3/biases/Initializer/zeros,ManibusConv/conv2/conv2_3/biases,ManibusConv/conv2/conv2_3/biases/Assign,ManibusConv/conv2/conv2_3/biases/read,ManibusConv/conv2/conv2_3/dilation_rate,ManibusConv/conv2/conv2_3/Conv2D,ManibusConv/conv2/conv2_3/BiasAdd,ManibusConv/conv2/conv2_3/Relu,ManibusConv/pool2/MaxPool,ManibusConv/dropout2/dropout/keep_prob,ManibusConv/dropout2/dropout/Shape,ManibusConv/dropout2/dropout/random_uniform/min,ManibusConv/dropout2/dropout/random_uniform/max,ManibusConv/dropout2/dropout/random_uniform/RandomUniform,ManibusConv/dropout2/dropout/random_uniform/sub,ManibusConv/dropout2/dropout/random_uniform/mul,ManibusConv/dropout2/dropout/random_uniform,ManibusConv/dropout2/dropout/add,ManibusConv/dropout2/dropout/Floor,ManibusConv/dropout2/dropout/div,ManibusConv/dropout2/dropout/mul,ManibusConv/conv3/conv3_1/weights/Initializer/random_uniform/shape,ManibusConv/conv3/conv3_1/weights/Initializer/random_uniform/min,ManibusConv/conv3/conv3_1/weights/Initializer/random_uniform/max,ManibusConv/conv3/conv3_1/weights/Initializer/random_uniform/RandomUniform,ManibusConv/conv3/conv3_1/weights/Initializer/random_uniform/sub,ManibusConv/conv3/conv3_1/weights/Initializer/random_uniform/mul,ManibusConv/conv3/conv3_1/weights/Initializer/random_uniform,ManibusConv/conv3/conv3_1/weights,ManibusConv/conv3/conv3_1/weights/Assign,ManibusConv/conv3/conv3_1/weights/read,ManibusConv/conv3/conv3_1/biases/Initializer/zeros,ManibusConv/conv3/conv3_1/biases,ManibusConv/conv3/conv3_1/biases/Assign,ManibusConv/conv3/conv3_1/biases/read,ManibusConv/conv3/conv3_1/dilation_rate,ManibusConv/conv3/conv3_1/Conv2D,ManibusConv/conv3/conv3_1/BiasAdd,ManibusConv/conv3/conv3_1/Relu,ManibusConv/conv3/conv3_2/weights/Initializer/random_uniform/shape,ManibusConv/conv3/conv3_2/weights/Initializer/random_uniform/min,ManibusConv/conv3/conv3_2/weights/Initializer/random_uniform/max,ManibusConv/conv3/conv3_2/weights/Initializer/random_uniform/RandomUniform,ManibusConv/conv3/conv3_2/weights/Initializer/random_uniform/sub,ManibusConv/conv3/conv3_2/weights/Initializer/random_uniform/mul,ManibusConv/conv3/conv3_2/weights/Initializer/random_uniform,ManibusConv/conv3/conv3_2/weights,ManibusConv/conv3/conv3_2/weights/Assign,ManibusConv/conv3/conv3_2/weights/read,ManibusConv/conv3/conv3_2/biases/Initializer/zeros,ManibusConv/conv3/conv3_2/biases,ManibusConv/conv3/conv3_2/biases/Assign,ManibusConv/conv3/conv3_2/biases/read,ManibusConv/conv3/conv3_2/dilation_rate,ManibusConv/conv3/conv3_2/Conv2D,ManibusConv/conv3/conv3_2/BiasAdd,ManibusConv/conv3/conv3_2/Relu,ManibusConv/conv3/conv3_3/weights/Initializer/random_uniform/shape,ManibusConv/conv3/conv3_3/weights/Initializer/random_uniform/min,ManibusConv/conv3/conv3_3/weights/Initializer/random_uniform/max,ManibusConv/conv3/conv3_3/weights/Initializer/random_uniform/RandomUniform,ManibusConv/conv3/conv3_3/weights/Initializer/random_uniform/sub,ManibusConv/conv3/conv3_3/weights/Initializer/random_uniform/mul,ManibusConv/conv3/conv3_3/weights/Initializer/random_uniform,ManibusConv/conv3/conv3_3/weights,ManibusConv/conv3/conv3_3/weights/Assign,ManibusConv/conv3/conv3_3/weights/read,ManibusConv/conv3/conv3_3/biases/Initializer/zeros,ManibusConv/conv3/conv3_3/biases,ManibusConv/conv3/conv3_3/biases/Assign,ManibusConv/conv3/conv3_3/biases/read,ManibusConv/conv3/conv3_3/dilation_rate,ManibusConv/conv3/conv3_3/Conv2D,ManibusConv/conv3/conv3_3/BiasAdd,ManibusConv/conv3/conv3_3/Relu,ManibusConv/pool3/MaxPool,ManibusConv/dropout3/dropout/keep_prob,ManibusConv/dropout3/dropout/Shape,ManibusConv/dropout3/dropout/random_uniform/min,ManibusConv/dropout3/dropout/random_uniform/max,ManibusConv/dropout3/dropout/random_uniform/RandomUniform,ManibusConv/dropout3/dropout/random_uniform/sub,ManibusConv/dropout3/dropout/random_uniform/mul,ManibusConv/dropout3/dropout/random_uniform,ManibusConv/dropout3/dropout/add,ManibusConv/dropout3/dropout/Floor,ManibusConv/dropout3/dropout/div,ManibusConv/dropout3/dropout/mul,ManibusConv/conv4/conv4_1/weights/Initializer/random_uniform/shape,ManibusConv/conv4/conv4_1/weights/Initializer/random_uniform/min,ManibusConv/conv4/conv4_1/weights/Initializer/random_uniform/max,ManibusConv/conv4/conv4_1/weights/Initializer/random_uniform/RandomUniform,ManibusConv/conv4/conv4_1/weights/Initializer/random_uniform/sub,ManibusConv/conv4/conv4_1/weights/Initializer/random_uniform/mul,ManibusConv/conv4/conv4_1/weights/Initializer/random_uniform,ManibusConv/conv4/conv4_1/weights,ManibusConv/conv4/conv4_1/weights/Assign,ManibusConv/conv4/conv4_1/weights/read,ManibusConv/conv4/conv4_1/biases/Initializer/zeros,ManibusConv/conv4/conv4_1/biases,ManibusConv/conv4/conv4_1/biases/Assign,ManibusConv/conv4/conv4_1/biases/read,ManibusConv/conv4/conv4_1/dilation_rate,ManibusConv/conv4/conv4_1/Conv2D,ManibusConv/conv4/conv4_1/BiasAdd,ManibusConv/conv4/conv4_1/Relu,ManibusConv/conv4/conv4_2/weights/Initializer/random_uniform/shape,ManibusConv/conv4/conv4_2/weights/Initializer/random_uniform/min,ManibusConv/conv4/conv4_2/weights/Initializer/random_uniform/max,ManibusConv/conv4/conv4_2/weights/Initializer/random_uniform/RandomUniform,ManibusConv/conv4/conv4_2/weights/Initializer/random_uniform/sub,ManibusConv/conv4/conv4_2/weights/Initializer/random_uniform/mul,ManibusConv/conv4/conv4_2/weights/Initializer/random_uniform,ManibusConv/conv4/conv4_2/weights,ManibusConv/conv4/conv4_2/weights/Assign,ManibusConv/conv4/conv4_2/weights/read,ManibusConv/conv4/conv4_2/biases/Initializer/zeros,ManibusConv/conv4/conv4_2/biases,ManibusConv/conv4/conv4_2/biases/Assign,ManibusConv/conv4/conv4_2/biases/read,ManibusConv/conv4/conv4_2/dilation_rate,ManibusConv/conv4/conv4_2/Conv2D,ManibusConv/conv4/conv4_2/BiasAdd,ManibusConv/conv4/conv4_2/Relu,ManibusConv/conv4/conv4_3/weights/Initializer/random_uniform/shape,ManibusConv/conv4/conv4_3/weights/Initializer/random_uniform/min,ManibusConv/conv4/conv4_3/weights/Initializer/random_uniform/max,ManibusConv/conv4/conv4_3/weights/Initializer/random_uniform/RandomUniform,ManibusConv/conv4/conv4_3/weights/Initializer/random_uniform/sub,ManibusConv/conv4/conv4_3/weights/Initializer/random_uniform/mul,ManibusConv/conv4/conv4_3/weights/Initializer/random_uniform,ManibusConv/conv4/conv4_3/weights,ManibusConv/conv4/conv4_3/weights/Assign,ManibusConv/conv4/conv4_3/weights/read,ManibusConv/conv4/conv4_3/biases/Initializer/zeros,ManibusConv/conv4/conv4_3/biases,ManibusConv/conv4/conv4_3/biases/Assign,ManibusConv/conv4/conv4_3/biases/read,ManibusConv/conv4/conv4_3/dilation_rate,ManibusConv/conv4/conv4_3/Conv2D,ManibusConv/conv4/conv4_3/BiasAdd,ManibusConv/conv4/conv4_3/Relu,ManibusConv/dropout4/dropout/keep_prob,ManibusConv/dropout4/dropout/Shape,ManibusConv/dropout4/dropout/random_uniform/min,ManibusConv/dropout4/dropout/random_uniform/max,ManibusConv/dropout4/dropout/random_uniform/RandomUniform,ManibusConv/dropout4/dropout/random_uniform/sub,ManibusConv/dropout4/dropout/random_uniform/mul,ManibusConv/dropout4/dropout/random_uniform,ManibusConv/dropout4/dropout/add,ManibusConv/dropout4/dropout/Floor,ManibusConv/dropout4/dropout/div,ManibusConv/dropout4/dropout/mul,ManibusConv/conv5/weights/Initializer/random_uniform/shape,ManibusConv/conv5/weights/Initializer/random_uniform/min,ManibusConv/conv5/weights/Initializer/random_uniform/max,ManibusConv/conv5/weights/Initializer/random_uniform/RandomUniform,ManibusConv/conv5/weights/Initializer/random_uniform/sub,ManibusConv/conv5/weights/Initializer/random_uniform/mul,ManibusConv/conv5/weights/Initializer/random_uniform,ManibusConv/conv5/weights,ManibusConv/conv5/weights/Assign,ManibusConv/conv5/weights/read,ManibusConv/conv5/biases/Initializer/zeros,ManibusConv/conv5/biases,ManibusConv/conv5/biases/Assign,ManibusConv/conv5/biases/read,ManibusConv/conv5/dilation_rate,ManibusConv/conv5/Conv2D,ManibusConv/conv5/BiasAdd,ManibusConv/conv5/Relu,ManibusConv/dropout5/dropout/keep_prob,ManibusConv/dropout5/dropout/Shape,ManibusConv/dropout5/dropout/random_uniform/min,ManibusConv/dropout5/dropout/random_uniform/max,ManibusConv/dropout5/dropout/random_uniform/RandomUniform,ManibusConv/dropout5/dropout/random_uniform/sub,ManibusConv/dropout5/dropout/random_uniform/mul,ManibusConv/dropout5/dropout/random_uniform,ManibusConv/dropout5/dropout/add,ManibusConv/dropout5/dropout/Floor,ManibusConv/dropout5/dropout/div,ManibusConv/dropout5/dropout/mul,ManibusConv/sm1/Reshape/shape,ManibusConv/sm1/Reshape,ManibusConv/sm1/Softmax,ManibusConv/sm1/Shape,ManibusConv/sm1/Reshape_1')
    # test()
