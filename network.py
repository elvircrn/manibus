import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner
from tensorflow.python.client import device_lib

import architectures as arch
import data
import preprocess


def get_run_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.contrib.learn.RunConfig(model_dir=get_flags().model_dir, session_config=config)


# Run only once in main
def initialize_flags():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.flags.DEFINE_string(
        flag_name='model_dir', default_value=data.MODEL_DIR,
        docstring='Output directory for model and training stats.')


def get_flags():
    return tf.app.flags.FLAGS


def random_sample(images):
    # tf.enable_eager_execution()
    # result = arch.yolo_arch_fast_020(inputs=random.choice(images))
    return


def run_and_get_loss(params, run_config):
    # dataset = preprocess.get_dataset(data.DATA_PATH)
    # threading.Thread(target=lambda: random_sample(dataset))
    runner = learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule="train_and_evaluate",
        hparams=params
    )
    return runner[0]['loss']


def get_experiment_params(data_path=data.DATA_PATH):
    return tf.contrib.training.HParams(
        learning_rate=0.00002,
        train_steps=90000,
        min_eval_frequency=50,
        architecture=arch.yolo_ar,
        dropout=0.6,
        run_preprocess=True,
        data_path=data_path
    )


def eager_hack():
    params = get_experiment_params()
    params.train_steps = 1
    run_and_get_loss(params, get_run_config())


def objective(args):
    params = get_experiment_params()
    params.learning_rate = args['learn_rate']
    params.dropout = args['dropout']
    run_config = get_run_config()
    loss = run_and_get_loss(params, run_config)
    return loss


def experiment_fn(run_config, params):
    run_config = run_config.replace(
        save_checkpoints_steps=params.min_eval_frequency)
    estimator = get_estimator(run_config, params)
    # Setup data loaders
    if params.run_preprocess:
        print('Running preprocess')
    datasets = preprocess.get_dataset(params.data_path) if params.run_preprocess else preprocess.preprocess_ego(
        params.data_path)

    train_input_fn, train_input_hook = get_train_inputs(
        batch_size=64, datasets=datasets)
    eval_input_fn, eval_input_hook = get_test_inputs(
        batch_size=64, datasets=datasets)
    # Define the experiment
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,  # Estimator
        train_input_fn=train_input_fn,  # First-class function
        eval_input_fn=eval_input_fn,  # First-class function
        train_steps=params.train_steps,  # Mini-batch steps
        min_eval_frequency=params.min_eval_frequency,  # Eval frequency
        train_monitors=[train_input_hook],  # Hooks for training
        eval_hooks=[eval_input_hook],  # Hooks for evaluation
        eval_steps=None  # Use evaluation feeder until its empty
    )
    return experiment


def get_estimator(run_config=None, params=None):
    if run_config is None:
        run_config = get_run_config()
    if params is None:
        params = get_experiment_params()

    return tf.estimator.Estimator(
        model_fn=model_fn,  # First-class function
        params=params,  # HParams
        config=run_config  # RunConfig
    )


def calculate_loss(labels, predictions):
    def x_i(t):
        return tf.slice(t, [0, 0, 0, 0], [-1, -1, -1, 1])

    def y_i(t):
        return tf.slice(t, [0, 0, 0, 1], [-1, -1, -1, 1])

    def w_i(t):
        return tf.slice(t, [0, 0, 0, 2], [-1, -1, -1, 1])

    def h_i(t):
        return tf.slice(t, [0, 0, 0, 3], [-1, -1, -1, 1])

    def c_i(t):
        return tf.slice(t, [0, 0, 0, 4], [-1, -1, -1, 1])

    obj = tf.reduce_sum(labels, 3)
    obj = tf.cast(tf.greater(obj, 0), dtype=tf.float32)
    obj = tf.reshape(obj, [-1, data.STRIDE_W, data.STRIDE_H, 1])
    noobj = tf.subtract(tf.constant(1, dtype=tf.float32), obj)

    center_loss = tf.reduce_sum(
        tf.multiply(obj, tf.add(tf.square(tf.subtract(x_i(labels), x_i(predictions))),
                                tf.square(tf.subtract(y_i(labels), y_i(predictions))))))

    size_loss = tf.reduce_sum(
        tf.multiply(obj, tf.add(tf.square(tf.subtract(tf.sqrt(w_i(labels)), tf.sqrt(w_i(predictions)))),
                                tf.square(tf.subtract(tf.sqrt(h_i(labels)), tf.sqrt(h_i(predictions)))))))

    classification_loss = tf.add(
        tf.reduce_sum(tf.multiply(obj, tf.square(tf.subtract(c_i(labels), c_i(predictions))))),
        tf.reduce_sum(tf.multiply(noobj, tf.square(tf.subtract(c_i(labels), c_i(predictions))))))

    loss = tf.add(center_loss, size_loss)
    loss = tf.add(loss, classification_loss)

    loss = tf.Print(loss, [loss], 'Loss: ')

    return loss


def model_fn(features, labels, mode, params):
    is_training = mode == ModeKeys.TRAIN
    # Define model's architecture
    predictions = params.architecture(inputs=features, dropout=params.dropout, is_training=is_training)
    # Loss, training and eval operations are not needed during inference.
    loss = None
    train_op = None
    if mode != ModeKeys.INFER:
        loss = calculate_loss(labels, predictions)
        train_op = get_train_op_fn(loss, params)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op
    )


def get_train_op_fn(loss, params):
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        # Hyper
        optimizer=tf.train.AdamOptimizer,
        learning_rate=params.learning_rate
    )


def f_score(predictions=None, labels=None, weights=None):
    p, update_op1 = tf.contrib.metrics.streaming_precision(predictions, labels)
    r, update_op2 = tf.contrib.metrics.streaming_recall(predictions, labels)
    eps = 1e-5
    return 2 * (p * r) / (p + r + eps), tf.group(update_op1, update_op2)


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)


def get_train_inputs(batch_size, datasets):
    iterator_initializer_hook = IteratorInitializerHook()

    def train_inputs():
        with tf.name_scope(data.TRAINING_SCOPE):
            images = datasets[0][0].reshape([-1, data.IMAGE_WIDTH, data.IMAGE_HEIGHT, 1])
            labels = datasets[0][1]
            images_placeholder = tf.placeholder(
                images.dtype, images.shape)
            labels_placeholder = tf.placeholder(
                labels.dtype, labels.shape)
            # Build dataset iterator
            dataset = tf.data.Dataset.from_tensor_slices(
                (images_placeholder, labels_placeholder))
            dataset = dataset.repeat(None)  # Infinite iterations
            dataset = dataset.shuffle(buffer_size=100)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            next_example, next_label = iterator.get_next()
            # Set run-hook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={images_placeholder: images,
                               labels_placeholder: labels})
            # Return batched (features, labels)
            return next_example, next_label

    # Return function and hook
    return train_inputs, iterator_initializer_hook


def get_test_inputs(batch_size, datasets):
    iterator_initializer_hook = IteratorInitializerHook()

    def test_inputs():
        with tf.name_scope(data.TEST_SCOPE):
            images = datasets[1][0].reshape([-1, data.IMAGE_WIDTH, data.IMAGE_HEIGHT, 1])
            labels = datasets[1][1]
            # Define placeholders
            images_placeholder = tf.placeholder(
                images.dtype, images.shape)
            labels_placeholder = tf.placeholder(
                labels.dtype, labels.shape)
            # Build dataset iterator
            dataset = tf.data.Dataset.from_tensor_slices(
                (images_placeholder, labels_placeholder))
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            next_example, next_label = iterator.get_next()
            # Set run-hook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={images_placeholder: images,
                               labels_placeholder: labels})
            return next_example, next_label

    # Return function and hook
    return test_inputs, iterator_initializer_hook


def run_experiment(argv=None):
    params = get_experiment_params(data_path=argv[0])
    run_config = tf.contrib.learn.RunConfig(model_dir=get_flags().model_dir)
    run_and_get_loss(params, run_config)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def run_network(data_path=data.DATA_PATH):
    enable_gpu = True

    if enable_gpu:
        print('Available GPUs: ', get_available_gpus())
        with tf.device("/gpu:0"):
            initialize_flags()
            tf.app.run(
                main=run_experiment,
                argv=[data_path]
            )
    else:
        initialize_flags()
        tf.app.run(
            main=run_experiment,
            argv=[data_path]
        )


def predict(estimator, images):
    images = np.reshape(images, [-1, data.IMAGE_HEIGHT, data.IMAGE_WIDTH, 1]).astype(dtype=np.float32)
    predictions = estimator.predict(input_fn=lambda: images)
    return predictions
