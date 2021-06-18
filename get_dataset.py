

#AUTO = tf.data.experimental.AUTOTUNE
#BATCH_SIZE = 4096
def get_dataset(batch_size):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    # On Kaggle you can also use KaggleDatasets().get_gcs_path() to obtain the GCS path of a Kaggle dataset
    #filenames = tf.io.gfile.glob('gs://celeba_bucket/*.tfrecord')
    filenames = tf.io.gfile.glob(GCS_DS_PATH + '/celeba.tfrecord')
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    dataset = dataset.map(_parse_image_function, num_parallel_calls=AUTO)
    dataset = dataset.repeat().shuffle(200000).batch(batch_size,drop_remainder=True).prefetch(AUTO)

    return dataset
