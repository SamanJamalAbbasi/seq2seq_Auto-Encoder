import tensorflow as tf
from model.auto_encoder import AutoEncoder
from data_utils import batch_iter

BATCH_SIZE = 64
NUM_EPOCHS = 100
MAX_DOCUMENT_LEN = 50


def train(train_x, train_y, word_dict):
    with tf.Session() as sess:
        model = AutoEncoder(word_dict, MAX_DOCUMENT_LEN)
        # Define training procedure
        global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(model.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

        # Summary
        tf.summary.scalar("loss", model.loss)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("auto_encoder", sess.graph)

        # Checkpoint
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(batch_x):
            feed_dict = {model.x: batch_x}
            _, step, summaries, loss, encoder_states, encoder_outputs = sess.run([train_op, global_step, summary_op,
                                                                                  model.loss, model.encoder_states,
                                                                                  model.encoder_outputs],
                                                                                 feed_dict=feed_dict)
            summary_writer.add_summary(summaries, step)
        outputs = []

        def eval(batch_x):
            feed_dict = {model.x: batch_x}
            encoder_outputs = sess.run([model.encoder_outputs], feed_dict=feed_dict)
            outputs.append(encoder_outputs[0])
            return outputs

        # Training loop
        batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)
        for batch_x, _ in batches:
            train_step(batch_x)
            step = tf.train.global_step(sess, global_step)
        batches = batch_iter(train_x, train_y, 1, 1)
        for batch_x, _ in batches:
            eval(batch_x)
        saver.save(sess, "checkpoint/model-100epc.ckpt", global_step=step)

        def sum_embedded_words(encode_outputs):
            sent_embedded = []
            for sent in encode_outputs:
                for word in sent:
                    sent_embedded.append(sum(word).tolist())
            return sent_embedded
        embedded_input = sum_embedded_words(outputs)
    return embedded_input
