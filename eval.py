import os
import tensorflow as tf
from data_load import get_batch
from model import Transformer
from hparams import Hparams
from utils import get_hypotheses, calc_bleu, postprocess, save_hparams, acc
import logging

logging.basicConfig(level=logging.INFO)
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)

logging.info("# Prepare eval batches")
eval_batches, num_eval_batches, num_eval_samples  = get_batch(hp.eval1, hp.eval2,
                                              100000, 100000,
                                              hp.vocab, hp.batch_size,
                                              shuffle=False)
iter = tf.data.Iterator.from_structure(eval_batches.output_types, eval_batches.output_shapes)
xs, ys = iter.get_next()
decoder_inputs, y, y_seqlen, sents2 = ys
eval_init_op = iter.make_initializer(eval_batches)

logging.info("# Load model")
m = Transformer(hp)
y_mask = m.y_masks(y)
y_hat, eval_summaries = m.eval(xs, ys, y_mask)
saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)    
    saver.restore(sess, ckpt)
    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)
    sess.run(eval_init_op)
    _y_hat, _y = sess.run([y_hat, y])
    print(_y_hat)
    print(_y)
    print(acc(_y_hat, _y))
    #hypotheses = get_hypotheses(1, 128, sess, y_hat, m.idx2token)

#print(hypotheses)