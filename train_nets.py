
from losses.face_losses import insightface_loss, cosineface_loss, combine_loss
from utils.data_process import parse_function
from nets.MobileFaceNet import inference
from utils.common import train
from datetime import datetime
import tensorflow as tf
import numpy as np
import argparse
import time
import os
import sys

slim = tf.contrib.slim

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=20, help='epoch to train the network')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--class_number', type=int, required=True,
                        help='class number depend on your training datasets, MS1M-V1: 85164, MS1M-V2: 85742')
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--weight_decay', default=5e-5, help='L2 weight regularization.')
    parser.add_argument('--lr_schedule', help='Number of epochs for learning rate piecewise.', default=[4, 8, 12, 16])
    parser.add_argument('--train_batch_size', default=8, help='batch size to train network')
    parser.add_argument('--tfrecords_file_path', default='./tfrecords/demo', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--summary_path', default='./checkpoint/demo/summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', default='./checkpoint/demo', help='the ckpt file save path')
    parser.add_argument('--log_file_path', default='./checkpoint/demo/logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=250, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--summary_interval', default=400, help='interval to save summary')
    parser.add_argument('--show_info_interval', default=20, help='intervals to save ckpt file')
    parser.add_argument('--pretrained_model', type=str, default='', help='Load a pretrained model before training starts.')
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.999)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', default=True)
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
                        help='Loss based on the norm of the activations in the prelogits layer.', default=2e-5)
    parser.add_argument('--prelogits_norm_p', type=float,
                        help='Norm to use for prelogits norm loss.', default=1.0)
    parser.add_argument('--loss_type', default='cosine', help='loss type, choice type are insightface/cosine/combine')
    parser.add_argument('--margin_s', type=float,
                        help='insightface_loss/cosineface_losses/combine_loss loss scale.', default=30.)
    parser.add_argument('--margin_m', type=float,
                        help='insightface_loss/cosineface_losses/combine_loss loss margin.', default=0.2)
    parser.add_argument('--margin_a', type=float,
                        help='combine_loss loss margin a.', default=1.0)
    parser.add_argument('--margin_b', type=float,
                        help='combine_loss loss margin b.', default=0.2)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    with tf.Graph().as_default():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        args = get_parser()

        # create log dir
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        log_dir = os.path.join(os.path.expanduser(args.log_file_path), subdir)
        if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
            os.makedirs(log_dir)

        # define global parameters
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        epoch = tf.Variable(name='epoch', initial_value=-1, trainable=False)
        # define placeholder
        inputs = tf.placeholder(name='img_inputs', shape=[None, *args.image_size, 1], dtype=tf.float32)
        labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
        phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=None, name='phase_train')

        # prepare train dataset
        # the image is substracted 127.5 and multiplied 1/128.
        # random flip left right
        tfrecords_f = os.path.join(args.tfrecords_file_path,args.tfrecords_file_path.split('/')[-1]+'.record-*')
        tfrecords_f = tf.data.Dataset.list_files(tfrecords_f)
        dataset = tf.data.TFRecordDataset(tfrecords_f)
        dataset = dataset.shuffle(buffer_size=args.buffer_size)
        dataset = dataset.map(parse_function)
        dataset = dataset.batch(args.train_batch_size)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        # pretrained model path
        pretrained_model = None
        if args.pretrained_model:
            pretrained_model = os.path.expanduser(args.pretrained_model)
            print('Pre-trained model: %s' % pretrained_model)

        # identity the input, for inference
        inputs = tf.identity(inputs, 'input')

        prelogits, net_points = inference(inputs, bottleneck_layer_size=args.embedding_size, phase_train=phase_train_placeholder, weight_decay=args.weight_decay)


        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Norm for the prelogits
        eps = 1e-5
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=args.prelogits_norm_p, axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * args.prelogits_norm_loss_factor)

        # inference_loss, logit = cos_loss(prelogits, labels, args.class_number)
        w_init_method = slim.initializers.xavier_initializer()
        if args.loss_type == 'insightface':
            inference_loss, logit = insightface_loss(embeddings, labels, args.class_number, w_init_method)
        elif args.loss_type == 'cosine':
            inference_loss, logit = cosineface_loss(embeddings, labels, args.class_number, w_init_method)
        elif args.loss_type == 'combine':
            inference_loss, logit = combine_loss(embeddings, labels, args.train_batch_size, args.class_number, w_init_method)
        else:
            assert 0, 'loss type error, choice item just one of [insightface, cosine, combine], please check!'
        tf.add_to_collection('losses', inference_loss)

        # total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([inference_loss] + regularization_losses, name='total_loss')

        # define the learning rate schedule
        learning_rate = tf.train.piecewise_constant(epoch, boundaries=args.lr_schedule, values=[0.1, 0.01, 0.001, 0.0001, 0.00001],
                                         name='lr_schedule')
        
        # define sess
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping, gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # calculate accuracy
        pred = tf.nn.softmax(logit)
        correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.cast(labels, tf.int64)), tf.float32)
        Accuracy_Op = tf.reduce_mean(correct_prediction)

        # summary writer
        summary = tf.summary.FileWriter(args.summary_path, sess.graph)
        summaries = []
        # add train info to tensorboard summary
        summaries.append(tf.summary.scalar('inference_loss', inference_loss))
        summaries.append(tf.summary.scalar('total_loss', total_loss))
        summaries.append(tf.summary.scalar('leraning_rate', learning_rate))
        summaries.append(tf.summary.histogram('img_inputs', inputs))
        summaries.append(tf.summary.histogram('img_labels', labels))
        tf.print("img_labels:", labels, output_stream=sys.stdout)


        # train op
        train_op = train(total_loss, global_step, args.optimizer, learning_rate, args.moving_average_decay,
                         tf.global_variables(), summaries, args.log_histograms)
        inc_global_step_op = tf.assign_add(global_step, 1, name='increment_global_step')
        inc_epoch_op = tf.assign_add(epoch, 1, name='increment_epoch')
        summary_op = tf.summary.merge(summaries)


        # saver to load pretrained model or save model
        # MobileFaceNet_vars = [v for v in tf.trainable_variables() if v.name.startswith('MobileFaceNet')]
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.saver_maxkeep)

        # init all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # load pretrained model
        if pretrained_model:
            print('Restoring pretrained model: %s' % pretrained_model)
            ckpt = tf.train.get_checkpoint_state(pretrained_model)
            print(ckpt)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # output file path
        if not os.path.exists(args.log_file_path):
            os.makedirs(args.log_file_path)

        count = 0
        acc_val = 0.0
        besti = 0
        best_acc = 0
        save_cnt_step = 0
        bestname = ""
        for i in range(args.max_epoch):
            sess.run(iterator.initializer)
            _ = sess.run(inc_epoch_op)
            count_step = 0
            avg_acc = 0
            sav_avg_acc = 0
            while True:
                try:
                    images_train, labels_train = sess.run(next_element)

                    feed_dict = {inputs: images_train, labels: labels_train, phase_train_placeholder: True}
                    start = time.time()
                    _, total_loss_val, inference_loss_val, reg_loss_val, _, acc_val = \
                    sess.run([train_op, total_loss, inference_loss, regularization_losses, inc_global_step_op, Accuracy_Op],
                             feed_dict=feed_dict)
                    end = time.time()
                    pre_sec = args.train_batch_size/(end - start)

                    count += 1
                    count_step += 1
                    avg_acc += acc_val
                    if i == 0:
                        save_cnt_step += 1                    
                    # print training information
                    if count > 0 and count % args.show_info_interval == 0:
                        print('epoch %d, total_step %d, total loss is %.2f , inference loss is %.2f, reg_loss is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
                              (i, count, total_loss_val, inference_loss_val, np.sum(reg_loss_val), acc_val, pre_sec))

                    # save summary
                    if count > 0 and count_step == save_cnt_step and i != 0:
                        feed_dict = {inputs: images_train, labels: labels_train, phase_train_placeholder: True}
                        summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                        summary.add_summary(summary_op_val, count)

                    # save ckpt files
                    if count > 0 and count_step == save_cnt_step and i != 0:
                        sav_avg_acc = avg_acc/count_step
                        filename = 'MobileFaceNet_epoch_%04d_acc_%.4f'%(i,sav_avg_acc) + '.ckpt'
                        filename = os.path.join(args.ckpt_path, filename)
                        saver.save(sess, filename)
                        if sav_avg_acc > best_acc:
                            best_acc = sav_avg_acc
                            besti = i
                            bestname = filename

                except tf.errors.OutOfRangeError:
                    if i == 0:
                        save_cnt_step -= 1 #ignore the last step in an epoch                
                    print("End of epoch %d, avg_trainacc: %.4f" % (i,sav_avg_acc))
                    print("best epoch: %d, best_avg_trainacc: %.4f"%(besti,best_acc))
                    print("best saved ckpt: %s"%(os.path.basename(bestname)))
                    break
                    
