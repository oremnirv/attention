import csv
import os
import numpy as np
import pandas as pd
import tensorflow as tf


def mkdir(folder):
    """
    """
    if os.path.exists(folder):
        print('Already exists')
        pass
    else:
        os.mkdir(folder)
        print('New folder {}'.format(folder))


def tf_summaries(run, string, train_loss_r, test_loss_r, tr_metric, te_metric, weights, names):
    """

    """
    tf.summary.scalar("training loss run {}".format(run), train_loss_r, step=string)
    tf.summary.scalar("test loss run {}".format(run), test_loss_r, step=string)
    tf.summary.scalar('train metric', tr_metric, step=string)
    tf.summary.scalar('test metric', te_metric, step=string)
    for idx, var in enumerate(weights):
        tf.summary.histogram(names[idx].numpy().decode('utf-8'), var, step=string)


def print_progress(epoch, batch_n, train_loss_r, test_loss_r, tr_metric, te_metric):
    """

    """
    print(
        'Epoch {} batch {} train Loss {:.4f} test Loss {:.4f} with training MSE metric {:.4f} and testing MSE metric '
        '{:.4f}'.format(
            epoch, batch_n,
            train_loss_r, test_loss_r, tr_metric, te_metric))

def tensorboard_embeddings(model, layer_num, meta_data, logdir):
    from tensorboard.plugins import projector
    # Save the weights we want to analyse as a variable. Note that the first
    # value represents any unknown word, which is not in the metadata, so
    # we will remove that value.
    subwords = np.concatenate(([0, 1], np.unique(meta_data)))
    with open(os.path.join(logdir, 'metadata.tsv'), "w") as f:
        for subword in subwords:
            f.write("learnt {}\n".format(subword))
        for unknown in np.arange(0, 2000)[~np.isin(range(0, 2000), subwords)]:
            f.write("unknown #{}\n".format(unknown))
    weights = tf.Variable(model.layers[layer_num].get_weights()[0])
    # Create a checkpoint from embedding, the filename and key are
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(logdir, "embedding.ckpt"))

    # Set up config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(logdir, config)


def write_speci(folder, names, shapes, context_p):
    with open(os.path.expanduser(folder + '_context_' + str(context_p) + '_speci.csv'), "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for name, shape in zip(names, shapes):
            writer.writerow([str(name.numpy()).split('/')[-2], str(shape.numpy())])


def load_spec(path, e, l, context_p, d=False, old=False, abc=False):
    if not os.path.exists(path + '_context_' + str(context_p) + '_speci.csv'):
        print('Does not exists')
        return (e, *l)
    else:
        df = np.array(pd.read_csv(path + '_context_' + str(context_p) + '_speci.csv'))
        ls = []
        if d:
            if old:
                for i in [1, 17, 19, 23, 25]:
                    ls.append(int(df[i][1].split('[')[1].split(']')[0]))
            elif abc:
                for i in [7, 9, 11, 13, 15]:
                    ls.append(int(df[i][1].split('[')[1].split(']')[0]))

            else:
                for i in [1, 11, 13, 15, 17]:
                    ls.append(int(df[i][1].split('[')[1].split(']')[0]))
        else:
            for i in [1, 9, 11, 13, 15]:
                ls.append(int(df[i][1].split('[')[1].split(']')[0]))
        return ls
