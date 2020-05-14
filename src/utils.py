import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import operator
from sklearn.model_selection import train_test_split
import numpy as np
import os
import random
import codecs
from functools import reduce
import shutil



# def to_categorical(y,num_classes):
#     return np.eye(num_classes,dtype=np.uint8)[y]

def set_seed(seed_value = 0):
    ''' Set detereministic seed'''
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.compat.v1.set_random_seed(seed_value)

def single_auc_loging(history,title,path_to_save):
    """
    Function for ploting nn-classifier performance. It makes two subplots.
    First subplot with train and val losses
    Second with val auc
    Function saves plot as a picture and as a pkl file

    :param history: history field of history object, witch returned by model.fit()
    :param title: Title for picture (also used as filename)
    :param path_to_save: Path to save file
    :return:
    """
    f, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,12))

    if 'loss' in history.keys():
        loss_key = 'loss'  # for simple NN
    elif 'class_out_loss' in history.keys():
        loss_key = 'class_out_loss'  # for DAL NN
    else:
        raise ValueError('Not found correct key for loss information in history')

    ax1.plot(history[loss_key],label='cl train loss')
    ax1.plot(history['val_%s' %loss_key],label='cl val loss')
    ax1.legend()
    min_loss_index,max_loss_value = min(enumerate(history['val_loss']), key=operator.itemgetter(1))
    ax1.set_title('min_loss_%.3f_epoch%d' % (max_loss_value, min_loss_index))
    ax2.plot(history['val_auc'])
    max_auc_index, max_auc_value = max(enumerate(history['val_auc']), key=operator.itemgetter(1))
    ax2.set_title('max_auc_%.3f_epoch%d' % (max_auc_value, max_auc_index))
    f.suptitle('%s' % (title))
    plt.savefig('%s/%s.png' % (path_to_save,title), figure=f)
    plt.close()
    with open('%s/%s.pkl' % (path_to_save,title), 'wb') as output:
        pickle.dump(history,output,pickle.HIGHEST_PROTOCOL)
        
def prepare_dirs(experiment_res_dir, train_subject):
    '''It creates (or clears, if exists) experiment folder with subjects '''
    path_to_subj = os.path.join(experiment_res_dir, str(train_subject))
    model_path = os.path.join(path_to_subj, 'checkpoints')
    if os.path.isdir(path_to_subj):
        shutil.rmtree(path_to_subj)
    os.makedirs(model_path)
    return path_to_subj

def write_results_table(subjs_test_stats,path_to_exp):
    '''Data dict should contain '''
    header = list(subjs_test_stats[list(subjs_test_stats.keys())[0]].keys())

    with codecs.open('%s/res.txt' %path_to_exp,'w', encoding='utf8') as f:
        f.write(reduce(lambda x,y: x+' {}'.format(y),header,'subj')+'\n')
        tmp_stats = {k:[] for k in header}
        for subj in subjs_test_stats:
            stats = subjs_test_stats[subj]
            [tmp_stats[k].append(stats[k]) for k in header]
            str_to_print = reduce(lambda x,y:x+'{:.2f}'.format(y),[stats[elem] for elem in header],u'{}:'.format(subj))
            f.write('{}\n'.format(str_to_print))

        f.write(reduce(lambda x,y:x+u'{:.{prec}f}±{:.{prec}f}'.format(np.mean(y),np.std(y),prec=2),
                       [tmp_stats[k] for k in header],u'MEAN:'))
        
def separte_last_block(x,y,test_size=0.2):
    x_t, x_nt = x[y == 1], x[y == 0]
    x_t_tr, x_t_tst = train_test_split(x_t, test_size=test_size, shuffle=False)
    x_nt_tr, x_nt_tst = train_test_split(x_nt, test_size=test_size, shuffle=False)
    x_tr = np.concatenate((x_t_tr,x_nt_tr),axis=0)
    y_tr = np.hstack((np.ones(x_t_tr.shape[0]),np.zeros(x_nt_tr.shape[0])))
    x_tst = np.concatenate((x_t_tst,x_nt_tst),axis=0)
    y_tst = np.hstack((np.ones(x_t_tst.shape[0]), np.zeros(x_nt_tst.shape[0])))
    return x_tr,y_tr,x_tst,y_tst










