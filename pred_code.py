import cPickle
import time
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import copy
import os
warnings.filterwarnings("ignore")   
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"      
def Iden(x):
    y = x
    return(y)

def predict_conv_net(datasets,
                   U,
                   img_w=300, 
                   filter_hs=[3,4,5],
                   hidden_units=[100,2], 
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=20, 
                   batch_size=50, 
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   use_valid_set=True,
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True):
    from conv_net_classes import *
    rng = np.random.RandomState(3435)
    # data[0] is training and data[1] is testing
    img_h = len(datasets[0][0])-1  
    # m length of word embedding
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    # prepare the feature maps here
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, 1, filter_w*filter_h))
        pool_sizes.append((img_h-1+1, img_w-filter_w+1))

    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    
    print parameters
    #define model architecture
    index = T.lscalar()

    x = T.tensor3('x',dtype='float32') 
    y = T.ivector('y')
    U = np.array(U, dtype='float32')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector(dtype='float32')
    zero_vec = np.zeros(img_w,dtype='float32')
    #### initialization
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))],allow_input_downcast=True)
    
    ########################### tree CNN start
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]*x.shape[-1]))[:,:,:,0:img_w*filter_hs[i]]
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w*filter_hs[i]),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
                               
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    ########################### tree CNN end
    
    ########################### sibling start
    sib = [range(1500,1800),range(1800,2100),range(2100,2400),range(2400,2700),range(2700,3000)]
    sib_templet = [sib[0]+sib[2],sib[0]+sib[1]+sib[2],sib[0]+sib[2]+sib[3],sib[0]+sib[1]+sib[2]+sib[3],sib[0]+sib[1]+sib[2]+sib[4]]    
    filter_shapes_sib = []
    pool_sizes_sib = []    
    filter_sib = [2,3,3,4,4]
    for filter_h in filter_sib:
        filter_shapes_sib.append((feature_maps, 1, 1, filter_w*filter_h))
        pool_sizes_sib.append((img_h-1+1, img_w-filter_w+1))    
        
    for i in xrange(len(filter_sib)):
        layer0_input_sib = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]*x.shape[-1]))[:,:,:,sib_templet[i]]
        filter_shape_sib = filter_shapes_sib[i]
        pool_size_sib = pool_sizes_sib[i]
        conv_layer_sib = LeNetConvPoolLayer(rng, input=layer0_input_sib,image_shape=(batch_size, 1, img_h, img_w*filter_sib[i]),
                                        filter_shape=filter_shape_sib, poolsize=pool_size_sib, non_linear=conv_non_linear) 
        
        layer1_input_sib = conv_layer_sib.output.flatten(2)
        conv_layers.append(conv_layer_sib)
        layer1_inputs.append(layer1_input_sib)    
    ########################### sibling end
    f = file("conv_layer_state.pkl", 'rb')
    conv_layers_params = cPickle.load(f)
    for i, conv_layer in enumerate(conv_layers):
    	conv_layer.__setstate__(conv_layers_params[i])
    f.close()    

    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*(len(filter_hs)+len(filter_sib)) ##m.m.   
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    
    f = file('classifier_state.pkl', 'rb')
    classifier.__setstate__(cPickle.load(f))
    f.close()    
    
    test_set_x = datasets[0][:,:img_h]
    test_pred_layers = []
    test_size = test_set_x.shape[0]

    for i, conv_layer in enumerate(conv_layers):
        if i < len(filter_hs):
            test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]*x.shape[-1]))[:,:,:,0:img_w*filter_hs[i]]
        else:
            test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]*x.shape[-1]))[:,:,:,sib_templet[i-len(filter_hs)]] 
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
        
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    test_prediction = theano.function([x], test_y_pred,allow_input_downcast=True)   
    
    print '*********** PREDICTING ***********'

    predictions = test_prediction(test_set_x[:])
       
    return predictions
             
def get_idx_for_tensor (sent, word_idx_map, max_l, k, filter_h):
    each_sent = copy.deepcopy(sent)
    for j, each_word in enumerate(each_sent[:-1]):
        for l, each_field in enumerate(each_word):
            if each_field in word_idx_map:
                each_sent[j][l] = word_idx_map[each_field]
            elif each_field == 0:
                continue
            else:
                print each_field
                
    return each_sent

def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            print word
    while len(x) < max_l+2*pad:
        x.append(0)
    return x


def make_idx_data(revs, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test, dev = [], [], []
    train_tensor, test_tensor, dev_tensor =[], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        sent.append(rev["y"])
        sent_tensor = get_idx_for_tensor(rev["tree"], word_idx_map, max_l, k, filter_h)
        #remove the following line to take seq cnn into consideration
        #sent_tensor = merge_sent_and_tree(sent, sent_tensor)
        if rev["split"]==2:            
            test.append(sent)
            test_tensor.append(sent_tensor)
        elif rev["split"]==1:
            train.append(sent)
            train_tensor.append(sent_tensor)
        elif rev["split"]==3:
            dev.append(sent)
            dev_tensor.append(sent_tensor)        
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    dev = np.array(dev,dtype="int")
    train_tensor = np.array(train_tensor,dtype="int")
    test_tensor = np.array(test_tensor,dtype="int")  
    dev_tensor = np.array(dev_tensor,dtype="int")
    return [train,test], [train_tensor,test_tensor]  
  
   
#if __name__=="__main__":
def main():
    import cPickle
    import time
    import numpy as np
    from collections import defaultdict, OrderedDict
    import theano
    import theano.tensor as T
    import re
    import warnings
    import sys
    import copy
    warnings.filterwarnings("ignore")
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
    print "loading data...",
    x = cPickle.load(open("TREC/new_Query_sib.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    word_idx_map["ROOT"] = 0
    # revs = ["y"","text,"split","num_words"]
    print "data loaded!"
    #input = sys.argv[1]
    mode = "-nonstatic"
    word_vectors = "-word2vec"
    #word_vectors = sys.argv[2]
    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    from conv_net_classes import *
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W
    results = []
    datasets, datasets_tensor = make_idx_data(revs, word_idx_map, max_l=56,k=300, filter_h=5)
    prediction = predict_conv_net(datasets_tensor,
                              U,
                              lr_decay=0.95,
                              filter_hs=[3,4,5],
                              conv_non_linear="relu",
                              hidden_units=[100,6], 
                              use_valid_set=True, 
                              shuffle_batch=True, 
                              n_epochs=20, 
                              sqr_norm_lim=9,
                              non_static=non_static,
                              batch_size=128,
                              dropout_rate=[0.5])
    if prediction[0] == 0:
        Answer_type = "ABBREVIATION"
    elif prediction[0] == 1:
        Answer_type = "ENTITY"
    elif prediction[0] == 2:
        Answer_type = "DESCRIPTION"
    elif prediction[0] == 3:
        Answer_type = "HUMAN"
    elif prediction[0] == 4:
        Answer_type = "LOCATION"
    elif prediction[0] == 5:
        Answer_type = "NUMERIC"

    print Answer_type
    return Answer_type

if __name__ == "__main__":
    main()
