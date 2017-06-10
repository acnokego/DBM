import tensorflow as tf
import numpy as np
import os 
import utils
import config
import itertools

class DBM(object):
    
    def __init__(self, num_visible=2, num_hidden=8, num_layers=2, CD_steps=1, gibb_steps=100, lr=0.001, 
                 batch_size=100, num_epochs=50, stddev=0.1, verbose=1, main_dir='dbm'):

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.CD_steps = CD_steps
        self.gibb_steps = gibb_steps
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.stddev = stddev
        self.verbose = verbose

        ### first RBM
        self.W_1 = None
        self.bv_1 = None
        self.bh_1 = None
        self.hidden_samples = None
        self.vrand = None
        self.hrand = None

        ### last RBM
        self.hidden_data = None
        self.W_2 = None
        self.bv_2 = None
        self.bh_2 = None
        self.hrand1 = None
        self.hrand2 = None

        ### compose
        self.input_data_compose = None
        self.hidden2 = None
        self.test_samples = None
        
        ###
        self.reconstruction = None

        self.main_dir = main_dir
        self.models_dir, self.data_dir, self.summary_dir = self._create_data_directories()
        self.tf_merged_summaries = None
        self.tf_summary_writer = None
        self.tf_session = None
        self.tf_saver = None

    def fit(self, train_set, validation_set=None, restore_previous_model=False):

        """ Fit the model to the training data.

        :param train_set: training set
        :param validation_set: validation set. optional, default None
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.

        :return: self
        """

        if validation_set is not None:
            self.validation_size = validation_set.shape[0]

        self._build_first_RBM()
        self._build_last_RBM()

        with tf.Session() as self.tf_session:

            self._initialize_tf_utilities_and_ops(restore_previous_model)
            self._train_first_RBM(train_set, validation_set)

            hidden_data = self.tf_session.run(self.hidden_samples, feed_dict = 
                                              self._create_feed_dict(train_set, True))
            val_hidden_data = self.tf_session.run(self.hidden_samples, feed_dict = 
                                              self._create_feed_dict(validation_set, True))

            self._train_last_RBM(hidden_data, val_hidden_data)
            self._compose()
            self._train_DBM(train_set, hidden_data, validation_set, val_hidden_data)
            self._plot_likelihood()
            self.reconstruction = self._run_sample()
            print("ok")

            #self.tf_saver.save(self.tf_session, self.model_path)

    def _initialize_tf_utilities_and_ops(self, restore_previous_model):

        """ Initialize TensorFlow operations: summaries, init operations, saver, summary_writer.
        Restore a previously trained model if the flag restore_previous_model is true.
        """

        self.tf_merged_summaries = tf.summary.merge_all()
        init_op = tf.initialize_all_variables()
        self.tf_saver = tf.train.Saver()

        self.tf_session.run(init_op)

        if restore_previous_model:
            self.tf_saver.restore(self.tf_session, self.model_path)

        self.tf_summary_writer = tf.summary.FileWriter(self.summary_dir, self.tf_session.graph_def)

    def _run_train_step(self, train_set, validation_set, pos='first'):

        """ Run a training step. A training step is made by randomly shuffling the training set,
        divide into batches and run the variable update nodes for each batch.

        :param train_set: training set

        :return: self
        """

        np.random.shuffle(train_set)

        batches = [_ for _ in utils.gen_batches(train_set, self.batch_size)]
        if pos == 'first':
            updates = [self.w_upd8_1, self.bh_upd8_1, self.bv_upd8_1, self.last_s_1]
        else:
            updates = [self.w_upd8_2, self.bh_upd8_2, self.bv_upd8_2, self.last_s_2]

        
        i = 0 
        start = True

        for batch in batches:
            upd = self.tf_session.run(updates, feed_dict=self._create_feed_dict(batch, start, pos))
            ### feed to next batch
            self.last_s = upd[3]
            start = False


    def _run_train_step_DBM(self, train_set, hidden_data):

        """ Run a training step. A training step is made by randomly shuffling the training set,
        divide into batches and run the variable update nodes for each batch.

        :param train_set: training set

        :return: self
        """

        np.random.shuffle(train_set)

        batches = [_ for _ in utils.gen_batches(train_set, self.batch_size)]
        batches_2 = [_ for _ in utils.gen_batches(hidden_data, self.batch_size)]

        
        updates = [self.w_1_upd8_3, self.w_2_upd8_3, self.bh_1_upd8_3, self.bh_2_upd8_3,
                       self.bv_1_upd8_3, self.bv_2_upd8_3]

        for batch_1, batch_2 in zip(batches, batches_2):
            upd = self.tf_session.run(updates, feed_dict=self._create_feed_dict(batch_1, True, pos='dbm', status='train', hidden_data=batch_2))

    def _run_validation_error_and_summaries(self, epoch, validation_set, pos, valid_hidden=None):

        """ Run the summaries and error computation on the validation set.

        :param epoch: current epoch
        :param validation_set: validation data

        :return: self
        """
        loss = self.loss_function_1
        if pos == 'last':            
            loss = self.loss_function_2
        elif pos == 'dbm':
            loss = self.loss_function_3
        if pos == 'dbm':
            result = self.tf_session.run( loss,
                    feed_dict=self._create_feed_dict(validation_set, True, pos, hidden_data=valid_hidden))
        else:
            result = self.tf_session.run( loss,
                    feed_dict=self._create_feed_dict(validation_set, True, pos))

        #summary_str = result[0]
        err = result
       # grad = result[2]

        #self.tf_summary_writer.add_summary(summary_str, 1)

        if self.verbose == 1:
            print("Validation cost at step %s: %s" % (epoch, err))
           # print("Validation grad at step %s: %s" % (epoch, grad))
            
    def _train_first_RBM(self, train_set, validation_set):

        """ Train the model.

        :param train_set: training set
        :param validation_set: validation set. optional, default None

        :return: self
        """
        print("Training first....")
        for i in range(self.num_epochs):
            self._run_train_step(train_set, validation_set)
           # self._plot_likelihood(i) 
            if validation_set is not None:
                self._run_validation_error_and_summaries(i, validation_set, 'first')

    def _train_last_RBM(self, train_set, validation_set):

        """ Train the model.

        :param train_set: training set
        :param validation_set: validation set. optional, default None

        :return: self
        """
        print("Training last...")
        for i in range(self.num_epochs):
            self._run_train_step(train_set, validation_set, pos='last')
            #self._plot_likelihood(i) 
            if validation_set is not None:
                self._run_validation_error_and_summaries(i, validation_set, 'last')
    def _train_DBM(self, train_set, hidden_data,validation_set, valid_hidden):

        """ Train the model.

        :param train_set: training set
        :param validation_set: validation set. optional, default None

        :return: self
        """
        print("Training DBM...")
        for i in range(self.num_epochs):
            self._run_train_step_DBM(train_set, hidden_data)
            #self._plot_likelihood(i) 
            if validation_set is not None:
                self._run_validation_error_and_summaries(i, validation_set, 'dbm', valid_hidden=valid_hidden)

    def _compose(self):

        self.input_data_compose, self.hidden2, self.hrand1, self.hrand2, self.vrand = self._create_placeholders_compose()

        h1probs, h1states = self.sample_hidden_from_visible(self.input_data_compose, status='test', hidden=self.hidden2)
        vprobs, h2states = self.sample_visible_from_hidden(h1states, status='test')

        positive_1 = self.compute_positive_association(self.input_data_compose, h1probs, h1states)
        positive_2 = self.compute_positive_association(self.hidden2, h1probs, h1states, 'last')

        for step in range(self.gibb_steps-1):
            h1probs_n, h1states = self.sample_hidden_from_visible(vprobs, status='test',hidden=h2states)
            vprobs, h2states = self.sample_visible_from_hidden(h1states, status='test')
            if step == (self.CD_steps)-1: 
                negative_1 = tf.matmul(tf.transpose(vprobs), h1probs)
                negative_2 = tf.matmul(tf.transpose(h2states), h1probs)
                bh_u = h1probs - h1probs_n
                bv_1_u = self.input_data_compose- vprobs
                bv_2_u = self.hidden2 - h2states

        self.test_samples = vprobs
        self.w_1_upd8_3 = self.W_1.assign_add(self.lr * (positive_1 - negative_1))
        self.w_2_upd8_3 = self.W_2.assign_add(self.lr * (positive_2 - negative_2))
        self.bh_1_upd8_3 = self.bh_1.assign_add(self.lr * tf.reduce_mean(bh_u, 0))
        self.bh_2_upd8_3 = self.bh_2.assign_add(self.lr * tf.reduce_mean(bh_u, 0))
        self.bv_1_upd8_3 = self.bv_1.assign_add(self.lr * tf.reduce_mean(bv_1_u, 0))
        self.bv_2_upd8_3 = self.bv_2.assign_add(self.lr * tf.reduce_mean(bv_2_u, 0))

        self.loss_function_3 = tf.sqrt(tf.reduce_mean(tf.square(bv_1_u)))
        _ = tf.summary.scalar("cost_3", self.loss_function_3)


    def _build_first_RBM(self):

        """ Build the Restricted Boltzmann Machine model in TensorFlow.

        :return: self
        """

        self.input_data, self.hrand, self.vrand, self.last_state= self._create_placeholders()
        self.W_1, self.bh_1, self.bv_1 = self._create_variables()
        hprobs0, hstates0, vprobs, hprobs1, hstates1 = self.gibbs_sampling_step(self.input_data)
        positive = self.compute_positive_association(self.input_data, hprobs0, hstates0)

        #negative phases (PCD)

        hprobs0_n, hstates0_n, vprobs_n, hprobs1_n, hstates1_n = self.gibbs_sampling_step(self.last_state)

        nn_input = vprobs_n

        ## for sample
        for step in range(self.gibb_steps - 1):
            ## for CD
            if step < (self.CD_steps-1):
                hprobs_n, hstates_n, vprobs_n, hprobs1_n, hstates1_n = self.gibbs_sampling_step(nn_input)
                nn_input = vprobs_n
            else :
                # for reconstruct
                hprobs_2, hstates_2, vprobs_2, hprobs1_2, hstates1_2 = self.gibbs_sampling_step(nn_input)
                nn_input = vprobs_2
        
        self.hidden_samples = hstates0
        ### for recon
        #self.samples = nn_input
        ###for PCD
        self.last_s_1 = vprobs_n

        negative = tf.matmul(tf.transpose(vprobs_n), hprobs1_n)

        self.encode = hprobs1_n  # encoded data, used by the transform method

        self.w_upd8_1 = self.W_1.assign_add(self.lr * (positive - negative))
        self.bh_upd8_1 = self.bh_1.assign_add(self.lr * tf.reduce_mean(hprobs0 - hprobs1_n, 0))
        self.bv_upd8_1 = self.bv_1.assign_add(self.lr * tf.reduce_mean(self.input_data - vprobs_n, 0))

        self.loss_function_1 = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - vprobs_n)))
        #self.loss_function = tf.reduce_sum(tf.square(self.input_data - vprobs))
        #self.loss_function = self.learning_rate*(positive-negative)
        #self.grad = self.learning_rate * (positive-negative)
        _ = tf.summary.scalar("cost_1", self.loss_function_1)



    def _build_last_RBM(self):

        """ Build the Restricted Boltzmann Machine model in TensorFlow.

        :return: self
        """

        self.hidden_data, self.hrand1, self.hrand2, self.last_state_2= self._create_placeholders(pos='last')
        self.W_2, self.bh_2, self.bv_2 = self._create_variables(pos='last')
        hprobs0, hstates0, vprobs, hprobs1, hstates1 = self.gibbs_sampling_step(self.hidden_data, 'last')
        positive = self.compute_positive_association(self.hidden_data, hprobs0, hstates0, 'last')

        #negative phases (PCD)

        hprobs0_n, hstates0_n, vprobs_n, hprobs1_n, hstates1_n = self.gibbs_sampling_step(self.last_state_2, 'last')

        nn_input = vprobs_n

        ## for sample
        for step in range(self.gibb_steps - 1):
            ##for CD
            if step < (self.CD_steps-1):
                hprobs_n, hstates_n, vprobs_n, hprobs1_n, hstates1_n = self.gibbs_sampling_step(nn_input, 'last')
                nn_input = vprobs_n
            else :
                # for reconstruct
                hprobs_2, hstates_2, vprobs_2, hprobs1_2, hstates1_2 = self.gibbs_sampling_step(nn_input, 'last')
                nn_input = vprobs_2
        
        ## for recon
        self.samples = nn_input
        ### for PCD
        self.last_s_2 = vprobs_n

        negative = tf.matmul(tf.transpose(vprobs_n), hprobs1_n)

        self.encode = hprobs1_n  # encoded data, used by the transform method

    
        self.w_upd8_2 = self.W_2.assign_add(self.lr * (positive - negative))
        self.bh_upd8_2 = self.bh_2.assign_add(self.lr * tf.reduce_mean(hprobs0 - hprobs1_n, 0))
        self.bv_upd8_2 = self.bv_2.assign_add(self.lr * tf.reduce_mean(self.hidden_data - vprobs_n, 0))
    

        self.loss_function_2 = tf.sqrt(tf.reduce_mean(tf.square(self.hidden_data - vprobs_n)))
        #self.loss_function = tf.reduce_sum(tf.square(self.input_data - vprobs))
        #self.loss_function = self.learning_rate*(positive-negative)
        #self.grad = self.learning_rate * (positive-negative)
        _ = tf.summary.scalar("cost_2", self.loss_function_2)

    def _run_sample(self):

        init_samples = np.random.rand(8000, self.num_visible)
        init_hidden2 = np.random.randint(2, size=(8000, self.num_hidden))

        test_samples = self.tf_session.run(self.test_samples, 
                        feed_dict = self._create_feed_dict(init_samples, True, status='test', hidden_data = init_hidden2))

        return test_samples


    def _create_placeholders(self, pos='first'):

        """ Create the TensorFlow placeholders for the model.

        :return: tuple(input(shape(None, num_visible)),
                       hrand(shape(None, num_hidden))
                       vrand(shape(None, num_visible)))
        """
        name_x = 'x-input'
        name_h = 'hrand'
        name_v = 'vrand'
        name_last = 'last-input'
        if pos != 'first':
            num_v = self.num_hidden
            num_h = self.num_hidden
            name_x = 'x-input2'
            name_h = 'hrand2'
            name_v = 'vrand2'
            name_last = 'last-input2'
        else :
            num_v = self.num_visible
            num_h = self.num_hidden
        
        x = tf.placeholder('float', [None, num_v], name=name_x)
        hrand = tf.placeholder('float', [None, num_h], name=name_h)
        vrand = tf.placeholder('float', [None, num_v], name=name_v)
        last = tf.placeholder('float', [None, num_v], name=name_last)

        return x, hrand, vrand, last
    def _create_placeholders_compose (self):

        """ Create the TensorFlow placeholders for the model.

        :return: tuple(input(shape(None, num_visible)),
                       hrand(shape(None, num_hidden))
                       vrand(shape(None, num_visible)))
        """
        x = tf.placeholder('float', [None, self.num_visible], name='x-input_c')
        #h1 = tf.placeholder('float', [None, self.num_hidden], name='h1-input')
        h2 = tf.placeholder('float', [None, self.num_hidden], name='h2-input')
        h1rand = tf.placeholder('float', [None, self.num_hidden], name='hrand1')
        h2rand = tf.placeholder('float', [None, self.num_hidden], name='hrand2')
        vrand = tf.placeholder('float', [None, self.num_visible], name='vrand')
       # last = tf.placeholder('float', [None, num_visible], name='last-input')

        return x, h2, h1rand, h2rand, vrand
###
    def _create_variables(self, pos='first' ):

        """ Create the TensorFlow variables for the model.

        :return: tuple(weights(shape(num_visible, num_hidden),
                       hidden bias(shape(num_hidden)),
                       visible bias(shape(num_visible)))
        """
        if pos != 'first':
            num_v = self.num_hidden
            num_h = self.num_hidden
        else :
            num_v = self.num_visible
            num_h = self.num_hidden

        #c = tf.constant([0.0, 0.0])
        W = tf.Variable(tf.random_normal((num_v, num_h), mean=0.0, stddev=0.01), name='weights')
        bh_ = tf.Variable(tf.zeros([num_h]), name='hidden-bias')
        #bh_ = tf.Variable(c, name='hidden-bias')
        bv_ = tf.Variable(tf.zeros([num_v]), name='visible-bias')
        #bv_ = tf.Variable(c, name='visible-bias')
        
        '''
        dW = tf.Variable(tf.random_normal((num_v, num_h), mean=0.0, stddev=0.01), name='weights_delta')
        dbh_ = tf.Variable(tf.zeros([num_h]), name='hidden-bias_delta')
        dbv_ = tf.Variable(tf.zeros([num_v]), name='visible-bias_delta')
        '''

        return W, bh_, bv_

    def gibbs_sampling_step(self, visible, pos='first'):

        """ Performs one step of gibbs sampling.

        :param visible: activations of the visible units

        :return: tuple(hidden probs, hidden states, visible probs,
                       new hidden probs, new hidden states)
        """
        

        hprobs, hstates = self.sample_hidden_from_visible(visible, pos)
        vprobs = self.sample_visible_from_hidden(hprobs, pos)
        hprobs1, hstates1 = self.sample_hidden_from_visible(vprobs, pos)

        return hprobs, hstates, vprobs, hprobs1, hstates1
    def sample_hidden_from_visible(self, visible, pos='first', status='train', hidden= None):

        """ Sample the hidden units from the visible units.
        This is the Positive phase of the Contrastive Divergence algorithm.

        :param visible: activations of the visible units

        :return: tuple(hidden probabilities, hidden binary states)
        """
        if status == 'train':
            if pos == 'first':
                transform_activation = 2 * (tf.matmul(visible, self.W_1) + self.bh_1)
                hprobs = tf.nn.sigmoid(transform_activation)
                hstates = utils.sample_prob(hprobs, self.hrand)
            else:
                transform_activation = (tf.matmul(visible, self.W_2) + self.bh_2)
                hprobs = tf.nn.sigmoid(transform_activation)
                hstates = utils.sample_prob(hprobs, self.hrand2)

        else:

            transform_activation = (tf.matmul(visible, self.W_1) + self.bh_1) + tf.matmul(hidden, self.W_2) + self.bv_2
            hprobs = tf.nn.sigmoid(transform_activation)
            hstates = utils.sample_prob(hprobs, self.hrand1)


        return hprobs, hstates


    def sample_visible_from_hidden(self, hidden, pos='first', status='train'):

        """ Sample the visible units from the hidden units.
        This is the Negative phase of the Contrastive Divergence algorithm.

        :param hidden: activations of the hidden units

        :return: visible probabilities
        """

 #       visible_activation = tf.matmul(hidden, tf.transpose(self.W)) + self.bv_
        '''
        if self.visible_unit_type == 'bin':
            vprobs = tf.nn.sigmoid(visible_activation)

        elif self.visible_unit_type == 'gauss':
            vprobs = tf.truncated_normal((1, self.num_visible), mean=visible_activation, stddev=self.stddev)

        else:
            vprobs = None
        '''
        if status == 'train':
            if pos == 'first' :
                visible_activation = tf.matmul(hidden, tf.transpose(self.W_1)) + self.bv_1
                vprobs = tf.truncated_normal((1, self.num_visible), mean=visible_activation, stddev=self.stddev)
            else:
                visible_activation =2*(tf.matmul(hidden, tf.transpose(self.W_2)) + self.bv_2)
                vprobs = tf.nn.sigmoid(visible_activation)
                vprobs = utils.sample_prob(vprobs, self.hrand1)
        else :
                visible_activation = tf.matmul(hidden, tf.transpose(self.W_1)) + self.bv_1
                vprobs = tf.truncated_normal((1, self.num_visible), mean=visible_activation, stddev=self.stddev)
                hidden_activation = (tf.matmul(hidden, tf.transpose(self.W_2)) + self.bv_2)
                h2probs = tf.nn.sigmoid(hidden_activation)
                h2states = utils.sample_prob(h2probs, self.hrand2)
                
                return vprobs, h2states

        return vprobs


##TODO 

    def compute_positive_association(self, visible, hidden_probs, hidden_states, pos= 'first'):

        """ Compute positive associations between visible and hidden units.

        :param visible: visible units
        result = self.tf_session.run([self.tf_merged_summaries, self.loss_function],
                                     feed_dict=self._create_feed_dict(validation_set))

        summary_str = result[0]
        err = result[1]

        :param hidden_probs: hidden units probabilities
        :param hidden_states: hidden units states

        :return: positive association = dot(visible.T, hidden)
        """
        '''
        if self.visible_unit_type == 'bin':
            positive = tf.matmul(tf.transpose(visible), hidden_states)

        elif self.visible_unit_type == 'gauss':
            positive = tf.matmul(tf.transpose(visible), hidden_probs)

        else:
            positive = None
        '''
        if pos != 'first':
            positive = tf.matmul(tf.transpose(visible), hidden_states)

        else:
            positive = tf.matmul(tf.transpose(visible), hidden_probs)

        return positive

    def _create_feed_dict(self, data, start, pos= 'first', status='train', hidden_data=None):

        """ Create the dictionary of data to feed to TensorFlow's session during training.

        :param data: training/validation set batch
        
        :param start: whether it is the start of a epoch 

        :return: dictionary(self.input_data: data, self.hrand: random_uniform, self.vrand: random_uniform)
        """
        if status == 'train':
            if pos == 'first':
                if start :
                    return {
                        self.input_data: data,
                        self.last_state: data,
                        self.hrand: np.random.rand(data.shape[0], self.num_hidden),
                        self.vrand: np.random.rand(data.shape[0], self.num_visible)
                    }
                else:
                    return {
                        self.input_data: data,
                        self.last_state: self.last_s,
                        self.hrand: np.random.rand(data.shape[0], self.num_hidden),
                        self.vrand: np.random.rand(data.shape[0], self.num_visible)
                    }
            elif pos == 'last':

                if start :
                    return {
                        self.hidden_data: data,
                        self.last_state_2: data,
                        self.hrand1: np.random.rand(data.shape[0], self.num_hidden),
                        self.hrand2: np.random.rand(data.shape[0], self.num_hidden)
                    }
                else:
                    return {
                        self.hidden_data: data,
                        self.last_state_2: self.last_s,
                        self.hrand1: np.random.rand(data.shape[0], self.num_hidden),
                        self.hrand2: np.random.rand(data.shape[0], self.num_hidden)
                    }
            elif pos == 'dbm' :
                return {
                    self.input_data_compose: data,
                    self.hidden2: hidden_data,
                    self.hrand1: np.random.rand(data.shape[0], self.num_hidden),
                    self.hrand2: np.random.rand(data.shape[0], self.num_hidden),
                    self.vrand: np.random.rand(data.shape[0], self.num_visible)
                }
                    

        else:
            return {
                self.input_data_compose: data,
                self.hidden2: hidden_data,
                self.hrand1: np.random.rand(data.shape[0], self.num_hidden),
                self.hrand2: np.random.rand(data.shape[0], self.num_hidden),
                self.vrand: np.random.rand(data.shape[0], self.num_visible)

            }
    def _create_data_directories(self):

        """ Create the three directories for storing respectively the models,
        the data generated by training and the TensorFlow's summaries.

        :return: tuple of strings(models_dir, data_dir, summary_dir)
        """

        self.main_dir = self.main_dir + '/' if self.main_dir[-1] != '/' else self.main_dir

        models_dir = config.models_dir + self.main_dir
        data_dir = config.data_dir + self.main_dir
        summary_dir = config.summary_dir + self.main_dir

        for d in [models_dir, data_dir, summary_dir]:
            if not os.path.isdir(d):
                os.mkdir(d)

        return models_dir, data_dir, summary_dir

    def _plot_likelihood(self):

        w_1, w_2 = self.tf_session.run([self.W_1, self.W_2])

        '''
        w (2,8)
        '''

        #print(w)
        #print(b_h)
        #print(b_v)

        h_values_1 = np.array( list(map(list, (itertools.product([0,1], repeat=self.num_hidden)))) ) 
        h_values_2 = np.array( list(map(list, (itertools.product([0,1], repeat=self.num_hidden)))) ) 
        #v = np.array(list( map(list, (itertools.product(np.arange(-15,15,0.1), repeat=self.num_visible)))))
        y = np.arange(-15,15,0.1)
        x = np.arange(-15,15, 0.1)
        v = np.array(list( map(list, (itertools.product(x,y)))))
        # energy function = a-b-c
        #print(v)

        likelihood = np.zeros((v.shape[0]))
        fake_normalize = 10 ** 0

        # sum of hidden nodes
        for i in range(h_values_1.shape[0]):
            a = np.dot( np.dot(v,w_1), h_values_1[i])

            for j in range(h_values_2.shape[0]):

                b = np.dot(np.dot(h_values_1[i], w_2), h_values_2[j])

                likelihood = likelihood + np.exp(a+b)/fake_normalize 
            #likelihood = likelihood + (-a + b + c)/fake_normalize 

        #x = np.arange(-15,15,0.1)
        #y = np.arange(-15,15,0.1)

        ranger = len(x)
        #print(ranger)
        #print(likelihood)

        X, Y = np.meshgrid(x, y)
        likeliout = np.array([likelihood[ranger * i: ranger * (i+1)] for i in range(ranger)])
        likeliout = np.transpose(likeliout)

        #likeliout = np.array([range(ranger) for i in range(ranger)])
        #print(likeliout)
        #print(X)
        #print(Y)
        
        from matplotlib import pyplot as  plt
        
        plt.figure()
        CS = plt.contour(X, Y, likeliout)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.title('DBM Relative Likelihood')
        plt.savefig("likelihood.png")
        #plt.savefig("test_likelihood_lr_"+str(self.learning_rate)+"_std_"+str(self.stddev)+"_numh_"+str(self.num_hidden)+".png")
        plt.close()
        #plt.show() 
