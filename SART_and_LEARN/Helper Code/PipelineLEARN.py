





class pipeline(object):

    def __init__(self, session, args, geometry):
        self.args = args
        self.sess = session
        self.geometry = geometry
        self.model = iterative_reco_model(geometry, np.zeros(geometry.volume_shape, dtype=np.float32))
        self.regularizer_weight = 0.5



#***************************** Need to further modify build_graph and train to resemble SART *****************************
    def build_graph(self, input_type, input_shape, label_shape):

        self.init_placeholder_graph()
        g_opt = tf.train.AdamOptimizer(self.learning_rate)

        # Tensor placeholders that are initialized later. Input and label shape are assumed to be equal
        self.input_placeholder = tf.placeholder(input_type, (None, input_shape[0], input_shape[1]))
        self.label_placeholder = tf.placeholder(input_type, (None, label_shape[0], label_shape[1]))

        # Make pairs of elements. (X, Y) => ((x0, y0), (x1)(y1)),....
        image_set = tf.data.Dataset.from_tensor_slices((self.input_placeholder, self.label_placeholder))
        # Identity mapping operation is needed to include multi-tthreaded queue buffering.
        image_set = image_set.map(lambda x, y: (x, y), num_parallel_calls=4).prefetch(buffer_size=200)
        # Batch dataset. Also do this if batchsize==1 to add the mandatory first axis for the batch_size
        image_set = image_set.batch(1)
        # Repeat dataset for number of epochs
        image_set = image_set.repeat(self.args.num_epochs + 1)
        # Select iterator
        self.iterator = image_set.make_initializable_iterator()

        self.input_element, self.label_element  = self.iterator.get_next()

        self.current_sino, self.current_reco = self.model.model(self.input_element)

        tv_loss_x = tf.image.total_variation(tf.transpose(self.current_reco))
        tv_loss_y = tf.image.total_variation(self.current_reco)

        self.loss = tf.reduce_sum(tf.squared_difference(self.label_element, self.current_sino)) + self.regularizer_weight*(tv_loss_x+tv_loss_y)
        self.train_op = g_opt.minimize(self.loss)




def train(self, zero_vector, acquired_sinogram):
    self.build_graph(zero_vector.dtype, zero_vector.shape, acquired_sinogram.shape)

    self.sess.run(tf.global_variables_initializer())
    self.sess.run(tf.local_variables_initializer())

    learning_rate = self.args.learning_rate

    zero_vector_train = np.expand_dims(zero_vector, axis=0)
    acquired_sinogram_train = np.expand_dims(acquired_sinogram, axis=0)

    # initialise iterator with train data
    #self.sess.run(self.train_init_op, feed_dict={self.input_placeholder: zero_vector_train, self.label_placeholder: acquired_sinogram_train})
    self.sess.run(self.iterator.initializer, feed_dict={self.input_placeholder: zero_vector_train, self.label_placeholder: acquired_sinogram_train})

    min_loss = 10000000000000000
    for epoch in range(1, self.args.num_epochs + 1):

        _ = self.sess.run([self.set_learning_rate], feed_dict={self.learning_rate_placeholder: learning_rate})

        _, loss, current_sino, current_reco, label = self.sess.run([self.train_op, self.loss, self.current_sino, self.current_reco, self.label_element])

        if loss > min_loss * 1.005:
            break
        if epoch % 50 is 0:
            print('Epoch: %d' % epoch)
            print('Loss %f' % loss)
        if min_loss > loss:
            min_loss = loss
            self.result = current_reco
#***************************************************************************************


#***************************** replaces model *****************************
class LEARN_model:

    def __init__(self, geometry, reco_initialization, strategyType):
        self.geometry = geometry
        self.reco = tf.get_variable(name='reco', dtype=tf.float32,
                                    initializer=tf.expand_dims(reco_initialization, axis=0),
                                    trainable=True, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

    def model(self, input_volume):
        self.updated_reco = tf.add(input_volume, self.reco)
        self.current_sino = projection_2d.parallel_projection2d(self.updated_reco, self.geometry)
        return self.current_sino, self.reco

#***************************************************************************************

