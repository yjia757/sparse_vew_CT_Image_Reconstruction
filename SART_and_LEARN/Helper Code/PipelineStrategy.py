
class PipelineStrategy(object):

    def __init__(self, session, args, geometry):
        self.args = args
        self.sess = session
        self.geometry = geometry
        #****************strategy initialization**************************************
        self.strategy_type = strategy_type 
        if(strategyType == "SART"):
            self.model = SART_model(geometry, np.zeros(geometry.volume_shape, dtype=np.float32))
        else if(strategyType == "LEARN"):
            self.model = LEARN_model(geometry, np.zeros(geometry.volume_shape, dtype=np.float32))
        #*****************************************************************************
        self.regularizer_weight = 0.5


    def init_placeholder_graph(self):
        self.learning_rate = tf.get_variable(name='learning_rate', dtype=tf.float32, initializer=tf.constant(0.0001), trainable=False)
        self.learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate_placeholder')
        self.set_learning_rate = self.learning_rate.assign(self.learning_rate_placeholder)

        


#******************************* execute chosen strategy ************************************
    def execute_strategy(self, zero_vector, aquired_sinogram, strategy_type):
        if(strategyType == "SART"):
            self.model = PipelineSART()
        else if(strategyType == "LEARN"):
            self.model = PipelineLEARN()

#************************************************************************************
