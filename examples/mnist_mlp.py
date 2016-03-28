from __future__ import division, print_function
from blocks.algorithms import GradientDescent, AdaDelta
from blocks.bricks.cost import SquaredError, AbsoluteError, \
    CategoricalCrossEntropy, BinaryCrossEntropy
from blocks.extensions import Printing, FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop

from theano import tensor

from blocks.bricks import MLP, Rectifier, Tanh, Linear, Softmax, Logistic, \
    Identity
from blocks.initialization import IsotropicGaussian, Constant

from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten, ScaleAndShift
from adversarial_autoencoder.algorithm import SequentialTrainingAlgorithm, \
    SequentialTrainingDataMonitoring
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy as np

dataset = MNIST(('train',))

batch_size = 120
data_stream = DataStream.default_stream(
    dataset=dataset,
    iteration_scheme=ShuffledScheme(
        examples=dataset.num_examples,
        batch_size=batch_size  # NOTE must be equally divisable
    )
)
data_stream = Flatten(
    data_stream,
    which_sources=['features']
)
data_stream = ScaleAndShift(
    data_stream,
    which_sources=['features'],
    scale=2.0,
    shift=-1.0
)

dims = [28*28, 28*2, 12]
# dims = [28*28, 8]
# activations = [Rectifier()]

encoder = MLP(
    activations=[Rectifier(), Identity()],
    dims=dims,
    weights_init=IsotropicGaussian(0.01),
    biases_init=Constant(0.0)
)
encoder.initialize()

decoder = MLP(
    activations=[Rectifier(), Tanh()],
    dims=dims[::-1],
    weights_init=IsotropicGaussian(0.01),
    biases_init=Constant(0.0)
)
decoder.initialize()

adversarial_predictor = MLP(
    activations=[Rectifier(), Rectifier(), Logistic()],
    dims=[encoder.output_dim, encoder.output_dim*2, encoder.output_dim, 1],
    weights_init=IsotropicGaussian(0.01),
    biases_init=Constant(0.0)
)
adversarial_predictor.initialize()



x = tensor.matrix('features')
z = encoder.apply(x)

x_rec = decoder.apply(z)

rng = MRG_RandomStreams(seed=123)
z_prior = rng.normal((batch_size, encoder.output_dim))  # ok?
z_prior_enc = tensor.concatenate([z_prior, z], axis=0)
y = tensor.constant(np.array([[1]]*batch_size + [[0]]*batch_size).astype('int64'))
y_hat = adversarial_predictor.apply(z_prior_enc)

cost_rec = SquaredError().apply(x, x_rec)
cost_rec.name = 'cost_rec'
cg_rec = ComputationGraph(cost_rec)

cost_adversarial = BinaryCrossEntropy().apply(y, y_hat)
cost_adversarial.name = 'cost_adversarial'
cg_adversarial = ComputationGraph(cost_adversarial)

cost_rec_prior = cost_rec - 10000 * cost_adversarial
cost_rec_prior.name = 'cost_rec_prior'

algorithm_rec_prior = GradientDescent(
    cost=cost_rec_prior,
    step_rule=AdaDelta(),
    parameters=cg_rec.parameters,
    on_unused_sources='warn'
)

algorithm_adversarial = GradientDescent(
    cost=cost_adversarial,
    step_rule=AdaDelta(),
    parameters=cg_adversarial.parameters,
    on_unused_sources='warn'
)

main_loop = MainLoop(
    algorithm=SequentialTrainingAlgorithm(
        algorithm_steps=[
            algorithm_rec_prior,
            algorithm_adversarial
        ]
    ),
    data_stream=data_stream,
    extensions=[
        SequentialTrainingDataMonitoring(
            traing_data_monitorings=[
                TrainingDataMonitoring(
                    variables=[cost_rec, cost_rec_prior],
                    after_epoch=True
                ),
                TrainingDataMonitoring(
                    variables=[cost_adversarial],
                    after_epoch=True
                )
            ],
            after_epoch=True
        ),
        Printing(after_epoch=True),
        FinishAfter(after_n_epochs=10)
    ]
)