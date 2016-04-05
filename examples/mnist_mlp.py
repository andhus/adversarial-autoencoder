from __future__ import division, print_function

import numpy as np
import theano

from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams

from blocks.algorithms import GradientDescent, AdaDelta, Momentum
from blocks.bricks.cost import SquaredError, BinaryCrossEntropy
from blocks.extensions import Printing, FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.bricks import MLP, Rectifier, Tanh, Logistic, Identity
from blocks.initialization import IsotropicGaussian, Constant
from blocks.filter import VariableFilter

from blocks_extras.extensions.plot import Plot

from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten, ScaleAndShift

from adversarial_autoencoder.algorithm import SequentialTrainingAlgorithm
from adversarial_autoencoder.monitoring import SequentialTrainingDataMonitoring


seed = 123
np.random.seed(seed=seed)
symbolic_rng = MRG_RandomStreams(seed=seed)

dataset = MNIST(('train',))

batch_size = 1000
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

dims = [28*28, 256, 128, 32, 2]

encoder = MLP(
    activations=[Rectifier() for dim in dims[1:-1]] + [Identity()],
    dims=dims,
    weights_init=IsotropicGaussian(0.01),
    biases_init=Constant(0.0)
)
encoder.initialize()

decoder = MLP(
    activations=[Rectifier() for dim in dims[1:-1]] + [Tanh()],
    dims=dims[::-1],
    weights_init=IsotropicGaussian(0.01),
    biases_init=Constant(0.0)
)
decoder.initialize()

adversarial_predictor = MLP(  # TODO Change to convolutional network!
    activations=[Rectifier(), Rectifier(), Logistic()],
    dims=[28*28, 256, 128, 1],
    weights_init=IsotropicGaussian(0.01),
    biases_init=Constant(0.0)
)
adversarial_predictor.initialize()


# RECONSTRUCTION
data = tensor.matrix('features')
data_encoded = encoder.apply(data)  # 'data' as opposed to 'prior'
data_decoded = decoder.apply(data_encoded)
cost_reconstruction = SquaredError().apply(
    data,
    data_decoded
)
cost_reconstruction.name = 'cost_reconstruction'
cg_reconstruction = ComputationGraph(cost_reconstruction)

# ADVERSARIAL
prior = symbolic_rng.normal(data_encoded.shape)
prior_decoded = decoder.apply(prior)

adversarial_input = tensor.concatenate(
    [data_decoded, prior_decoded],
    axis=0
)
adversarial_prediction = adversarial_predictor.apply(
    adversarial_input
)
data_target = tensor.zeros((data.shape[0], 1))  # 'negative' examples
prior_target = tensor.ones((prior.shape[0], 1))  # 'positive' examples
adversarial_target = tensor.concatenate([data_target, prior_target])

cost_adversarial = BinaryCrossEntropy().apply(
    adversarial_target,
    adversarial_prediction
)
cost_adversarial.name = 'cost_adversarial'
cg_adversarial = ComputationGraph(cost_adversarial)


# CONFUSION COST
adversarial_prior_prediction = adversarial_predictor.apply(
    prior_decoded
)
cost_confusion_factor = 30.0
cost_confusion = cost_confusion_factor * BinaryCrossEntropy().apply(
    tensor.zeros((prior.shape[0], 1)),  # 'negative' examples, make it belive it is part of data distribution
    adversarial_prior_prediction
)
cost_confusion.name = 'cost_confusion'

cost_autoencoder = cost_reconstruction + cost_confusion
cost_autoencoder.name = 'cost_autoencoder'

algorithm_autoencoder = GradientDescent(
    cost=cost_autoencoder,
    step_rule=AdaDelta(),
    parameters=cg_reconstruction.parameters,
    on_unused_sources='warn'
)

vf = VariableFilter(bricks=adversarial_predictor.children)
adversarial_parameters = vf(cg_adversarial.parameters)

algorithm_adversarial = GradientDescent(
    cost=cost_adversarial,
    step_rule=AdaDelta(),
    parameters=adversarial_parameters,
    on_unused_sources='warn'
)


# Main Loop
monitor_kwargs = {'every_n_batches': 100}
main_loop = MainLoop(
    algorithm=SequentialTrainingAlgorithm(
        algorithm_steps=[
            algorithm_adversarial,
            algorithm_autoencoder
        ]
    ),
    data_stream=data_stream,
    extensions=[
        SequentialTrainingDataMonitoring(
            training_data_monitoring_steps=[
                TrainingDataMonitoring(
                    variables=[cost_adversarial],
                    **monitor_kwargs
                ),
                TrainingDataMonitoring(
                    variables=[
                        cost_autoencoder,
                        cost_reconstruction,
                        cost_confusion
                    ],
                    **monitor_kwargs
                )
            ],
            **monitor_kwargs
        ),
        Printing(after_epoch=True),
        Plot(
            document='MLP AdvAE',
            channels=[
                [cost_adversarial.name],
                [cost_autoencoder.name, cost_reconstruction.name, cost_confusion.name]
            ],
            start_server=True,
            **monitor_kwargs
        ),
        FinishAfter(after_n_epochs=1000)
    ]
)

main_loop.run()

reconstruct_f = theano.function(
    inputs=[data],
    outputs=data_decoded
)

decode_f = theano.function(
    inputs=[data_encoded],
    outputs=data_decoded
)
