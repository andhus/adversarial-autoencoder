from __future__ import division, print_function
from blocks_extras.extensions.plot import Plot

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

dims = [28*28, 128, 64, 12]

encoder = MLP(
    activations=[Rectifier(), Rectifier(), Identity()],
    dims=dims,
    weights_init=IsotropicGaussian(0.01),
    biases_init=Constant(0.0)
)
encoder.initialize()

decoder = MLP(
    activations=[Rectifier(), Rectifier(), Tanh()],
    dims=dims[::-1],
    weights_init=IsotropicGaussian(0.01),
    biases_init=Constant(0.0)
)
decoder.initialize()

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

algorithm_reconstruction = GradientDescent(
    cost=cost_reconstruction,
    step_rule=AdaDelta(),
    parameters=cg_reconstruction.parameters,
    on_unused_sources='warn'
)

main_loop = MainLoop(
    algorithm=SequentialTrainingAlgorithm(
        algorithm_steps=[
            algorithm_reconstruction
        ]
    ),
    data_stream=data_stream,
    extensions=[
        SequentialTrainingDataMonitoring(
            training_data_monitoring_steps=[
                TrainingDataMonitoring(
                    variables=[
                        cost_reconstruction,
                    ],
                    every_n_batches=50
                )
            ],
            every_n_batches=50
        ),
        Printing(after_epoch=True),
        FinishAfter(after_n_epochs=100),
        Plot(
            document='AE',
            channels=[[cost_reconstruction.name]],
            start_server=True,
            every_n_batches=50
        )
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