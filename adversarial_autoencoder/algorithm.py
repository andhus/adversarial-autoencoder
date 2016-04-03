from __future__ import division, print_function

from blocks.algorithms import TrainingAlgorithm


class SequentialTrainingAlgorithm(TrainingAlgorithm):
    """ Training Algorithm that combines several TrainingAlgorithms:s to be
    executed in sequence (for each same batch).

    See base class for further docs.

    Parameters
    ----------
    algorithm_steps : (TrainingAlgorithm) | [TrainingAlgorithm]
    """
    def __init__(
        self,
        algorithm_steps
    ):
        self._type_check(algorithm_steps)
        self.algorithm_steps = algorithm_steps

    @staticmethod
    def _type_check(algorithm_steps):
        type_error_msg = 'algorithm_steps must be tuple or list of ' \
                         'blocks.algorithms.TrainingAlgorithm'
        if isinstance(algorithm_steps, (list, tuple)):
            for algorithm_step in algorithm_steps:
                if not isinstance(algorithm_step, TrainingAlgorithm):
                    raise TypeError(type_error_msg)
        else:
            raise TypeError(type_error_msg)

    def initialize(self, **kwargs):
        for algorithm in self.algorithm_steps:
            algorithm.initialize(**kwargs)

    def process_batch(self, batch):
        for algorithm in self.algorithm_steps:
            algorithm.process_batch(batch)
