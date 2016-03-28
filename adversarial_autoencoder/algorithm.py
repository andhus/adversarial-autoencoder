from __future__ import division, print_function

from blocks.algorithms import TrainingAlgorithm
from blocks.extensions import SimpleExtension


class SequentialTrainingAlgorithm(TrainingAlgorithm):
    """ Training Algorithm that combines several TrainingAlgorithms:s to be
    executed in sequence for each batch.
    """
    def __init__(
        self,
        algorithm_steps
    ):
        self.algorithm_steps = algorithm_steps

    def initialize(self, **kwargs):
        for algorithm in self.algorithm_steps:
            algorithm.initialize(**kwargs)

    def process_batch(self, batch):
        for algorithm in self.algorithm_steps:
            algorithm.process_batch(batch)


class SequentialTrainingDataMonitoring(SimpleExtension):
    """
    """
    def __init__(self, traing_data_monitorings, **kwargs):
        kwargs.setdefault("before_training", True)
        super(SequentialTrainingDataMonitoring, self).__init__(**kwargs)
        self.training_data_monitorings = traing_data_monitorings

    @property
    def main_loop(self):
        if not hasattr(self, '_main_loop'):
            raise ValueError("main loop must be assigned to extension first")
        return self._main_loop

    @main_loop.setter
    def main_loop(self, value):
        self._main_loop = value
        for tdm in self.training_data_monitorings:
            tdm.main_loop = value

    def do(self, callback_name, *args):
        """Initializes the buffer or commits the values to the log.

        What this method does depends on from what callback it is called.
        When called within `before_training`, it initializes the
        aggregation buffer and instructs the training algorithm what
        additional computations should be carried at each step by adding
        corresponding updates to it. In all other cases it writes
        aggregated values of the monitored variables to the log.

        """
        if callback_name == 'before_training':
            if not isinstance(self.main_loop.algorithm,
                              SequentialTrainingAlgorithm):
                raise ValueError
            for algorithm, training_data_monitoring in zip(
                self.main_loop.algorithm.algorithm_steps,
                self.training_data_monitorings
            ):
                algorithm.add_updates(
                    training_data_monitoring._buffer.accumulation_updates
                )
                training_data_monitoring._buffer.initialize_aggregators()
        else:
            for training_data_monitoring in self.training_data_monitorings:
                training_data_monitoring.do(callback_name, *args)