from __future__ import division, print_function

from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import TrainingDataMonitoring

from adversarial_autoencoder.algorithm import SequentialTrainingAlgorithm


class SequentialTrainingDataMonitoring(SimpleExtension):
    """ Training data monitoring Extension to be used along with the
    SequentialTrainingAlgorithm. The standard blocks TrainingDataMonitoring
    extension does only support the training algorithm to be of the type
    DifferentiableCostMinimizer. A Simple way to support monitoring with
    multiple DifferentiableCostMinimizer algorithms is to define one
    TrainingDataMonitoring corresponding to each one of them.

    Parameters
    ----------
    training_data_monitoring_steps :
            [TrainingDataMonitoring] | (TrainingDataMonitoring)
        The variables to monitor in each of the TrainingDataMonitoring
        extensions must correspond to the DifferentiableCostMinimizer in the
        SequentialTrainingAlgorithm _based on order_.
    """
    def __init__(
        self,
        training_data_monitoring_steps,
        **kwargs
    ):
        kwargs.setdefault("before_training", True)
        super(SequentialTrainingDataMonitoring, self).__init__(**kwargs)
        self.training_data_monitoring_steps = training_data_monitoring_steps

    @staticmethod
    def _type_check(training_data_monitoring_steps):
        type_error_msg = 'training_data_monitoring_steps must be tuple or ' \
                         'list of blocks.monitoring.TrainingDataMonitoring'
        if isinstance(training_data_monitoring_steps, (list, tuple)):
            for training_data_monitoring_step in training_data_monitoring_steps:
                if not isinstance(
                    training_data_monitoring_step,
                    TrainingDataMonitoring
                ):
                    raise TypeError(type_error_msg)
        else:
            raise TypeError(type_error_msg)

    @property
    def main_loop(self):
        if not hasattr(self, '_main_loop'):
            raise ValueError("main loop must be assigned to extension first")
        return self._main_loop

    @main_loop.setter
    def main_loop(self, value):
        self._main_loop = value
        for tdm in self.training_data_monitoring_steps:
            tdm.main_loop = value

    def do(
        self,
        callback_name,
        *args
    ):
        """ When called 'before_training' each of the training data monitoring
        extensions is set up with its corresponding training algorithm (based on
        order) in the SequentialTrainingAlgorithm.

        For any other call th call_back_name and *args are simply forwarded to
        each of the training data monitoring extensions.

        See base class for further docs.
        """
        if callback_name == 'before_training':
            if not isinstance(self.main_loop.algorithm,
                              SequentialTrainingAlgorithm):
                raise ValueError(
                    'SequentialTrainingDataMonitoring only works together with '
                    'the SequentialTrainingAlgorithm, got {}'.format(
                        self.main_loop.algorithm
                    )
                )
            for algorithm, training_data_monitoring_step in zip(
                self.main_loop.algorithm.algorithm_steps,
                self.training_data_monitoring_steps
            ):
                algorithm.add_updates(
                    training_data_monitoring_step._buffer.accumulation_updates
                )
                training_data_monitoring_step._buffer.initialize_aggregators()
        else:
            for training_data_monitoring_step in self.training_data_monitoring_steps:
                training_data_monitoring_step.do(callback_name, *args)
