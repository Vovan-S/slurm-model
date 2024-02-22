from abc import ABC, abstractmethod
from datetime import timedelta
from typing import TypeAlias
import numpy as np
import pandas as pd

from slurm_model.data import ReadOnlySlurmDBModel, JobRecord


class RuntimeEstimator(ABC):
    """
    Generic runtime estimator, that will be used by dispatching algorithm.
    """

    @abstractmethod
    def estimate(self, job: JobRecord, db: ReadOnlySlurmDBModel) -> timedelta:
        """
        Predicts runtime of given job. Estimator can use data in db for feature
        engeneering.
        """


JobFeatures: TypeAlias = np.ndarray | pd.DataFrame
JobTarget: TypeAlias = np.ndarray | float


class StatelessRuntimeEstimator(ABC):
    """
    Partial realisation, that can be used for defining and traning stateless
    models.

    This class defines operations needed for materializing datasets with custom
    features:
        - `extract_features` method defines how features must be extracted
        - `timedelta_to_y` and `y_to_timedelta` class methods define how class
          represents target data (by default this is timedelta in seconds),
          `timedelta_to_y` defines conversion from `timedelta` to numerical
          representation, `y_to_timedelta` defines inverse conversion.

    General algorithm for creating dataset (X, y) for class `cls` then would be:
        create empty db
        create list of features
        create list of targets
        for each record in real data sorted by submit time:
            make job record from row as if it was just submitted
            get real elapsed time from row
            append `cls.extract_features(job record, db)` to features list
            append `cls.timedelta_to_y(elapsed time)` to target list
            make full job record frow row and insert it into db
        X := concatination of features list
        y := concatination of targets list

    General algorith of inference is implemented in `estimate` method.
    """

    @classmethod
    @abstractmethod
    def extract_features(cls, job: JobRecord, db: ReadOnlySlurmDBModel) -> JobFeatures:
        """
        This method extracts all needed features for given job and db state.

        This method will be used for inference and should be used for creating
        intermediate train or test datasets.
        """

    @abstractmethod
    def predict(self, X: JobFeatures) -> JobTarget:
        """This method implements model logic"""

    @classmethod
    def timedelta_to_y(cls, runtime: timedelta) -> JobTarget:
        """
        This method defines conversion from timedelta to model specific target
        representation.
        """
        return runtime.total_seconds()

    @classmethod
    def y_to_timedelta(cls, prediction: JobTarget) -> timedelta:
        """
        This method defines conversion from model specific target
        representation to timedelta.
        """
        return timedelta(seconds=prediction)

    def estimate(self, job: JobRecord, db: ReadOnlySlurmDBModel) -> timedelta:
        x = self.extract_features(job, db)
        y = self.predict(x)
        return self.y_to_timedelta(y)
