"""
This file defines abstractions for slurm db, that are needed for 
scheduling modeling and runtime estimation feature engeneering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Generic, TypeVar, Iterable
from datetime import datetime, timedelta


class JobState(Enum):
    """
    Enum representing job state in slurm.

    For reference see:
        https://github.com/SchedMD/slurm/blob/master/slurm/slurm.h#L261
    """

    PENDING = auto()
    RUNNING = auto()
    SUSPENDED = auto()
    COMPLETE = auto()
    CANCELLED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    NODE_FAIL = auto()
    PREEMPTED = auto()
    BOOT_FAIL = auto()
    DEADLINE = auto()
    OOM = auto()


class JobRecord(ABC):
    """
    Record representing some part of job record in slurmdb, that can be used by
    ML models.

    This class currently represents not all fields of original record in
    slurmdb, because of specifics of a project, but can be extended later.

    For reference see:
        https://github.com/SchedMD/slurm/blob/master/slurm/slurmdb.h#L835
    """

    @property
    @abstractmethod
    def alloc_nodes(self) -> int:
        """Number of required nodes"""

    @property
    @abstractmethod
    def elapsed(self) -> timedelta:
        """Elapsed time of job, if job is running this is `now() - job.start`"""

    @property
    @abstractmethod
    def end(self) -> datetime | None:
        """Time when job was finished"""

    @property
    @abstractmethod
    def exitcode(self) -> int:
        """Exit code of job, if it is terminated properly"""

    @property
    @abstractmethod
    def gid(self) -> str:
        """Group id of user, that submitted job"""

    @property
    @abstractmethod
    def jobid(self) -> int:
        """Unique id of job in system"""

    @property
    @abstractmethod
    def jobname(self) -> str:
        """Name that user gave to the job"""

    @property
    @abstractmethod
    def nodes(self) -> list[str]:
        """
        List of allocated nodes names or empty list, if job was not
        started
        """

    @property
    @abstractmethod
    def partition(self) -> str:
        """Target cluster partition"""

    @property
    @abstractmethod
    def priority(self) -> int:
        """Priority of job, that will be considered when scheduling"""

    @property
    @abstractmethod
    def req_cpus(self) -> int:
        """Number of required CPUs"""

    @property
    @abstractmethod
    def start(self) -> datetime | None:
        """Time when job was started"""

    @property
    @abstractmethod
    def state(self) -> JobState:
        """Current state of job"""

    @property
    @abstractmethod
    def submit(self) -> datetime:
        """Time when job was submitted"""

    @property
    @abstractmethod
    def timelimit(self) -> timedelta:
        """User estimation of maximum runtime of a job"""

    @property
    @abstractmethod
    def uid(self) -> str:
        """Id of user, that submitted job"""


class JobOuterInfo(ABC):
    """Information that is not directly presented in slurm db, but may known"""

    @property
    @abstractmethod
    def field(self) -> str | None:
        """Area of study, that is associated with user"""


RecordType = TypeVar("RecordType")


class SimpleSelectQuery(Generic[RecordType], ABC):
    """
    Builder of simple select query (without joins) in slurm db. Each query is
    associated with table.
    """

    @abstractmethod
    def where(self, condition: str, **kwargs) -> SimpleSelectQuery[RecordType]:
        """
        General filtering condition, that can be `eval`-ed with all the
        properties of record being injected into namespace. All the kwargs is also
        injected.

        Engine will try its best to optimize this filter, but in worst case it
        will evaluate this condition for every row and leave only ones, those
        conditions was evaluated to true-ish value.
        """

    @abstractmethod
    def order_by(
        self, expression: str, descending: bool = False, **kwargs
    ) -> SimpleSelectQuery[RecordType]:
        """
        General sort expression, that can be `eval`-ed with all the properties
        of record being injected into namespace, all the kwargs are also
        injected.

        There can be multiple `order_by` methods.

        Result of expression must be comparable.

        Engine will try its best to optimize this sort, but in worst case it
        will evaluate this condition for every row and leave only ones, those
        conditions was evaluated to true-ish value.
        """

    @abstractmethod
    def execute(self) -> Iterable[RecordType]:
        """Stream results of query"""


class ReadOnlySlurmDBModel(ABC):
    """
    This class represents simplified model of slurm db.

    This class can be used for feature engeneering for runtime estimation.
    """

    @abstractmethod
    def get_job(self, jobid: int) -> JobRecord | None:
        """Returns job with given id or None, if there is no such job"""

    @abstractmethod
    def stream_jobs(self) -> Iterable[JobRecord]:
        """
        Yields all jobs in job db.

        If write operation occurs while this operation, result is undefined.
        """

    @abstractmethod
    def select_jobs(self) -> SimpleSelectQuery[JobRecord]:
        """Returns query builder for selecting jobs"""

    @abstractmethod
    def get_job_outer_info(self, jobid: int) -> JobOuterInfo:
        """Get other information about job"""


class SlurmDBModel(ReadOnlySlurmDBModel):
    @abstractmethod
    def set_job(self, rec: JobRecord):
        """Inserts or updates job record in database"""
