"""
Job State Machine
-----------------
Enforces strict lifecycle transitions for asynchronous CAE simulation jobs.
Prevents invalid operations such as solving without validation or skipping meshing.
"""

from typing import Set, Dict
from ..domain.models import JobState


VALID_TRANSITIONS: Dict[JobState, Set[JobState]] = {
    JobState.CREATED: {JobState.CONFIGURED, JobState.CANCELLED, JobState.FAILED},
    JobState.CONFIGURED: {JobState.VALIDATING, JobState.CONFIGURED, JobState.CANCELLED, JobState.FAILED},
    JobState.VALIDATING: {JobState.QUEUED, JobState.CONFIGURED, JobState.FAILED, JobState.CANCELLED},
    JobState.QUEUED: {JobState.IMPORTING, JobState.CANCELLED, JobState.FAILED},
    JobState.IMPORTING: {JobState.MESHING, JobState.FAILED, JobState.CANCELLED},
    JobState.MESHING: {JobState.SOLVING, JobState.FAILED, JobState.CANCELLED},
    JobState.SOLVING: {JobState.POST_PROCESSING, JobState.FAILED, JobState.CANCELLED},
    JobState.POST_PROCESSING: {JobState.COMPLETED, JobState.FAILED},
    JobState.COMPLETED: set(),  # Terminal
    JobState.FAILED: set(),     # Terminal
    JobState.CANCELLED: set(),  # Terminal
}


class InvalidStateTransitionError(Exception):
    """Raised when an illegal lifecycle transition is attempted."""
    pass


class JobStateMachine:
    """Manages state transitions and lifecycle validation for simulation jobs."""

    @staticmethod
    def can_transition(current_state: JobState, target_state: JobState) -> bool:
        """Checks if a transition from current_state to target_state is permitted."""
        allowed = VALID_TRANSITIONS.get(current_state, set())
        return target_state in allowed

    @classmethod
    def transition(cls, current_state: JobState, target_state: JobState) -> JobState:
        """Transitions state or raises InvalidStateTransitionError."""
        if not cls.can_transition(current_state, target_state):
            raise InvalidStateTransitionError(
                f"Geçersiz durum geçişi: '{current_state.value}' -> '{target_state.value}'. "
                f"İzin verilen sonraki durumlar: {[s.value for s in VALID_TRANSITIONS.get(current_state, set())]}"
            )
        return target_state

    @staticmethod
    def is_terminal(state: JobState) -> bool:
        """Returns True if the state is terminal (COMPLETED, FAILED, CANCELLED)."""
        return state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}


RUNNING_STATES = {
    JobState.QUEUED, JobState.IMPORTING, JobState.MESHING,
    JobState.SOLVING, JobState.POST_PROCESSING,
}


def check_update(current: JobState, target: JobState) -> None:
    """The rules ``JobStorage.update_job_state`` enforces on every change.

    These are deliberately looser than ``VALID_TRANSITIONS``, which describes the
    ideal path: validation is recorded as a report rather than a state, and a
    run that died with its process is left in a running state that has to be
    restartable.  What is refused is what would corrupt a result:

    * a COMPLETED or CANCELLED job cannot be run again or moved anywhere except
      back to CONFIGURED (a new configuration), so a finished result cannot be
      overwritten behind its provenance record;
    * a job that was never configured cannot be queued.

    A failure can be recorded for any job that has not finished, and a running or
    configured job can be cancelled.
    """
    if target == current:
        return
    if current in (JobState.COMPLETED, JobState.CANCELLED) and target != JobState.CONFIGURED:
        raise InvalidStateTransitionError(
            "'%s' durumundaki iş '%s' durumuna geçemez. Sonucu korumak için işi 'configure_*' ile "
            "yeniden yapılandırın ya da 'clone_job' ile kopyalayın." % (current.value, target.value)
        )
    if target == JobState.QUEUED and current == JobState.CREATED:
        raise InvalidStateTransitionError(
            "Yapılandırılmamış iş çözüme alınamaz. Önce 'configure_*' aracını çalıştırın."
        )
    if target == JobState.CANCELLED and current in (JobState.CREATED, JobState.FAILED):
        raise InvalidStateTransitionError(
            "'%s' durumundaki iş iptal edilemez." % current.value
        )
