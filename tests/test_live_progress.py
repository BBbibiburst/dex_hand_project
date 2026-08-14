"""Formatting contracts for multi-worker terminal progress."""

from io import StringIO

from source.runtime.progress import LiveWorkerProgress


class InteractiveBuffer(StringIO):
    def isatty(self) -> bool:
        return True


def test_live_progress_renders_one_row_per_active_worker() -> None:
    stream = InteractiveBuffer()
    progress = LiveWorkerProgress(total=127, workers=2, stream=stream)
    progress.update(
        {
            "worker": "W1",
            "object_id": "ycb:002_master_chef_can",
            "phase": "EVOLUTION",
            "current": 12,
            "total": 20,
            "detail": "stable=8 archive=32",
        }
    )
    progress.update(
        {
            "worker": "W2",
            "object_id": "ycb:003_cracker_box",
            "phase": "TASK_PRECHECK",
            "current": 18,
            "total": 72,
            "detail": "scene=3 pull=0cm",
        }
    )
    progress.render()
    output = stream.getvalue()
    assert "Overall" in output
    assert "ycb:002_master_chef_can" in output
    assert "EVOLUTION" in output
    assert "12/20" in output
    assert "ycb:003_cracker_box" in output
    assert "TASK_PRECHECK" in output
    assert "18/72" in output


def test_completed_worker_row_is_removed() -> None:
    stream = InteractiveBuffer()
    progress = LiveWorkerProgress(total=1, workers=1, stream=stream)
    progress.update(
        {
            "worker": "W1",
            "object_id": "ycb:test",
            "phase": "DYNAMIC_LIFT",
            "current": 1,
            "total": 1,
        }
    )
    progress.mark_completed(object_id="ycb:test", solved=True)
    assert progress.completed == 1
    assert progress.solved == 1
    assert not progress.states
