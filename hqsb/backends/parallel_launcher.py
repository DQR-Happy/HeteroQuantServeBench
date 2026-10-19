"""Private torchrun agent; Linux parent death cannot leave GPU ranks behind."""


def parent_lifetime():
    import ctypes
    import os
    import signal

    if os.name != "posix":
        raise RuntimeError("L9_LINUX_PROCESS_SUPERVISION_REQUIRED")
    if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0:
        raise OSError("L9_PARENT_DEATH_SIGNAL_FAILED")
    if os.getppid() == 1:
        raise RuntimeError("L9_SUPERVISOR_ALREADY_EXITED")


if __name__ == "__main__":
    parent_lifetime()
    from torch.distributed.run import main

    main()
