"""Usage diagnostics shared by installed analyze workflows."""


def check_usage_error(done) -> None:
    assert done.returncode == 2
    assert done.stdout == ""
    assert done.stderr.startswith("usage: tilefoundry analyze")
