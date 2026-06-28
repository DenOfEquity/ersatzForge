import sys
import textwrap
import traceback

from modules_forge import colour_code as cc


exception_records = []


def format_traceback(tb):
    return [[f"{x.filename}, line {x.lineno}, {x.name}", x.line] for x in traceback.extract_tb(tb)]


def format_exception(e, tb):
    return {"exception": str(e), "traceback": format_traceback(tb)}


def get_exceptions():
    try:
        return list(reversed(exception_records))
    except Exception as e:
        return str(e)


def record_exception():
    _, e, tb = sys.exc_info()
    if e is None:
        return

    if exception_records and exception_records[-1] == e:
        return

    exception_records.append(format_exception(e, tb))

    if len(exception_records) > 5:
        exception_records.pop(0)


def report(message: str, *, exc_info: bool = False) -> None:
    """
    Print an error message to stderr, with optional traceback.
    """

    record_exception()

    for line in message.splitlines():
        print("***", line, file=sys.stderr)
    if exc_info:
        print(textwrap.indent(traceback.format_exc(), "    "), file=sys.stderr)
        print("---", file=sys.stderr)


def display(e: Exception, task, *, full_traceback=False):
    record_exception()

    print(f"{task or 'error'}: {type(e).__name__}", file=sys.stderr)
    te = traceback.TracebackException.from_exception(e)
    if full_traceback:
        # include frames leading up to the try-catch block
        te.stack = traceback.StackSummary(traceback.extract_stack()[:-2] + te.stack)
    print(*te.format(), sep="", file=sys.stderr)


already_displayed = {}


def display_once(e: Exception, task):
    record_exception()

    if task in already_displayed:
        return

    display(e, task)

    already_displayed[task] = 1


def run(code, task):
    try:
        code()
    except Exception as e:
        display(task, e)


def check_versions():
    from packaging import version
    from modules import shared

    expected_torch_version = "2.3.1"
    expected_xformers_version = "0.0.27"
    expected_gradio_version = "4.40.0"

    version_mismatch = False
    if version.parse(shared.torch_version) < version.parse(expected_torch_version):
        version_mismatch = True
        print(f"{cc.WARNING}Torch version mismatch:{cc.RESET} {shared.torch_version} installed; minimum {expected_torch_version} expected.")
        print(f"To reinstall, run with commandline argument {cc.INFO2}--reinstall-torch{cc.RESET}.")
        print("Note that this will cause a lot of large files to be downloaded.")

    if shared.xformers_available[0]:
        if version.parse(shared.xformers_available[1]) < version.parse(expected_xformers_version):
            version_mismatch = True
            print(f"{cc.WARNING}XFormers version mismatch:{cc.RESET} {shared.xformers_available[1]} installed; {expected_xformers_version} expected.")
            print(f"To reinstall, run with commandline argument {cc.INFO2}--reinstall-xformers{cc.RESET}.")

    if shared.gradio_version != expected_gradio_version:
        version_mismatch = True
        print(f"{cc.WARNING}Gradio version mismatch:{cc.RESET} {shared.gradio_version} installed; {expected_gradio_version} expected.")
        print("Using a different version of Gradio is extremely likely to break the program.")
        print("Common reasons why you have a mismatched Gradio version:")
        print("    - you use --skip-install commandline argument.")
        print("    - you use webui.py to start the program instead of launch.py.")
        print("    - an extension installs the incompatible version.")

    if version_mismatch:
        print(f"Use {cc.INFO2}--skip-version-check{cc.RESET} commandline argument to disable version checks.")
