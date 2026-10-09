This is derived from https://pypi.org/project/gradio-modal/ (Apache license), credit to *aliabid94*, with some minor changes:
avoiding typical install method into VEnv as that was broken, maybe a Gradio version incompatibility
adding `self._constructor_args = {}` to __init__() allows it to run
trimmed CSS, from 1480602 bytes to 822 bytes (and this is with a commented out alternative layout included, and proper formatting)
JS is now at 20756 bytes, from 545520 bytes. The functionality I'm using it for is unchanged, but I can't be sure something wasn't lost.
