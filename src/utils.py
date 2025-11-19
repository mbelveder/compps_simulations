import os


class ChangeDir:
    """
    Context manager for changing the current working directory
    and then return back to the one you started from. Prevents from
    unwanted change of the working derictory.
    """
    def __init__(self, new_path):
        self.new_path = os.path.expanduser(new_path)

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, *_):
        os.chdir(self.saved_path)
