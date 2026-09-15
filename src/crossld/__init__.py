from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("crossld")
except PackageNotFoundError:
    __version__ = "unknown"
