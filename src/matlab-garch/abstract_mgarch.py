from abc import ABC, abstractmethod
import numpy as np
import matlab.engine

class AbstractMGARCH(ABC):
    # Singleton matlab engine instance
    _engine = None

    def __init__(self):
        if AbstractMGARCH._engine is None:
            AbstractMGARCH._engine = matlab.engine.start_matlab()

    @property
    def engine(self):
        return AbstractMGARCH._engine

    def close(self):
        if AbstractMGARCH._engine is not None:
            AbstractMGARCH._engine.quit()
            AbstractMGARCH._engine = None

    @staticmethod
    def py_to_ml(data):
        """
        Convert Python data (numeric types, lists, tuples, or numpy arrays)
        into a MATLAB compatible data type.
        """
        # Convert numpy array to list, if needed.
        if isinstance(data, np.ndarray):
            data = data.tolist()
        # If data is a number, wrap it in a nested list.
        if not isinstance(data, (list, tuple)):
            data = [[data]]
        # If data is a 1D list, convert it into a 2D row vector.
        elif data and not isinstance(data[0], (list, tuple)):
            data = [data]
        return matlab.double(data)

    @staticmethod
    def ml_to_py(data):
        """
        Convert MATLAB numeric data (e.g., matlab.double) into a Python numpy array.
        """
        try:
            # Attempt to convert to a list.
            py_list = list(data)
            # Some MATLAB types return rows as their elements.
            if py_list and not isinstance(py_list[0], (list, tuple)):
                py_list = [py_list]
            return np.array(py_list)
        except Exception:
            return data

    @abstractmethod
    def run(self, *args, **kwargs):
        """Implement the method to run your MGARCH model."""
        pass


