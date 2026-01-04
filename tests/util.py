import logging
import sys
import numpy as np
from io import StringIO
from pathlib import Path
import json
import unittest
import torch
from torch import Tensor
from zensols.deeplearn import TorchConfig
from zensols.deeplearn.vectorize import SparseTensorFeatureContext
from zensols.config import ImportConfigFactory, ImportIniConfig

logger = logging.getLogger(__name__)


class Should(object):
    def __init__(self, path: Path, is_write: bool, dtype: str = 'identity'):
        self._path = path
        self._dtype = dtype
        self._is_write = is_write
        self._data: dict[str, str] = {}

    @staticmethod
    def _write_torch(arr: Tensor, fmt='%.6g') -> str:
        """Serialize a NumPy array to a StringIO buffer as text, preserving
        dtype and shape.

        """
        if not isinstance(arr, Tensor):
            raise TypeError('Expected a Tensor')
        arr = arr.detach().numpy()
        buf = StringIO()
        buf.write(f'# dtype: {arr.dtype}\n')
        buf.write('# shape: ' + ' '.join(map(str, arr.shape)) + '\n')
        if arr.ndim == 1:
            np.savetxt(buf, arr[None, :], fmt=fmt)
        else:
            np.savetxt(buf, arr, fmt=fmt)
        return buf.getvalue()

    @staticmethod
    def _read_torch(val: str) -> np.ndarray:
        """Deserialize a NumPy array from a StringIO buffer, restoring dtype and
        shape.

        """
        buf = StringIO(val)
        buf.seek(0)
        dtype_line = buf.readline()
        shape_line = buf.readline()
        dtype = np.dtype(dtype_line.split(":")[1].strip())
        shape = tuple(map(int, shape_line.split(":")[1].split()))
        data = np.loadtxt(buf, dtype=dtype)
        # Handle scalar, 1-D, and higher-D uniformly
        if shape == ():
            return np.array(data, dtype=dtype)
        arr = np.array(data, dtype=dtype)
        # loadtxt returns (n,) or (n, m); reshape fixes ambiguity
        return torch.from_numpy(arr.reshape(shape))

    def load(self):
        assert not self._is_write
        read_fn = self._read_torch if self._dtype == 'torch' else lambda x: x
        self._data = dict(map(
            lambda t: (t[0], read_fn(t[1])),
            json.loads(self._path.read_text()).items()))

    def save(self):
        assert self._is_write
        self._path.write_text(json.dumps(self._data, indent=4))

    def __call__(self, name: str, arr: Tensor):
        if self._is_write:
            self[name] = arr
        else:
            arr = self[name]
        return arr

    def __getitem__(self, name: str) -> Tensor:
        return self._data[name]

    def __setitem__(self, name: str, arr: Tensor):
        write_fn = self._write_torch if self._dtype == 'torch' else lambda x: x
        val = write_fn(arr)
        if name in self._data:
            assert val == self._data[name]
        self._data[name] = val


class TestFeatureVectorization(unittest.TestCase):
    def setUp(self):
        if hasattr(self.__class__, 'CONF_FILE'):
            path = self.CONF_FILE
        else:
            path = 'test-resources/features.conf'
        #config = AppConfig(path)
        self.fac = ImportConfigFactory(ImportIniConfig(path), shared=True)
        self.sent_text = 'I am a citizen of the United States of America.'
        self.def_parse = ('I', 'am', 'a', 'citizen', 'of',
                          'the United States of America', '.')
        if not hasattr(self.__class__, 'NO_VECTORIZER'):
            self.vmng = self.fac.instance('feature_vectorizer_manager')
        self.sent_text2 = self.sent_text + " My name is Paul Landes."
        self.maxDiff = sys.maxsize

    def assertTensorEquals(self, should, tensor):
        def pr():
            import torch
            torch.set_printoptions(threshold=10_000)
            print()
            print('output:')
            if 1:
                print(tensor)
            else:
                print(tensor.to_sparse())
            print('_' * 80)

        if not should.shape == tensor.shape:
            pr()
        self.assertEqual(should.shape, tensor.shape)
        try:
            eq = TorchConfig.equal(should, tensor)
        except RuntimeError as e:
            logger.error(f'error comparing {should} with {tensor}')
            raise e
        if not eq:
            logger.error(f'tensor {should} does not equal {tensor}')
        if not eq:
            pr()
        self.assertTrue(eq)

    def _to_sparse(self, arr: Tensor):
        return SparseTensorFeatureContext.to_sparse(arr)[0][0]
