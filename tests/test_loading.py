import hashlib
import mmap
from pathlib import Path
import tempfile
import unittest
import numpy as np
from maxsim_pruning.loading import copy_into


@unittest.skipUnless(hasattr(mmap.mmap,'madvise') and hasattr(mmap,'MADV_DONTNEED'),
                     'Platform lacks mapping-page advice')
class LoadingTests(unittest.TestCase):
    def test_raw_and_npy_offsets_bitwise_and_source_unchanged(self):
        bits=np.arange(4096,dtype=np.uint16).reshape(128,32)
        bits[0,:4]=[0,32768,31744,32257]  # +0, -0, +inf, NaN payload
        expected=bits.view(np.float16)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for offset in (0,126,mmap.ALLOCATIONGRANULARITY+126):
                path=root/f'raw-{offset}'
                path.write_bytes(b'x'*offset+expected.tobytes())
                before=hashlib.sha256(path.read_bytes()).hexdigest()
                source=np.memmap(path,mode='r',dtype=np.float16,shape=expected.shape,offset=offset)
                for chunk in (1,127,4096,2**20):
                    target=np.empty_like(expected)
                    self.assertIs(copy_into(source,target,chunk_bytes=chunk),target)
                    self.assertEqual(target.tobytes(),expected.tobytes())
                source._mmap.close()
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)
            np.save(root/'array.npy',expected)
            source=np.load(root/'array.npy',mmap_mode='r')
            target=np.empty_like(expected);copy_into(source,target,chunk_bytes=63)
            self.assertEqual(target.tobytes(),expected.tobytes());source._mmap.close()

    def test_reject_writable_mapping_views_and_bad_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'array.npy';np.save(path,np.ones((8,4),dtype=np.float16))
            source=np.load(path,mmap_mode='r');target=np.empty((8,4),dtype=np.float16)
            for view in (source[1:],source[:,::2],source.reshape(4,8)):
                with self.assertRaises(ValueError):copy_into(view,np.empty_like(view))
            for chunk in (0,-1,1.5,True):
                with self.assertRaises(ValueError):copy_into(source,target,chunk_bytes=chunk)
            with self.assertRaises(ValueError):copy_into(source,target.astype(np.float32))
            with self.assertRaises(ValueError):copy_into(source,np.asarray(source))
            writable=np.load(path,mmap_mode='r+')
            with self.assertRaises(ValueError):copy_into(writable,target)
            writable._mmap.close();source._mmap.close()
            with self.assertRaises(ValueError):copy_into(source,target)


if __name__=='__main__':unittest.main()
