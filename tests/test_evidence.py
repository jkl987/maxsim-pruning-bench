import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from maxsim_pruning.evidence import (EvidenceError, verify_manifest, median_time,
                                    rows_by_query, paired_summary, hybrid_summary)


class EvidenceTests(unittest.TestCase):
    def test_corruption_rejected_even_under_optimized_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'evidence').mkdir();(root/'data').write_bytes(b'original')
            (root/'evidence/SHA256SUMS.json').write_text(json.dumps(
                {'data':{'sha256':hashlib.sha256(b'original').hexdigest()}}))
            self.assertEqual(verify_manifest(root),1)
            (root/'data').write_bytes(b'changed')
            with self.assertRaises(EvidenceError):verify_manifest(root)
            proc=subprocess.run([sys.executable,'-O','-c',
                'from maxsim_pruning.evidence import verify_manifest; import sys; verify_manifest(sys.argv[1])',tmp],
                capture_output=True,text=True)
            self.assertNotEqual(proc.returncode,0)
            self.assertIn('SHA256 mismatch',proc.stderr)

    def test_manifest_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'evidence').mkdir()
            (root/'evidence/SHA256SUMS.json').write_text(json.dumps({'../outside':{'sha256':'0'*64}}))
            with self.assertRaisesRegex(EvidenceError,'escapes root'):verify_manifest(root)

    def test_timing_and_query_validation(self):
        for times in ([1,2],[1,2,3,4,float('nan')],[1,2,3,4,-1],[1,2,3,4,0]):
            with self.assertRaises(EvidenceError):median_time(times)
        with self.assertRaises(EvidenceError):
            rows_by_query({'complete':True,'records':[{'query_id':'a'},{'query_id':'a'}]},2)
        with self.assertRaises(EvidenceError):rows_by_query({'complete':False,'records':[]},0)

    def test_pairing_is_by_id_and_not_row_order(self):
        a={'b':2.,'a':1.};b={'a':2.,'b':4.}
        z=paired_summary(a,b,resamples=100)
        self.assertEqual(z['relative_change'],-.5)
        self.assertEqual(z['query_bootstrap_95ci'],[-.5,-.5])
        self.assertEqual(z['query_wins'],2)
        with self.assertRaises(EvidenceError):paired_summary(a,{'a':2.,'c':4.})

    def test_hybrid_aligns_sessions_and_preserves_query_unit(self):
        records=[{'query_id':str(i),'times_ms':{'mixed24':[2.]*5,'local24':[3.]*5},
                  'methods':{m:{'state_bytes_per_doc':24,'upper_violations':0,'strict_missing':0}
                             for m in ['mixed24','local24']}} for i in range(64)]
        sessions=[{'dataset':'fixture','complete':True,'failures':[],'records':copy.deepcopy(records)} for _ in range(3)]
        sessions[1]['records'].reverse()
        z=hybrid_summary(sessions)
        self.assertEqual(z['paired_queries'],64)
        self.assertEqual(z['relative_change'],.5)
        sessions[1]['records'][0]['methods']['local24']['upper_violations']=1
        with self.assertRaises(EvidenceError):hybrid_summary(sessions)


if __name__=='__main__':unittest.main()
