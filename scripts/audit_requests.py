"""Print the host-only action audit for one run directory."""

import argparse
import json

from saas_bench.evaluation.request_audit import audit_run


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", help="Experiment run directory")
args = parser.parse_args()
print(json.dumps(audit_run(args.run_dir), ensure_ascii=False, indent=2))
