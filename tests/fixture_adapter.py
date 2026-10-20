"""Test-only transport stub returning literal fixture values, not a reconstruction."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
request = json.loads(Path(args.request).read_text())
fixture = json.loads((Path(__file__).parent / "fixtures" / "protocol_toy.json").read_text())
metadata = {"case_id": request["case_id"], "method": "hand-authored-test-fixture", "missing_pose_policy": "fixture-only"}
np.savez_compressed(args.output, volume=np.array(fixture["volume"], dtype=np.uint8), index_to_world_mm=np.array(fixture["index_to_world_mm"], dtype=float), metadata_json=np.array(json.dumps(metadata)))
