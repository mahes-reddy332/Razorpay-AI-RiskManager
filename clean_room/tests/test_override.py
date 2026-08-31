import json
import os
import subprocess
import sys

def test_override_creates_valid_json_with_hash(tmp_path):
    override_file = str(tmp_path / "test_overrides.json")
    
    # Run CLI to create first record
    cmd1 = [
        sys.executable, "src/human_override.py",
        "--account", "ACC_TEST_001",
        "--decision", "SAFE",
        "--note", "Verified via customer call",
        "--output-file", override_file
    ]
    res1 = subprocess.run(cmd1, capture_output=True, text=True, cwd=".")
    assert res1.returncode == 0, f"Error: {res1.stderr}"
    
    # Run CLI to create second record (tests hash chaining)
    cmd2 = [
        sys.executable, "src/human_override.py",
        "--account", "ACC_TEST_002",
        "--decision", "HIGH_RISK",
        "--note", "Confirmed syndicate node",
        "--output-file", override_file
    ]
    res2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=".")
    assert res2.returncode == 0, f"Error: {res2.stderr}"
    
    assert os.path.exists(override_file)
    with open(override_file, "r") as f:
        data = json.load(f)
        
    assert "records" in data
    assert len(data["records"]) == 2
    assert data["records"][0]["account_id"] == "ACC_TEST_001"
    assert data["records"][1]["account_id"] == "ACC_TEST_002"
    # Verify hash chain linkage
    assert data["records"][1]["previous_hash"] == data["records"][0]["hash"]

def test_override_tamper_evidence(tmp_path):
    override_file = str(tmp_path / "test_tamper.json")
    
    # Create records
    subprocess.run([sys.executable, "src/human_override.py", "--action", "record", "--account", "ACC_001", "--decision", "SAFE", "--note", "test", "--output-file", override_file], check=True)
    subprocess.run([sys.executable, "src/human_override.py", "--action", "record", "--account", "ACC_002", "--decision", "HIGH_RISK", "--note", "test", "--output-file", override_file], check=True)
    
    # Verify should pass
    res_valid = subprocess.run([sys.executable, "src/human_override.py", "--action", "verify", "--output-file", override_file], capture_output=True)
    assert res_valid.returncode == 0
    
    # Tamper with the first record
    with open(override_file, "r") as f:
        data = json.load(f)
    
    data["records"][0]["override_decision"] = "HIGH_RISK"
    
    with open(override_file, "w") as f:
        json.dump(data, f)
        
    # Verify should fail
    res_invalid = subprocess.run([sys.executable, "src/human_override.py", "--action", "verify", "--output-file", override_file], capture_output=True)
    assert res_invalid.returncode != 0
