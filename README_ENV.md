# vSAVE Environment (Frozen)

## Setup (reproducible)
```bash
./env_setup_vsave.sh

Verify lock parity

source .venv/bin/activate
pip freeze | sort | diff -u requirements.lock.sorted.txt - | head

Artifacts
	•	requirements.lock.txt: source of truth
	•	requirements.now.txt: snapshot of current environment
	•	env_report.txt: python/pip summary
