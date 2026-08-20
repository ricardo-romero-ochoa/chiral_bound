PYTHON ?= python

.PHONY: test bootstrap milestone1 milestone2 milestone3 validate production manifest clean

test:
	PYTHONPATH=src $(PYTHON) -m pytest

bootstrap:
	PYTHONPATH=src $(PYTHON) scripts/milestone3_bootstrap.py

milestone1:
	PYTHONPATH=src $(PYTHON) scripts/milestone1_audit.py
	PYTHONPATH=src $(PYTHON) scripts/milestone1_oligomer_grid.py

milestone2:
	PYTHONPATH=src $(PYTHON) scripts/milestone2_audit.py

milestone3: bootstrap
	PYTHONPATH=src $(PYTHON) scripts/milestone3_audit.py

validate: test milestone1 milestone2 milestone3

production:
	PYTHONPATH=src $(PYTHON) scripts/milestone3_generate.py
	PYTHONPATH=src $(PYTHON) scripts/milestone3_ssa_verify.py
	PYTHONPATH=src $(PYTHON) scripts/milestone3_bootstrap.py

manifest:
	$(PYTHON) scripts/build_manifest.py

clean:
	rm -f MANIFEST.sha256
