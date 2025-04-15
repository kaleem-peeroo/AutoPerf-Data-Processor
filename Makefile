PYTHON = python3
DATA_DIR = data
TEST_DIR = tests

run:
	$(PYTHON) src/main.py

# Run unit tests
t:
	pytest -vv

# Run unit tests and then run normally
test-run:
	$(PYTHON) -m unittest discover $(TEST_DIR)
	make run

# Clean generated files
clean:
	find . -name "__pycache__" -exec rm -rf {} \;

# Install dependencies
install:
	pip install -r requirements.txt

# Show help
help:
	@echo "Available targets:"
	@echo "  make preprocess [DATASET=<name>] - Preprocess dataset (default: 3pi_no_noise)"
	@echo "  make run						  - Run the model"
	@echo "  make test                        - Run unit tests"
	@echo "  make test-run                    - Run unit tests and then run the code"
	@echo "  make clean                       - Remove generated files"
	@echo "  make install                     - Install dependencies"
	@echo "  make help                        - Show this help"
