# Contributing

Thank you for considering a contribution! Here's everything you need to get started.

---

## Getting Started

```bash
git clone https://github.com/your-username/python-yolo-demo.git
cd python-yolo-demo
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

---

## How to Contribute

### Bug Reports
Open an issue using the **Bug Report** template. Include:
- OS and Python version
- Steps to reproduce
- Expected vs. actual behaviour
- Any error messages or tracebacks

### Feature Requests
Open an issue using the **Feature Request** template describing the use case and proposed solution.

### Pull Requests
1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
2. Keep changes focused — one feature or fix per PR.
3. Follow the existing code style (PEP 8, docstrings, type hints).
4. Test manually before submitting (no test suite yet — contributions welcome!).
5. Update `CHANGELOG.md` under `[Unreleased]`.
6. Open the PR against `main` and fill in the template.

---

## Code Style

- PEP 8 compliance; `ruff` is configured in `pyproject.toml`.
- Type hints on all public functions and methods.
- Docstrings on all public classes and methods.
- Keep `app.py` (GUI) and `detector.py` (inference) decoupled — no GUI imports in `detector.py`.

---

## Code of Conduct

Be respectful and constructive. Harassment or discrimination of any kind will not be tolerated.
