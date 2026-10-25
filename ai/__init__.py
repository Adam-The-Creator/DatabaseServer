"""
AI based data analysis and decision support (requirements AI-01..AI-17). See README.md.

An additive module like analysis/ and feedback/: it only reads the existing data, writes nothing but its results
into the file store, and every function can be switched off in config.py (AI-03). Importing the package is cheap: the
heavy libraries (numpy, scikit-learn, PyTorch) are only imported by the function that needs them, so a server without
them still starts and says why a function is unavailable.
"""

VERSION = "1.0"
