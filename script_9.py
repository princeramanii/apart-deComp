# Create setup.py for package installation
setup_py_content = '''"""Setup script for DecompRouter package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="decomp-router",
    version="0.1.0",
    author="Yash Ramani, Prince Ramani, Hisham Hanif",
    author_email="team@decomprouter.dev",
    description="Mechanistic Interpretability for Dynamic Task Decomposition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/decomp-router",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/decomp-router/issues",
        "Documentation": "https://decomp-router.readthedocs.io/",
        "Source Code": "https://github.com/yourusername/decomp-router",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "viz": [
            "streamlit>=1.20.0",
            "gradio>=3.0.0",
            "dash>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "decomp-router=decomp_router.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "decomp_router": ["data/*.json", "models/*.pt"],
    },
    keywords="ai, machine-learning, task-decomposition, model-routing, interpretability, safety",
    zip_safe=False,
)
'''

with open('setup.py', 'w') as f:
    f.write(setup_py_content)

print("✅ Created setup.py")

# Create LICENSE file (MIT License)
license_content = '''MIT License

Copyright (c) 2025 Yash Ramani, Prince Ramani, Hisham Hanif

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

with open('LICENSE', 'w') as f:
    f.write(license_content)

print("✅ Created LICENSE file")