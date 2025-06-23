# Contributing to DecompRouter

Thank you for your interest in contributing to DecompRouter! This document outlines the process for contributing to our Enhanced ADaPT framework with mechanistic interpretability.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/decomp-router.git
   cd decomp-router
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

## 🔧 Development Setup

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=decomp_router

# Run specific test file
pytest tests/test_confidence_scorer.py
```

### Code Quality
We use several tools to maintain code quality:

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy decomp_router/

# Run all quality checks
pre-commit run --all-files
```

### Running Examples
```bash
# Run the demo notebook
jupyter lab experiments/demo_notebook.ipynb

# Run basic example
python -m decomp_router.examples.basic_usage
```

## 📋 Contribution Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Found a bug? Please open an issue with detailed reproduction steps
2. **Feature Requests**: Have an idea for improvement? Share it in our discussions
3. **Code Contributions**: Bug fixes, new features, performance improvements
4. **Documentation**: Improve docs, add examples, write tutorials
5. **Research**: Contribute new interpretability techniques or evaluation methods

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards:
   - Write clear, documented code
   - Add tests for new functionality
   - Update documentation as needed
   - Follow PEP 8 style guidelines

3. **Test your changes**:
   ```bash
   pytest
   black --check .
   flake8 .
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new confidence scoring factor"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub with:
   - Clear description of changes
   - Link to related issues
   - Screenshots/examples if applicable

### Coding Standards

#### Python Code Style
- Follow PEP 8
- Use type hints for all functions
- Write docstrings for all public methods
- Keep functions focused and under 50 lines when possible

#### Example:
```python
def calculate_confidence(self, task_prompt: str, context: str = "") -> ConfidenceAnalysis:
    """
    Calculate multi-factor confidence score for a task.
    
    Args:
        task_prompt: The task to analyze
        context: Additional context for analysis
        
    Returns:
        ConfidenceAnalysis with detailed breakdown
        
    Raises:
        ValueError: If task_prompt is empty
    """
    if not task_prompt.strip():
        raise ValueError("Task prompt cannot be empty")
    
    # Implementation here
    pass
```

#### Testing Standards
- Write unit tests for all new functions
- Use pytest fixtures for common setup
- Aim for >90% test coverage
- Include both positive and negative test cases

#### Example:
```python
def test_confidence_scorer_basic():
    """Test basic confidence scoring functionality."""
    scorer = ConfidenceScorer()
    
    result = scorer.calculate("Analyze climate data")
    
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.breakdown) == 7
    assert result.uncertainty >= 0.0
```

## 🧠 Research Contributions

### Mechanistic Interpretability
If you're contributing interpretability research:

1. **Document your approach** with mathematical formulations
2. **Provide empirical validation** on standard benchmarks
3. **Include visualization tools** for analysis
4. **Write clear explanations** of the mechanistic insights

### Model Routing Algorithms
For routing algorithm contributions:

1. **Benchmark against existing methods** on cost and performance
2. **Provide theoretical analysis** when possible
3. **Include safety considerations** in your design
4. **Test on diverse task types**

## 📚 Documentation

### Code Documentation
- Use Google-style docstrings
- Include examples in docstrings when helpful
- Document complex algorithms with inline comments
- Update README.md for major changes

### Research Documentation
- Add papers to `docs/references.bib`
- Include algorithm descriptions in `docs/algorithms/`
- Provide notebooks demonstrating new features
- Write blog posts for significant contributions

## 🛡️ Safety and Ethics

### Safety Requirements
All contributions must:
- Include safety validation for new features
- Pass existing safety tests
- Consider potential misuse scenarios
- Document safety implications

### Ethical Guidelines
- Respect user privacy in all features
- Avoid implementing harmful capabilities
- Consider bias implications of routing decisions
- Include fairness considerations in evaluations

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear description** of the issue
2. **Reproduction steps** with minimal example
3. **Expected vs actual behavior**
4. **Environment details** (Python version, OS, etc.)
5. **Error logs** and stack traces

### Bug Report Template
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Import DecompRouter
2. Call function with parameters...
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.9.0]
- DecompRouter version: [e.g. 0.1.0]

**Additional context**
Any other context about the problem.
```

## 💡 Feature Requests

For feature requests:

1. **Check existing issues** to avoid duplicates
2. **Describe the use case** and motivation
3. **Propose implementation approach** if you have ideas
4. **Consider backward compatibility**
5. **Discuss in GitHub Discussions** first for major features

## 🏆 Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Invited to co-author research papers for significant contributions
- Given commit access for sustained contributions

## 📞 Getting Help

- **GitHub Discussions**: For questions and brainstorming
- **GitHub Issues**: For bug reports and feature requests
- **Email**: team@decomprouter.dev for private inquiries

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make DecompRouter better! 🎉