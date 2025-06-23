# Create the main Python package structure and core files
import os

# Create directory structure
directories = [
    "decomp_router",
    "decomp_router/core",
    "decomp_router/confidence", 
    "decomp_router/routing",
    "decomp_router/safety",
    "decomp_router/interpretability",
    "experiments",
    "tests",
    "visualizations",
    "datasets",
    "benchmarks",
    "docs"
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

# Create __init__.py files
init_files = [
    "decomp_router/__init__.py",
    "decomp_router/core/__init__.py", 
    "decomp_router/confidence/__init__.py",
    "decomp_router/routing/__init__.py",
    "decomp_router/safety/__init__.py",
    "decomp_router/interpretability/__init__.py"
]

for init_file in init_files:
    with open(init_file, 'w') as f:
        f.write("# DecompRouter package\n")
    print(f"Created: {init_file}")

print("\n✅ Repository structure created successfully!")