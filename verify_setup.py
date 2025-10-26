"""
Setup Verification Script
Run this script to verify that all packages are installed correctly.
"""

import sys

def verify_package(package_name, import_name=None):
    """Verify that a package can be imported."""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✓ {package_name} installed successfully")
        return True
    except ImportError as e:
        print(f"✗ {package_name} failed to import: {e}")
        return False

def main():
    print("=" * 60)
    print("AdventureWorks Data Science Project - Setup Verification")
    print("=" * 60)
    print()

    packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("statsmodels", "statsmodels"),
        ("prophet", "prophet"),
        ("imbalanced-learn", "imblearn"),
        ("joblib", "joblib"),
        ("jupyter", "jupyter"),
        ("notebook", "notebook"),
        ("ipykernel", "ipykernel"),
    ]

    print("Checking Core Data Science Packages:")
    print("-" * 60)
    results = []

    for package_name, import_name in packages:
        results.append(verify_package(package_name, import_name))

    print()
    print("-" * 60)
    print(f"Results: {sum(results)}/{len(results)} packages installed successfully")
    print("-" * 60)

    if all(results):
        print("✓ All packages installed correctly!")
        print()
        print("You're ready to start your data science project!")
        print()
        print("Next steps:")
        print("1. Place your CSV files in data/raw/")
        print("2. Create notebooks in notebooks/")
        print("3. Start with data exploration and preprocessing")
    else:
        print("✗ Some packages failed to install.")
        print("Please check the errors above and reinstall if needed.")
        print()
        print("To reinstall all packages:")
        print("  pip install -r requirements.txt")
        return 1

    print()
    print("=" * 60)
    print("Project Structure:")
    print("=" * 60)
    print()

    import os
    from pathlib import Path

    project_root = Path(__file__).parent

    print(f"Project Root: {project_root}")
    print()

    directories = [
        "data/raw",
        "data/processed",
        "models/revenue_forecasting",
        "models/churn_prediction",
        "models/return_risk",
        "notebooks",
        "src",
        "outputs/predictions",
        "outputs/visualizations",
        "outputs/reports"
    ]

    print("Directory Structure:")
    for directory in directories:
        path = project_root / directory
        if path.exists():
            print(f"  ✓ {directory}/")
        else:
            print(f"  ✗ {directory}/ (missing)")

    print()
    print("Python Modules:")
    modules = [
        "src/data_preprocessing.py",
        "src/feature_engineering.py",
        "src/model_training.py",
        "src/evaluation.py"
    ]

    for module in modules:
        path = project_root / module
        if path.exists():
            print(f"  ✓ {module}")
        else:
            print(f"  ✗ {module} (missing)")

    print()
    print("=" * 60)
    print("Python Version:", sys.version)
    print("=" * 60)

    return 0

if __name__ == "__main__":
    sys.exit(main())
