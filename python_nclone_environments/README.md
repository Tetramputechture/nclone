# N-Clone Python Environments

This project provides Python environments for interacting with N-Clone levels, primarily for testing and reinforcement learning agent development. It may utilize the `nplay_headless_cpp` bindings from the main `nclone-cpp` repository.

## Project Structure

It's assumed you have a project structure like this:

/your_project_root_directory/  (e.g., /home/tetra/projects/nclone/)
|-- test_environment.py
|-- setup.py  <-- This setup file we created
|-- README.md <-- This README file
|-- nclone_environments/
|   |-- __init__.py
|   |-- basic_level_no_gold/
|   |   |-- __init__.py
|   |   |-- basic_level_no_gold.py
|   |-- ... (other environments)
|-- (Potentially the nclone-cpp repository cloned here or elsewhere)

## Setup and Installation

### Prerequisites

- Python 3.6+
- Pygame (and any other dependencies listed in `setup.py`)

### Running `test_environment.py` Directly

The `test_environment.py` script is designed to be run directly. It includes logic to modify `sys.path` so that it can find the `nclone_environments` package.

1.  **Navigate to the project root directory** (e.g., `/home/tetra/projects/nclone/`):
    ```bash
    cd /path/to/your_project_root_directory
    ```
2.  **Ensure `nclone_environments` is present** in this directory.
3.  **Run the script:**
    ```bash
    python test_environment.py
    ```

### Installing as a Package (for external use)

If you want to import `nclone_environments` from another Python project, you can install it as a package.

1.  **Navigate to the project root directory** (the one containing `setup.py`, `test_environment.py`, and the `nclone_environments` folder).
    ```bash
    cd /path/to/your_project_root_directory
    ```
2.  **Install the package:**
    For development (editable install):
    ```bash
    pip install -e .
    ```
    For a regular install:
    ```bash
    pip install .
    ```

    This will install the `nclone_environments` package into your Python environment.

3.  **Using in another project:**
    You can then import it in your Python code:
    ```python
    from nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
    # or other environments
    ```

## Explanation of Changes for `ImportError`

The original `ImportError: attempted relative import with no known parent package` in `test_environment.py` occurred because:
1.  The script was run as the top-level module (e.g., `python test_environment.py`).
2.  It used a relative import (`from .nclone_environments...`). Relative imports are for modules within the same package. When a file is run directly, it doesn't belong to a package, so `.` has no meaning.

The fix involves:
1.  **Adding `__init__.py` files:** Empty `__init__.py` files were added to `nclone_environments/` and `nclone_environments/basic_level_no_gold/` to ensure they are treated as Python packages.
2.  **Modifying `test_environment.py`:**
    *   The import was changed to an absolute import: `from nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold`.
    *   Code was added at the beginning of `test_environment.py` to add its own directory (the project root) to `sys.path`. This allows Python to find the `nclone_environments` package when the script is run from any location.
3.  **Adding `setup.py`:** This file makes the `nclone_environments` package installable, which is the standard way to make Python code reusable across projects.
