# PyCharm notes

Open the repository root, not the `notebooks` folder by itself.

Use one interpreter/kernel for both Python files and notebooks. In recent PyCharm versions, open an `.ipynb`, select the project interpreter as the Jupyter kernel, and run cells normally.

The notebooks add `src/` to `sys.path` themselves, so no editable install is required. If you prefer one:

```bash
pip install -e .
```
