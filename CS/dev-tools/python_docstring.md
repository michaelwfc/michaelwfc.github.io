# Google Docstring Format
[Documenting Python Code: A Complete Guide](https://realpython.com/documenting-python-code/)
[PEP 257 – Docstring Conventions](https://peps.python.org/pep-0257/)
[Google docstrings](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings)
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
[Example Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google-style-python-docstrings)

```python
def abc(a: int, c = [1,2]):
    """_summary_

    Args:
        a (int): _description_
        c (list, optional): _description_. Defaults to [1,2].

    Raises:
        AssertionError: _description_

    Returns:
        _type_: _description_
    """
    if a > 10:
        raise AssertionError("a is more than 10")

    return c
```

# Tools:

## vscode with autoDocString

## pycharm 
Editor > Code Style > Python > Docstrings.
Tools > Python Integrate tools > Docstrings > Docstrings Format: Google
