project = "mini-gpr"
copyright = "2024-2025, John Gardner"
author = "John Gardner"
release = "0.0.0"

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    # "jaxtyping": ("https://jaxtyping.readthedocs.io/en/latest/", None),
}

html_title = "mini-gpr"
html_theme = "furo"
html_static_path = ["_static"]
autodoc_member_order = "bysource"
maximum_signature_line_length = 50
autodoc_typehints = "description"

# override the default css to match the furo theme
nbsphinx_prolog = """
.. raw:: html

    <style>
        .jp-RenderedHTMLCommon tbody tr:nth-child(odd),
        div.rendered_html tbody tr:nth-child(odd) {
            background: var(--color-code-background);
        }
        .jp-RenderedHTMLCommon tr,
        .jp-RenderedHTMLCommon th,
        .jp-RenderedHTMLCommon td,
        div.rendered_html tr,
        div.rendered_html th,
        div.rendered_html td {
            color: var(--color-content-foreground);
        }
        .jp-RenderedHTMLCommon tbody tr:hover,
        div.rendered_html tbody tr:hover {
            background: #3c78d8aa;
        }
        div.nbinput.container div.input_area {
            /* border radius of 10px, but no outline */
            border-radius: 10px;
            border-style: none;
        }
        div.nbinput.container div.input_area > div.highlight > pre {
            padding: 10px;
            border-radius: 10px;
        }

    </style>
"""
