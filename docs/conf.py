#!/usr/bin/env python3

from datetime import datetime
import os
import sys

import guzzle_sphinx_theme

sys.path.extend(os.path.dirname(__file__))
extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "guzzle_sphinx_theme",
]

suppress_warnings = ['image.nonlocal_uri']
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

project = "Spaun"
copyright = "2015-2017, Applied Brain Research"
author = "Applied Brain Research"
version = release = datetime.now().strftime("%Y-%m-%d")
language = None

todo_include_todos = True

intersphinx_mapping = {
    "nengo": ("https://www.nengo.ai/nengo", None)
}

# HTML theming
pygments_style = "sphinx"
templates_path = ["_templates"]
html_static_path = ["_static"]

html_theme_path = guzzle_sphinx_theme.html_theme_path()
html_theme = "guzzle_sphinx_theme"

html_theme_options = {
    "project_nav_name": project,
    "base_url": "https://xchoo.github.io/spaun",
}

# Other builders
htmlhelp_basename = project

latex_elements = {
    # "papersize": "letterpaper",
    # "pointsize": "11pt",
    # "preamble": "",
    # "figure_align": "htbp",
}

latex_documents = [
    (master_doc,  # source start file
     "%s.tex" % project.lower(),  # target name
     "%s Documentation" % project,  # title
     author,  # author
     "manual"),  # documentclass
]

man_pages = [
    # (source start file, name, description, authors, manual section).
    (master_doc, project.lower(), "%s Documentation" % project, [author], 1)
]

texinfo_documents = [
    (master_doc,  # source start file
     project,  # target name
     "%s Documentation" % project,  # title
     author,  # author
     project,  # dir menu entry
     "Spaun",  # description
     "Miscellaneous"),  # category
]
