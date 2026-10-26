# ACL paper

Source: `paper_en.tex`; compiled paper: `paper_en.pdf`.

The unmodified `acl.sty` and `acl_natbib.bst` come from the official
[ACL style repository](https://github.com/acl-org/acl-style-files), retrieved on
2026-10-26. The bibliography is maintained in `references.bib`.

Build from this directory with pdfLaTeX and BibTeX:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error paper_en.tex
```

The source uses `\usepackage[final]{acl}` to retain the existing author names.
For an anonymous review submission, change `final` to `review` and rebuild.
Use the venue's call for papers to determine the required mode and page limit.

Scientific content, results, and the two figure PDFs are preserved from the
previous manuscript. Existing incomplete bibliography metadata is marked with
`CHECK` comments in `references.bib`; the Index training settings still need
completion in the hyperparameter table. Author email addresses are also pending.
