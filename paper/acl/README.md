# ACL paper

Title: **Rainbow Cookie at VLSP 2026: Small Models, Long Laws**.

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

Formatting rechecked on 2026-10-27 against the official ACL template and
[ACLPUB guidelines](https://acl-org.github.io/ACLPUB/formatting.html): both style
files match upstream byte for byte; seven A4 pages; default ACL geometry,
two-column layout, caption and bibliography styles; all PDF fonts embedded;
no unresolved citations/references or overfull boxes. The appendix uses the
default ACL page alignment. Both vector diagrams use the full text width for
legibility. All seven rendered pages were visually checked. The file remains in
`final` mode, with authors visible and no page or review line numbers.

The revised paper has seven pages including references and appendices, with all
major sections retained. The reported score/cost tables and the two figure PDFs
are preserved. Prose was condensed and implementation descriptions and formulas
were checked against the local training and inference code.

See [VERIFICATION.md](VERIFICATION.md) for the audit of all 23 original references,
the 20 retained citations, formula checks and the provenance of the added Index
training configuration. All retained sources were verified against primary
records. The document-count change from 575 to 612 is now verified; the origin
of the internal validation partition still needs clarification. Author email
addresses have been supplied and added in the original author order. The
main-table baseline/Index scores are retained from the original paper by author
decision; their run provenance remains incomplete. Saved predictions were rescored on CPU and
document-bootstrap intervals recomputed; no model inference or training was run.
See verification/CLAIM_AUDIT.md for claim-level source checks and the remaining
run-provenance gaps.
