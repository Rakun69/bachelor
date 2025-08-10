Place LaTeX sources under thesis_sections/, with subfolders:
- img/: figures
- misc/: titlepage.tex, self-assertion.tex, etc.
- bib/references.bib

Adjust in bachelorarbeit.tex:
- \graphicspath{{./img/}{../data/visualizations/}}
- \addbibresource{bib/references.bib}
- \input{misc/titlepage}
- \input{misc/self-assertion}

