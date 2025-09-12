# Standing Wave Spacetime Project

This repository contains the LaTeX source, figures, and simulation code for the *Discrete Standing Wave Spacetime* paper.

---

## Directory Structure

- `src/`  
  Python source code, virtual environment, and build scripts.  
- `figures/`  
  Generated plots (PDF format) for inclusion in the LaTeX document.  
- `build/`  
  Build artifacts such as the compiled PDF of the paper and the MP4 animation.  
- `standing_wave.tex`  
  Main LaTeX source.  
- `references.bib`  
  Bibliography file.  

---

## Setup

Create and activate a local Python virtual environment:

```bash
cd src
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
