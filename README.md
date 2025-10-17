# Standing Wave Spacetime Project

This repository contains the LaTeX source, figures, and simulation code for the *Discrete Standing Wave Spacetime* paper.

---

## 🛠️ Canonical Build

You may use any TeX editor (TeXShop, TeXworks, Overleaf, etc.) for local development, but the **official build** of the paper is always generated with `latexmk` to ensure consistency.  

From the repository root, run:

```bash
latexmk -pdf -interaction=nonstopmode -output-directory=build src/standing_wave.tex
```

### VS Code Build Task (Alternative)

For convenience, you can also build the paper directly from VS Code using the preconfigured build task.  
This task runs the same `latexmk` process as the command line and has been set as the default build target.  

To use it, press **Ctrl+Shift+B** (Windows/Linux) or **⌘+Shift+B** (Mac) and select the **"Build LaTeX PDF"** task.  

This produces `build/standing_wave.pdf` just like the command line method.

---

## 📄 Outputs

- [Final paper (PDF)](build/standing_wave.pdf)  
- [Formation animation (MP4)](build/black_surface_formation.mp4)

---

## 📂 Directory Structure

- `src/`  
  Python source code, virtual environment, and LaTeX source (`standing_wave.tex`, `references.bib`).
  
  The python file is used for generating the complete video showing the formation of the Black Surface from flat space and for exporting a set of individual frames used in the PDF output.

- `figures/`  
  Generated plots (PDF format) for inclusion in the LaTeX document.  

- `build/`  
  Build artifacts such as the compiled PDF of the paper and the MP4 animation.  

---

## ⚙️ Setup

Create and activate a local Python virtual environment:

On Mac:
```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:
```bash
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

- Verify the Python path:

```powershell
Get-Command python
Get-Command pip
```

- Confirm packages:

```powershell
pip list
```

Install the project python modules
```powershell
cd src
pip install -r requirements.txt
```
