# Third-party data notice

The MIT licence in `LICENSE` covers this project's own source code and documentation only.
The project does **not** claim ownership of any PhysioNet database or of the libraries it depends on.

## ECG databases

The V4 model was trained on scalograms derived from two PhysioNet databases. Both are made available under the
**Open Data Commons Attribution License v1.0 (ODC-By 1.0)**, <https://opendatacommons.org/licenses/by/1-0/>,
which permits redistribution and derived databases with attribution.

| Database | Records used | Version / DOI | Licence |
|---|---|---|---|
| MIT-BIH Arrhythmia Database (`mitdb`) | all 48 records | 1.0.0, <https://doi.org/10.13026/C2F305> | ODC-By 1.0 |
| MIT-BIH ST Change Database (`stdb`) | 27 records, 301–327 (record 300 not used) | 1.0.0, <https://doi.org/10.13026/C2ZW2H> | ODC-By 1.0 |

The original project code stored the `stdb` records in a folder named "INCART". All 27 local header files are
byte-identical to the official `stdb` headers (see `docs/DATASETS.md`). The St Petersburg INCART database was
**not** used by the V4 model.

### Files in this repository derived from PhysioNet data

| Files | Source |
|---|---|
| `examples/beats/*.csv` (5 beats) | `mitdb` records 100, 107, 119, 208, 232 |
| `examples/scalograms/*.png` (10 images) | `mitdb` records 102, 104, 105, 201, 213, 221, 232 (exact record per file in `examples/manifest.json`) |
| `docs/assets/colour_scalogram_example.png` | Jet-colormap rendering of `examples/scalograms/N_idx1875.png` (`mitdb` record 105), used only to demonstrate rejection of colour input |
| Screenshots in `screenshots/` and outputs saved in `notebooks/ECG_CWT_CNN_V4_Pipeline.ipynb` | Computed from `mitdb` and `stdb` |

**Contains information from the MIT-BIH Arrhythmia Database and the MIT-BIH ST Change Database, which are made
available under the ODC Attribution License.**

Raw records and the full generated datasets are **not** redistributed here.

### Citations

- Moody, G., & Mark, R. (2005). *MIT-BIH Arrhythmia Database* (version 1.0.0). PhysioNet. https://doi.org/10.13026/C2F305
- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Eng in Med and Biol* 20(3):45–50 (May–June 2001).
- Albrecht P. S-T segment characterization for long-term automated ECG analysis. M.S. thesis, MIT Dept. of Electrical
  Engineering and Computer Science, 1983. (*MIT-BIH ST Change Database*, version 1.0.0, PhysioNet,
  https://doi.org/10.13026/C2ZW2H)
- Pollard T, Moody BE, Lehman L, Gow B, Fernandes C, Xie C, Johnson A, Mark RG, Heldt T. PhysioNet as a global platform
  for biomedical research. *Nature Health* 1(8):792–795 (2026). https://doi.org/10.1038/s44360-026-00096-z
- Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE.
  PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.
  *Circulation* 101(23):e215–e220 (2000).

## Software

Runtime dependencies (TensorFlow, Keras, Gradio, NumPy, pandas, Pillow, PyWavelets, OpenCV) are installed from
PyPI under their own licences and are not vendored in this repository.
