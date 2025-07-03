![HistoMSC](https://github.com/user-attachments/assets/52d786dd-9e8f-43b6-a900-5b843bac3b11)

# HistoMSC: Morse-Smale Complex Analysis for Histopathology Images


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14510390.svg)](https://doi.org/10.5281/zenodo.14510390)


HistoMSC is a computational pipeline for analyzing tissue interactions in histopathology images using Morse-Smale Complex theory. This tool enables the identification and quantification of tissue interface patterns, particularly useful for studying tumor-stroma interactions in cancer research.

## Features

- Automated tissue type detection and classification
- Morse-Smale Complex based topological analysis
- Interface pattern quantification
- Compatible with QuPath for visualization
- Supports whole slide imaging (WSI) analysis

## Installation

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/yourusername/HistoMSC.git
cd HistoMSC
```

2. Download model files:
   - Download the required model files from [Zenodo](https://doi.org/10.5281/zenodo.14510390)
   - Extract and place model files in their respective directories under `models/`
   - See README files in each model directory for specific file placement instructions

3. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate histomsc
```

## Directory Structure

```
HistoMSC/
├── config_msc.json         # Configuration file
├── histo_msc.py           # Main script
├── models/                # Pre-trained models (download from Zenodo)
│   ├── point/            # Point detection model
│   ├── squeezenet/       # SqueezeNet model
│   └── yolo/             # YOLO model
├── morse_smale_src/       # Morse-Smale Complex implementation (submodule)
└── test/                  # Test data and examples
    ├── *.json            # Example annotation files
    ├── *.csv             # Example results
    └── *.svs             # Test WSI (download from Zenodo)
```

## Usage

1. Configure analysis parameters in `config_msc.json`
2. Run the analysis:
```bash
python histo_msc.py --image path/to/image.svs --output path/to/output
```

## Viewing Results in QuPath

1. Open QuPath
2. Import the WSI file
3. File -> Import -> GeoJSON
4. Select the generated JSON files:
   - `*_ann.json`: Tissue annotations
   - `*_sig_*.json`: Interface signatures
   - `*_msc.json`: MSC critical points
   - `*_cont.json`: Contour information

## Output Files

- `*_ann.json`: Tissue type annotations
- `*_sig_*.json`: Interface signature files
- `*_msc.json`: MSC critical points
- `*_cont.json`: Tissue contours
- `*_cp.json`: Critical points
- `*_inf.csv`: Interface metrics

## Test Data and Models

### Pre-trained Models
The following model files are hosted on Zenodo due to their size:
- Point detection model (`models/point/export.pkl`)
- SqueezeNet model (`models/squeezenet/SqueezeNetPanNuke-inpainted-blur0.001-192-off8.pth`)
- YOLO model (`models/yolo/best.pt`)

Download these files from: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14510390.svg)](https://doi.org/10.5281/zenodo.14510390)

### Test Data
The `test/` directory contains:
- Example WSI file (download from Zenodo)
- Example annotation files and analysis results
- See `test/README.md` for detailed information

## Dependencies

### Morse-Smale Complex Implementation
The Morse-Smale Complex implementation is from the [morse_smale](https://github.com/uncommoncode/morse_smale) repository by uncommoncode. It is included as a Git submodule in the `morse_smale_src` directory.

To update the submodule after cloning:
```bash
git submodule update --init --recursive
```
## Demonstration Video
[![HistoMSC Demo](https://img.youtube.com/vi/KkOd9mL6X2E/0.jpg)](https://www.youtube.com/watch?v=KkOd9mL6X2E)

Watch our [demonstration video](https://youtu.be/KkOd9mL6X2E) to see HistoMSC in action! The video showcases:
- AI-based WSI processing and analysis
- Visual annotations and tissue classification
- Morse-Smale Complex analysis of tissue interfaces
- Interface pattern quantification and metrics
- QuPath integration and result visualization
## Citation

If you use this tool in your research, please cite:
@article{AHMAD2025109991,
title = {HistoMSC: Density and topology analysis for AI-based visual annotation of histopathology whole slide images},
journal = {Computers in Biology and Medicine},
volume = {190},
pages = {109991},
year = {2025},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2025.109991},
url = {https://www.sciencedirect.com/science/article/pii/S0010482525003427},
author = {Zahoor Ahmad and Khaled Al-Thelaya and Mahmood Alzubaidi and Faaiz Joad and Nauman Ullah Gilal and William Mifsud and Sabri Boughorbel and Giovanni Pintore and Enrico Gobbetti and Jens Schneider and Marco Agus},
keywords = {Histopathology, AI-based annotation, Nuclei localization, Kernel density estimation, Morse–Smale complex},
abstract = {We introduce an end-to-end framework for the automated visual annotation of histopathology whole slide images. Our method integrates deep learning models to achieve precise localization and classification of cell nuclei with spatial data aggregation to extend classes of sparsely distributed nuclei across the entire slide. We introduce a novel and cost-effective approach to localization, leveraging a U-Net architecture and a ResNet-50 backbone. The performance is boosted through color normalization techniques, helping achieve robustness under color variations resulting from diverse scanners and staining reagents. The framework is complemented by a YOLO detection architecture, augmented with generative methods. For classification, we use context patches around each nucleus, fed to various deep architectures. Sparse nuclei-level annotations are then aggregated using kernel density estimation, followed by color-coding and isocontouring. This reduces visual clutter and provides per-pixel probabilities with respect to pathology taxonomies. Finally, we use Morse–Smale theory to generate abstract annotations, highlighting extrema in the density functions and potential spatial interactions in the form of abstract graphs. Thus, our visualization allows for exploration at scales ranging from individual nuclei to the macro-scale. We tested the effectiveness of our framework in an assessment by six pathologists using various neoplastic cases. Our results demonstrate the robustness and usefulness of the proposed framework in aiding histopathologists in their analysis and interpretation of whole slide images.}
}

## License

[Add license information]
