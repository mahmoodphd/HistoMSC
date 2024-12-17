
![HistoMSC](https://github.com/user-attachments/assets/52d786dd-9e8f-43b6-a900-5b843bac3b11)







# HistoMSC: Morse-Smale Complex Analysis for Histopathology Images

HistoMSC is a computational pipeline for analyzing tissue interactions in histopathology images using Morse-Smale Complex theory. This tool enables the identification and quantification of tissue interface patterns, particularly useful for studying tumor-stroma interactions in cancer research.

## Features

- Automated tissue type detection and classification
- Morse-Smale Complex based topological analysis
- Interface pattern quantification
- Compatible with QuPath for visualization
- Supports whole slide imaging (WSI) analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HistoMSC.git
cd HistoMSC
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate histomsc
```

## Directory Structure

```
HistoMSC/
├── config_msc.json         # Configuration file
├── histo_msc.py           # Main script
├── models/                # Pre-trained models
│   ├── point/
│   ├── squeezenet/
│   └── yolo/
├── morse_smale_src/       # MSC implementation
└── test/                  # Test data and examples
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

## Citation

If you use this tool in your research, please cite:
[Add citation information]

## License

[Add license information]
