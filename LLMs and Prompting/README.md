## Files

- `5541_hw5.ipynb`: combined notebook containing **Task 1**, **Task 2**, and **Task 3**
- `csci5541-s26-hw5-ChengFu-3c.json`: Task 3 JSON output file
- `NLP_HW5.pdf`: final PDF report

## Environment

The notebook was designed to run in **Google Colab**.

Recommended runtime:
- Python 3
- GPU runtime (tested on NVIDIA T4 in Colab)

## Dependency Installation

Since this homework was completed in Colab, dependencies are installed inside the notebook using `pip install` cells.

Run the dependency installation cell first. The main packages used are:

```bash
pip install transformers datasets torch pandas accelerate
