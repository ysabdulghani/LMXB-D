# LMXB-D: A Low Mass Black Hole X-Ray Binary Distance Estimator

Calculate mass/distance probabilities based on the probability distribution (MCMC distribution) of the soft state model `ezdskbb` normalization and the soft-to-hard transition period power law flux in 0.5 to 200 keV range using `cflux`. It uses the statistical framework prescriped in [Abdulghani et. al. 2004](https://link-url-here.org).

## Getting Started

These instructions will get you a copy of the script running on your local machine.

### Prerequisites

What you need to install the software:

- Python 3
- NumPy
- Matplotlib
- PyTorch
- SciPy
- H5py
- Boost Histogram

### Installation

Can just grab download the script if you have all the prerequisites

Clone the repository and install the required packages.

```bash
git clone https://github.com/ysabdulghani/lmxbd/
cd lmxbd
pip install numpy matplotlib torch scipy h5py boost-histogram
```

### Usage

Generate MCMC distrubtion of normalization of `ezdiskbb` in xspec, ensure that your model follows this notation (order matters):
```
any_absorption_model*(powerlaw+ezdiskbb+any_other_model_components)
```
Generate MCMC distribution of the soft-to-hard transition flux using `cflux`, ensure your energy range is (0.5-200 keV) and that you are only using one cflux component (the order does not matter).

##Using script

positional arguments:
  chainFilenames  Input .txt file containing the chain filenames in
                  separate lines (include subdirectory if not in same
                  path). Must be generated using xspec chain command

optional arguments:
  -h, --help      show this help message and exit
  --softonly      Flag to calculate for soft state only
