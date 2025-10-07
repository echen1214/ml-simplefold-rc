# esm trainer

## lighting studio setup for training

- clone this repo to the studio workspace
- follow instructions in the [simplefold README](../../README.md):
  conda is installed by default in pytorch studios, but does not allow creating new conda environments. just run the pip install commands
- clone the alignbio data repo:
```
cd ~
git clone git@github.com:Align-to-Innovate/the-protein-engineering-tournament-2023
```

- create esm embedding cache dir:
```
mkdir -p ~/data/esm-cache
```

- **Run training with the Lightning Studio config:**
   ```bash
   cd /teamspace/studios/this_studio/ml-simplefold-rc/src/esm
   python train.py --config-name train_lightning_studio
   ```
