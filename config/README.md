# config 
## architecture
The config architecture in this project has three aspect.
1. `config/defaults.py` file. In this file, these are base config for all models. These configs are using `yacs` module to manage.
2. `experiment/` folder. In this folder, config files are set in every folders named by model name.
3. `config/config.py` file. In this file, some configs are parsed using `argparse` model for simplicity.
## note
There are some important things you should know.
1. The config has priority. The priority order by descend is command line, `experiment/model/base.yaml`, `config/defaults.py` and `config/config.py`. Which means if your config in higher priority manner is different from lower manner, the config in higher priority manner will take effect.
2. If you set `--resume`, the program will automatically load last model. If you set `model.weights model/path`, the program will load model at `model/path`. If you set these two configs simultaneously, `--resume` config will take effect (as described in 1).