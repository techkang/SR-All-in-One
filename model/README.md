# add your new model
If you want to add your own model, you can follow these steps:
1. create a file in `model` folder and implement your model.
2. in `model/__init__.py`, append a line which is `from xxx import XX`.
3. in `config/defaults.py`, add your model's config.
4. in  `experiment` folder, create a folder named by your model, and create a `base.yaml` in that folder.
5. run `python train.py  --config-file experiment/yourmodel/base.yaml` in project folder.

