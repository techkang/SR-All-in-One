import multiprocessing

from config import CfgNode as CN

_C = CN()

_C.output_dir = ''
_C.trainer = 'Trainer'

# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.cudnn_benchmark = False
_C.device = 'cuda'
_C.num_gpus = 1
_C.upscale_factor = 4
_C.start_iter = 0

_C.model = CN()
_C.model.name = 'DPDNN'
_C.model.weights = ''
_C.model.pixel_mean = [103.530, 116.280, 123.675]
_C.model.in_channels = 3
_C.model.out_channels = 3

_C.datasets = CN()
_C.datasets.path = 'dataset'
_C.datasets.train_folder = 'train_HR'
_C.datasets.train = []
_C.datasets.test = []
_C.datasets.interpolation = 'bicubic'
_C.datasets.input_size = 32
_C.datasets.gray = False
_C.datasets.re_generate = False

_C.dataloader = CN()
_C.dataloader.num_workers = multiprocessing.cpu_count()
_C.dataloader.batch_size = 32

# DPDNN settings
_C.dpdnn = CN()
_C.dpdnn.iteration = 6

_C.dbpn = CN()
_C.dbpn.num_features = 64
_C.dbpn.num_blocks = 7
_C.dbpn.norm_type = False
_C.dbpn.active = 'prelu'

_C.edsr = CN()
_C.edsr.num_features = 256
_C.edsr.num_blocks = 32
_C.edsr.res_scale = 0.1

_C.srfbn = CN()
_C.srfbn.num_features = 64
_C.srfbn.num_steps = 4
_C.srfbn.num_groups = 6
_C.srfbn.active = 'prelu'
_C.srfbn.norm_type = False

_C.rdn = CN()
_C.rdn.num_features = 64
_C.rdn.num_blocks = 16
_C.rdn.num_layers = 8

_C.meta = CN()
_C.meta.backbone = 'RDN'

_C.rcan = CN()
_C.rcan.n_resgroups = 10
_C.rcan.n_resblocks = 20
_C.rcan.n_feats = 64

_C.mirnet = CN()
_C.mirnet.num_features = 64
_C.mirnet.num_blocks = 2
_C.mirnet.num_groups = 3

_C.optimizer = CN()
_C.optimizer.name = 'Adam'
_C.optimizer.lr = 0.0001

_C.lr_scheduler = CN()
_C.lr_scheduler.step_size = 5000
_C.lr_scheduler.gamma = 0.5

_C.solver = CN()
_C.solver.max_iter = 30000
_C.solver.save_interval = 1000
_C.solver.test_interval = 500
_C.solver.ohem_k = 0
_C.solver.loss = 'l1_loss'
_C.solver.bias_loss = True
_C.solver.bias_weight = [0.299, 0.587, 0.114]

_C.tensorboardX = CN()
_C.tensorboardX.clear_before = True
_C.tensorboardX.save_freq = 100
_C.tensorboardX.image_num = 2
_C.tensorboardX.name = ''
