import logging
import config
from utils import utils
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import os
from dataloader.data import get_dataset
from torch.utils.data import DataLoader
from approaches.train import Appr

logger = logging.getLogger(__name__)

args = config.parse_args()
args = utils.prepare_sequence_train(args)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(mixed_precision=args.mixed_precision,
                          fp16=args.fp16, kwargs_handlers=[ddp_kwargs])

# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# If passed along, set the training seed now.
if args.seed is not None:
    set_seed(args.seed)

if accelerator.is_main_process:
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
accelerator.wait_for_everyone()

dataset = get_dataset(args)
model = utils.lookfor_model(args)

train_loader = DataLoader(dataset[args.task]['train'], batch_size=args.batch_size, shuffle=True, num_workers=8)
test_loaders = []
replay_loaders = []

for eval_t in range(args.task + 1):
    test_dataset = dataset[eval_t]['test']
    #replay_dataset = dataset[eval_t]['replay']
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    #replay_dataloader = DataLoader(replay_dataset, batch_size=args.batch_size)
    test_loaders.append(test_dataloader)
    #replay_loaders.append(replay_dataloader)

appr = Appr(args)
appr.train(model, train_loader, test_loaders, replay_loaders, accelerator)