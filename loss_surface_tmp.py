# %%
import torch
import utils.utils as utils
import argparse

# %%
"""
우리 방법론 : sabma
1순위 baseline : ptl
2순위 baseline : swag

유의할 점은 sabma(우리방법론)할 때는 backbone 자체를 load해야 되서 backbone_path argument에 ~/sabma_model.pt를 넣어줘야 됩니다!
baseline들은 model_path만 잘 넣어주면 돌아갈껍니다.
forward는 loss 계산하는 것만 일단 넣어놨는데 지금 제공하는 코드 바탕으로 loss surface를 우리 방법론과 baseline을 함께 그리면 될 것 같습니다.
서버는 39,40,41번 서버만 이용하셔야 됩니다! (37번에는 데이터셋이 없어요..)
"""

parser = argparse.ArgumentParser(description="plotting loss surface")

parser.add_argument("--method",
        type=str,
        default='sabma',
        choices=['swag', 'ptl', 'sabma'],
        help='select learning method (sabma == ours)')

parser.add_argument('--model_path',
        type=str,
        default=None,
        help='path of the sampled model (e.g. ~~~ptl_1.pt, ~~~sabma_1.pt, ~~~swag_1.pt)')

parser.add_argument("--backbone_path",
        type=str,
        default=None,
        help='when run sabma, we need to load backbone model (e.g. ~~~~/sabma_model.pt)')

args = parser.parse_args()

# %%
MODEL='resnet18'
DATASET = "cifar10"
DAT_PER_CLS = 10
PRE_TRAINED=True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

utils.set_seed(SEED)
# %%
## Load Data
tr_loader, val_loader, te_loader, num_classes = utils.get_dataset(dataset=DATASET,
                                                        data_path='/data1/lsj9862/data/cifar10',
                                                        dat_per_cls=DAT_PER_CLS,
                                                        use_validation=True,
                                                        batch_size=256,
                                                        num_workers=4,
                                                        seed=SEED,
                                                        aug=True,
                                                        )


# %%
## Load Backbone Model
model = utils.get_backbone(MODEL, num_classes, DEVICE, PRE_TRAINED).to(DEVICE)

# Set criterion
criterion = torch.nn.CrossEntropyLoss()


# %%
## Load Best Model
if args.method == 'swag':
    model = torch.load(args.model_path)
elif args.method == 'sabma':
    model = torch.load(args.backbone_path)
    params = torch.load(args.model_path)
    model.load_state_dict(params, strict=False)
elif args.method == 'ptl':
    checkpoint = torch.load(args.model_path)
    for key in list(checkpoint.keys()):
        if 'backbone.' in key:
            new_key = key.replace('backbone.', '')
            checkpoint[new_key] = checkpoint.pop(key)
        elif 'classifier.' in key:
            new_key = key.replace('classifier', 'fc')
            checkpoint[new_key] = checkpoint.pop(key)
    model.load_state_dict(checkpoint)


# %%
## Evaluation (loss만 계산)

loss_sum = 0.0
num_objects_total = len(tr_loader.dataset)

model.eval()
with torch.no_grad():
    for _, (input, target) in enumerate(tr_loader):
        input, target = input.to(DEVICE), target.to(DEVICE)
        if args.method in ['swag', 'ptl']:
            pred = model(input)
        elif args.method in ['sabma']:
            pred = model(params, input)
        else:
            raise NotImplementedError()
        batch_loss = criterion(pred, target)
        loss_sum += batch_loss.item() * input.size(0)
    
    total_loss = loss_sum / num_objects_total