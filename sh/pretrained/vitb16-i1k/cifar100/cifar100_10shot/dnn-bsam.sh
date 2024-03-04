# ------------------------------------------------------
## conda activate /data1/lsj9862/miniconda3/envs/bsam
# ------------------------------------------------------
for seed in 0 # 0 1 2 
do
CUDA_VISIBLE_DEVICES=3 python3 run_baseline.py --method=dnn --optim=bsam --rho=0.001 --dataset=cifar100 --use_validation --dat_per_cls=10 \
--model=vitb16-i1k --pre_trained  --lr_init=5e-1 --epochs=100 --wd=1e-3 --noise_scale=1e-4 \
--scheduler=cos_decay --seed=${seed} --no_amp --ignore_wandb
done
# ------------------------------------------------------
# ------------------------------------------------------

# # Log
# ------------------------------
# Set sam optimizer with lr_init 0.001 / wd 0.01 / momentum 0.9 / rho 0.01 / noise_scale 0.0001
# ------------------------------
# Set cos_decay lr scheduler
# ------------------------------
# Set AMP Scaler for sam
# ------------------------------
# Start training dnn with sam optimizer from 1 epoch!
# -------  --------  --------  ---------  --------  ----------  ---------  ---------  ---------  --------
#   epoch  method          lr    tr_loss    tr_acc    val_loss    val_acc    val_nll    val_ece      time
# -------  --------  --------  ---------  --------  ----------  ---------  ---------  ---------  --------
#       1  dnn-sam     0.0000     6.4532    0.7000      6.1929     1.2400     6.1928     0.2094   54.3137
#       2  dnn-sam     0.0001     6.2477    0.7000      5.7361     1.2000     5.7361     0.1403   22.9080
#       3  dnn-sam     0.0002     5.5794    1.3000      5.0041     1.1000     5.0041     0.0632   22.8230
#       4  dnn-sam     0.0003     5.0157    1.6000      4.7740     1.5600     4.7740     0.0255   23.1281
#       5  dnn-sam     0.0004     4.8124    1.5000      4.6678     1.2800     4.6678     0.0139   23.0249
#       6  dnn-sam     0.0005     4.6624    2.1000      4.6228     1.3800     4.6228     0.0078   23.1547
#       7  dnn-sam     0.0006     4.6190    2.6000      4.5911     1.6800     4.5911     0.0074   23.1261
#       8  dnn-sam     0.0007     4.5852    2.3000      4.5590     2.0000     4.5590     0.0073   23.2336
#       9  dnn-sam     0.0008     4.5522    3.2000      4.5311     2.4200     4.5311     0.0075   22.9681
#      10  dnn-sam     0.0009     4.5109    3.3000      4.4767     2.8600     4.4767     0.0079   22.7314
# -------  --------  --------  ---------  --------  ----------  ---------  ---------  ---------  --------
#   epoch  method          lr    tr_loss    tr_acc    val_loss    val_acc    val_nll    val_ece      time
# -------  --------  --------  ---------  --------  ----------  ---------  ---------  ---------  --------
#      11  dnn-sam     0.0010     4.4330    3.7000      4.3855     3.5600     4.3855     0.0081   23.3758
#      12  dnn-sam     0.0010     4.3227    6.9000      4.2761     4.5400     4.2761     0.0100   23.3101
#      13  dnn-sam     0.0010     4.1463   10.0000      4.1177     6.9200     4.1177     0.0118   23.1775
#      14  dnn-sam     0.0010     3.9661   14.6000      3.9157     9.9000     3.9157     0.0247   23.2437
#      15  dnn-sam     0.0010     3.6792   20.9000      3.6503    15.3200     3.6503     0.0463   23.3139
#      16  dnn-sam     0.0009     3.2768   31.0000      3.3520    20.0600     3.3520     0.0430   23.0040
#      17  dnn-sam     0.0009     2.8110   41.8000      3.0132    25.6000     3.0132     0.0332   23.2126
#      18  dnn-sam     0.0009     2.4580   50.6000      2.7027    32.2800     2.7027     0.0333   23.1042
#      19  dnn-sam     0.0009     2.1294   56.6000      2.4611    36.6600     2.4611     0.0316   23.0474
#      20  dnn-sam     0.0009     1.7471   64.3000      2.2821    41.0000     2.2821     0.0232   23.1251
# -------  --------  --------  ---------  --------  ----------  ---------  ---------  ---------  --------
#   epoch  method          lr    tr_loss    tr_acc    val_loss    val_acc    val_nll    val_ece      time
# -------  --------  --------  ---------  --------  ----------  ---------  ---------  ---------  --------
#      21  dnn-sam     0.0009     1.6286   67.3000      2.1630    43.9600     2.1629     0.0256   22.6673
#      22  dnn-sam     0.0009     1.3868   73.1000      2.0295    46.6400     2.0295     0.0233   22.9492
#      23  dnn-sam     0.0009     1.2713   75.2000      1.9278    49.8800     1.9277     0.0257   22.6967
#      24  dnn-sam     0.0009     1.0482   80.9000      1.8244    52.2400     1.8244     0.0206   23.1948
#      25  dnn-sam     0.0009     0.9153   84.4000      1.7056    55.3200     1.7056     0.0164   22.9845
#      26  dnn-sam     0.0009     0.8540   86.7000      1.7294    55.4400     1.7294     0.0215   22.9272
#      27  dnn-sam     0.0008     0.8290   85.9000      1.6880    56.3000     1.6880     0.0194   23.1501