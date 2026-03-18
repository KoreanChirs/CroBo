# DMC
python3 hydra_launcher.py --config-name DMC_BC_config.yaml --multirun hydra/launcher=joblib \
        wandb.project=dmc_test wandb.entity=cortexbench \
        env=dmc_walker_stand-v1,dmc_walker_walk-v1,dmc_reacher_easy-v1,dmc_cheetah_run-v1,dmc_finger_spin-v1 \
           seed=1,2,3,4,5 embedding="$1"

python3 hydra_launcher.py --config-name DMC_BC_config.yaml --multirun hydra/launcher=joblib \
        wandb.project=dmc_test wandb.entity=cortexbench \
        env=dmc_walker_stand-v1,dmc_walker_walk-v1,dmc_reacher_easy-v1,dmc_cheetah_run-v1,dmc_finger_spin-v1 \
           seed=6,7,8,9,10 embedding="$1"