#在he上测试unet 收敛的不同效果

import  os
for i in range(5):

    train_log_path = f"/data/yibiao/nn_loss/experiments/he/log/train_unet_{i}.hdf5"
    inversion_log_path = f"/data/yibiao/nn_loss/experiments/he/log/inversion_unet_{i}.hdf5"
    checkpoint_path = f"/data/yibiao/nn_loss/experiments/he/log/model_unet_{i}.pt"

    os.system(f"python scripts/train.py --config config/template_he.yaml     generate_par.train_log_path={train_log_path} generate_par.inversion_log_path={inversion_log_path} generate_par.checkpoint_path={checkpoint_path}")
    os.system(f"python scripts/inversion.py --config config/template_he.yaml   generate_par.train_log_path={train_log_path} generate_par.inversion_log_path={inversion_log_path} generate_par.checkpoint_path={checkpoint_path}")
