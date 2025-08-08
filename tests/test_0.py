#在marmousi上测试unet的不同下采样效果


test_list = [
    [[2,2,2,2],[True,True,True,True,True]],
    [[2,2,2,2],[False,True,True,True,True]],
    [[4,2,2,2],[True,True,True,True,True]],
    [[4,2,2,2],[False,True,True,True,True]],
    [[4,4,2,2],[True,True,True,True,True]],
    [[4,4,2,2],[False,True,True,True,True]],
]

import  os
for i,x in enumerate(test_list):
    rates = x[0]
    skip_on = x[1]
    rates_str = str(rates).replace(" ", "")
    skip_on_str = str(skip_on).replace(" ", "")

    train_log_path = f"/data/yibiao/nn_loss/experiments/marmousi/log/train_unet_{i}.hdf5"
    inversion_log_path = f"/data/yibiao/nn_loss/experiments/marmousi/log/inversion_unet_{i}.hdf5"
    checkpoint_path = f"/data/yibiao/nn_loss/experiments/marmousi/log/model_unet_{i}.pt"
    os.system(f"python scripts/train.py --config config/template.yaml  train_stage.rates={rates_str}  train_stage.skip_on={skip_on_str} generate_par.train_log_path={train_log_path} generate_par.inversion_log_path={inversion_log_path} generate_par.checkpoint_path={checkpoint_path}")
    os.system(f"python scripts/inversion.py --config config/template.yaml  train_stage.rates={rates_str} train_stage.skip_on={skip_on_str} generate_par.train_log_path={train_log_path} generate_par.inversion_log_path={inversion_log_path} generate_par.checkpoint_path={checkpoint_path}")
