#在he上测试unet data_norm的不同效果

test_list = [
    ['shot_norm','batch'],
    ['shot_norm','batch'],
    ['shot_norm',None],
    ['shot_norm',None],

    [None,'batch'],
    [None,'batch'],
    [None,None],
    [None,None],
]


import  os
for i,x in enumerate(test_list):
    data_norm = test_list[i][0]
    model_norm = test_list[i][1]


    train_log_path = f"/data/yibiao/nn_loss/experiments/he/log/train_unet_{i}.hdf5"
    inversion_log_path = f"/data/yibiao/nn_loss/experiments/he/log/inversion_unet_{i}.hdf5"
    checkpoint_path = f"/data/yibiao/nn_loss/experiments/he/log/model_unet_{i}.pt"

    os.system(f"python scripts/train.py --config config/template_he.yaml    generate_par.data_norm={data_norm}  train_stage.model.model_norm={model_norm} generate_par.train_log_path={train_log_path} generate_par.inversion_log_path={inversion_log_path} generate_par.checkpoint_path={checkpoint_path}")
    os.system(f"python scripts/inversion.py --config config/template_he.yaml   generate_par.data_norm={data_norm}   train_stage.model.model_norm={model_norm}  generate_par.train_log_path={train_log_path} generate_par.inversion_log_path={inversion_log_path} generate_par.checkpoint_path={checkpoint_path}")
