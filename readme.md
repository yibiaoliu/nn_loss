# 文件结构
```shell
| nn_loss          
  |-- config               #一次实验（从模型训练到使用模型反演）对应一个config文件
  |-- data                 #定义了模型训练时和反演阶段时的dataset
  |-- modules              #定义自编码网络模型
  |-- prepare              #对于速度模型需要提前处理，转换为需要的格式
  |-- scripts
      |-- train.py         #训练阶段脚本
      |-- inversion.py     #反演阶段脚本
  |-- tests                #为方便记录实验的一个文件夹
  |-- utils                #工具包
  |-- vis.ipynb            #可视化实验结果[训练阶段损失，自编码器还原效果，反演阶段损失，反演效果]
```

# 使用流程
1. 处理数据得到需要的几个文件。[观测数据obs_signal.npy、接收器位置rec_loc.npy、观测器位置src_loc.npy、初始速度模型vp_init.npy、真实速度模型vp_true.npy、地震子波wavelet.npy]
2. 编写`config`文件
3. 运行脚本
```shell
python scripts/train.py --config config/template.yaml 
python scripts/inversion.py --config config/template.yaml 
```
4. 可视化分析
