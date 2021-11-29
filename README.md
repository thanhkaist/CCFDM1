# CCFDM: Sample-efficient Reinforcement Learning Representation Learning with Curiosity Contrastive Forward Dynamics Model

This repository is the official implementation of [CCFDM]() for the DeepMind control experiments. Our implementation of SAC is based on [SAC+AE](https://github.com/denisyarats/pytorch_sac_ae) by Denis Yarats. 

## Installation 

All of the dependencies are in the `conda_env.yml` file. They can be installed manually or with the following command:

```
conda env create -f conda_env.yml
```

## Instructions
To train CCFDM on all the tasks from image-based observations run `bash script/run_all_ri.sh` from the root of this directory. You can modify to try different environments / hyperparamters by changing the scripts in the `script` folder.

In your console, you should see printouts that look like:

```
| train | E: 221 | S: 28000 | D: 18.1 s | R: 785.2634 | BR: 3.8815 | A_LOSS: -305.7328 | CR_LOSS: 190.9854 | CU_LOSS: 0.0000
| train | E: 225 | S: 28500 | D: 18.6 s | R: 832.4937 | BR: 3.9644 | A_LOSS: -308.7789 | CR_LOSS: 126.0638 | CU_LOSS: 0.0000
| train | E: 229 | S: 29000 | D: 18.8 s | R: 683.6702 | BR: 3.7384 | A_LOSS: -311.3941 | CR_LOSS: 140.2573 | CU_LOSS: 0.0000
| train | E: 233 | S: 29500 | D: 19.6 s | R: 838.0947 | BR: 3.7254 | A_LOSS: -316.9415 | CR_LOSS: 136.5304 | CU_LOSS: 0.0000
```

Log abbreviation mapping:

```
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - mean episode reward
BR - average reward of sampled batch
A_LOSS - average loss of actor
CR_LOSS - average loss of critic
CU_LOSS - average loss of the CURL encoder
```

All data related to the run is stored in the specified `working_dir`. To enable model or video saving, use the `--save_model` or `--save_video` flags. For all available flags, inspect `train.py`. To visualize progress with tensorboard run:

```
tensorboard --logdir log --port 6006
```

and go to `localhost:6006` in your browser. If you're running headlessly, try port forwarding with ssh. 

For GPU accelerated rendering, make sure EGL is installed on your machine and set `export MUJOCO_GL=egl`. For environment troubleshooting issues, see the DeepMind control documentation.

## References

This is the code for the paper 
> Thanh Nguyen, Tung M. Luu, Thang Vu, Chang D. Yoo. Sample-efficient Reinforcement Learning Representation Learning with Curiosity Contrastive Forward Dynamics Model. IROS 2021 - 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems [[ArXiv](https://arxiv.org/abs/2103.08255)]

If you want to cite this paper:
```
@article{nguyen2021sample,
  title={Sample-efficient Reinforcement Learning Representation Learning with Curiosity Contrastive Forward Dynamics Model},
  author={Nguyen, Thanh and Luu, Tung M and Vu, Thang and Yoo, Chang D},
  journal={arXiv preprint arXiv:2103.08255},
  year={2021}
}
```

## Acknowledgment

This work was partly supported by Institute for Information &communications Technology Planning & Evaluation(IITP)grant funded by the Korea government(MSIT) (No. 2019-0-01396, Development of framework for analyzing, detecting,mitigating of bias in AI model and training data and No. 2021-0-01381, Development of Causal AI through Video Understanding
and Reinforcement Learning)

