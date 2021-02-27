DOMAIN_NAME='reacher'
TASK_NAME='easy'
ACTION_REPEAT=4
EXP_NAME='curl_sac_e2e_ri'
N_TRAIN_STEPS=125000
BATCH_SIZE=128

CUDA_VISIBLE_DEVICES=5 python train_modify_logging_step.py --domain_name ${DOMAIN_NAME} --task_name ${TASK_NAME} \
    --encoder_type pixel --action_repeat ${ACTION_REPEAT} \
    --save_tb --pre_transform_image_size 100 --image_size 84 \
    --work_dir ./tmp --agent curl_sac_e2e_ri --frame_stack 3 \
    --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 \
    --batch_size ${BATCH_SIZE} --seed -1 --num_train_steps ${N_TRAIN_STEPS} \
    --encoder_lr 1e-3 --idm_lr 1e-3 \
    --exp ${EXP_NAME}

CUDA_VISIBLE_DEVICES=5 python train_modify_logging_step.py --domain_name ${DOMAIN_NAME} --task_name ${TASK_NAME} \
    --encoder_type pixel --action_repeat ${ACTION_REPEAT} \
    --save_tb --pre_transform_image_size 100 --image_size 84 \
    --work_dir ./tmp --agent curl_sac_e2e_ri --frame_stack 3 \
    --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 \
    --batch_size ${BATCH_SIZE} --seed -1 --num_train_steps ${N_TRAIN_STEPS} \
    --encoder_lr 1e-3 --idm_lr 1e-3 \
    --exp ${EXP_NAME}