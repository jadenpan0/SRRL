
Time=$(date +%Y-%m-%d_%H-%M-%S)
#Time="2025-04-25_07-41-25"
echo "Start time: ${Time}"
SaveInterval=2
SavePath="/data/panjiadong/project/NIPS25/outputs/B2-DiffuRL/model/lora"
PromptFile="/data/panjiadong/project/NIPS25/B2-DiffuRL-main_xl/config/prompt/single3_train.json"
RandomPrompt=1
ExpName="exp_B2DiffuRL_b5_p3"
Seed=300
Beta1=1
Beta2=1
BatchCnt=32
Total_StageCnt=50
StageCnt=5
SplitStepLeft=1
SplitStepRight=20
TrainEpoch=2
AccStep=64
LR=0.0001
ModelVersion="/data/panjiadong/model/stable-diffusion-xl-base-1.0"
NumStep=20
History_Cnt=8
PosThreshold=0.5
NegThreshold=-0.5
SplitTime=5
Dev_Id=2
Total_resample_num=10

CUDA_FALGS="--config.dev_id ${Dev_Id}"
SAMPLE_FLAGS="--config.sample.num_batches_per_epoch ${BatchCnt} --config.sample.num_steps ${NumStep} --config.prompt_file ${PromptFile} --config.prompt_random_choose ${RandomPrompt} --config.split_time ${SplitTime}" # 
EXP_FLAGS="--config.exp_name ${Time} --config.save_path ${SavePath} --config.pretrained.model ${ModelVersion}"

#for sample_num in $(seq 0 $((Total_resample_num)))
#do 
for step in $(seq 0 $((Total_StageCnt-1)))
do
    i=$((step%StageCnt))
    resample_num=$((step/StageCnt))
    interval=$((SplitStepRight-SplitStepLeft+1))
    level=$((i*interval/StageCnt))
    cur_split_step=$((level+SplitStepLeft))

    RESAMPLE_NUM="--config.resample_num ${resample_num}"
    RUN_FLAGS="--config.run_name stage${step} --config.split_step ${cur_split_step} --config.eval.history_cnt ${History_Cnt} --config.eval.pos_threshold ${PosThreshold} --config.eval.neg_threshold ${NegThreshold}"
    temp_seed=$((Seed+i))
    RANDOM_FLAGS="--config.seed ${temp_seed}"
    TRAIN_FLAGS="--config.train.save_interval ${SaveInterval} --config.train.num_epochs ${TrainEpoch} --config.train.beta1 ${Beta1} --config.train.beta2 ${Beta2} --config.train.gradient_accumulation_steps ${AccStep} --config.train.learning_rate ${LR}"
    LORA_FLAGS=""
    if [ $step != 0 ]; then
        minus_i=$((step-1))
        cur_epoch=${TrainEpoch}
        checkpoint=$((cur_epoch/SaveInterval))
        LORA_FLAGS="--config.resume_from ${SavePath}/${Time}/stage${minus_i}/checkpoints/checkpoint_${checkpoint}"
    fi

    echo "||=========== round: ${step} ===========||"
    echo $CUDA_FALGS
    echo $TRAIN_FLAGS
    echo $SAMPLE_FLAGS
    echo $RANDOM_FLAGS
    echo $EXP_FLAGS
    echo $RUN_FLAGS
    echo $LORA_FLAGS

    python3 train_xl.py $CUDA_FALGS $TRAIN_FLAGS $SAMPLE_FLAGS $RANDOM_FLAGS $EXP_FLAGS $RUN_FLAGS $LORA_FLAGS $SELECT_FLAGS $RESAMPLE_NUM

    sleep 2
done
#done

