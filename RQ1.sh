#!/bin/bash

# =========================
# Configurable lists
# =========================
datasets=(
   "re1-ob" 
   "re1-ss"
    
   "re2-ob" 
   "re2-ss"

   "train-ticket"
)



#----------------------------------------#
#------------Proposed--------------------#
#----------------------------------------#
methods=(RMDnet) 
model_class=("SFlexRCA")  #SFlexRCA is DeepReasonRCA
scalar_type_options=("Robust" "Standard")  
combine_baro_options=(true) # SET TO TRUE TO COMBINE ReasonRCA
seeds=(1 2)

for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        for method in "${methods[@]}"; do
            for model in "${model_class[@]}"; do
                for combine_baro in "${combine_baro_options[@]}"; do
                    for scalar_type in "${scalar_type_options[@]}"; do
                        echo "Dataset: $dataset, Method: $method, Model: $model, Combine"
                        # skip baro + combine_baro=true
                        if [ "$method" = "baro" ] && [ "$combine_baro" = true ]; then
                            continue
                        fi

                        #activate conda environment called RCAEval
                        source ~/miniconda3/etc/profile.d/conda.sh
                        conda activate RCAEval
                        cmd="python main.py --dataset $dataset --method $method --seed $seed --model_class $model --scaler_type $scalar_type"

                        if [ "$combine_baro" = true ]; then
                            cmd="$cmd --combine_baro_post"
                        fi

                        echo "Running: $cmd"
                        eval $cmd
                    done
                done
            done
        done
    done
done





#----------------------------------------#
#--------Deep learning methods-----------#
#----------------------------------------#
methods=(RMDnet) 
model_class=("iTransformer" "Fits" "Dlinear")  
scalar_type_options=("Robust") 
combine_baro_options=(false)
seeds=(1 2)
#
for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        for method in "${methods[@]}"; do
            for model in "${model_class[@]}"; do
                for combine_baro in "${combine_baro_options[@]}"; do
                    for scalar_type in "${scalar_type_options[@]}"; do
                        echo "Dataset: $dataset, Method: $method, Model: $model, Combine"
                        # skip baro + combine_baro=true
                        if [ "$method" = "baro" ] && [ "$combine_baro" = true ]; then
                            continue
                        fi

                        #activate conda environment called RCAEval
                        source ~/miniconda3/etc/profile.d/conda.sh
                        conda activate RCAEval
                        cmd="python main.py --dataset $dataset --method $method --seed $seed --model_class $model --scaler_type $scalar_type"

                        if [ "$combine_baro" = true ]; then
                            cmd="$cmd --combine_baro_post"
                        fi

                        echo "Running: $cmd"
                        eval $cmd
                    done
                done
            done
        done
    done
done


#----------------------------------------#
#----statistical methods on their own----#
#----------------------------------------#
methods=(e_diagnosis nsigma baro)
model_class=("iTransformer") 
scalar_type_options=("Robust") 
combine_baro_options=(false)
seeds=(1 2)
#
for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        for method in "${methods[@]}"; do
            for model in "${model_class[@]}"; do
                for combine_baro in "${combine_baro_options[@]}"; do
                    for scalar_type in "${scalar_type_options[@]}"; do
                        echo "Dataset: $dataset, Method: $method, Model: $model, Combine"
                        # skip baro + combine_baro=true
                        if [ "$method" = "baro" ] && [ "$combine_baro" = true ]; then
                            continue
                        fi

                        #activate conda environment called RCAEval
                        source ~/miniconda3/etc/profile.d/conda.sh
                        conda activate RCAEval
                        cmd="python main.py --dataset $dataset --method $method --seed $seed --model_class $model --scaler_type $scalar_type"

                        if [ "$combine_baro" = true ]; then
                            cmd="$cmd --combine_baro_post"
                        fi

                        echo "Running: $cmd"
                        eval $cmd
                    done
                done
            done
        done
    done
done