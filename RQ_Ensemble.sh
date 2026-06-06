datasets=(
    "re1-ss" 
)

##----combining with statistical methods----#
methods=(RMDnet)
model_class=("SFlexRCA") 
scalar_type_options=("Standard" "Robust" "Quantile" "MAD" "IQR" "ModifiedZ") 
ensemble_methods=("rank")
combine_baro_options=(true) # SET TO TRUE TO COMBINE ReasonRCA
seeds=(1 2)

for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        for method in "${methods[@]}"; do
            for model in "${model_class[@]}"; do
                for combine_baro in "${combine_baro_options[@]}"; do
                    for scalar_type in "${scalar_type_options[@]}"; do
                        for ensemble_method in "${ensemble_methods[@]}"; do
                            echo "ensemble_method: $ensemble_method, combine_baro: $combine_baro, method: $method, model: $model, scalar_type: $scalar_type, seed: $seed"
                            # skip baro + combine_baro=true
                            if [ "$method" = "baro" ] && [ "$combine_baro" = true ]; then
                                continue
                            fi

                            #activate conda environment called RCAEval
                            source ~/miniconda3/etc/profile.d/conda.sh
                            conda activate RCAEval
                            cmd="python main.py --dataset $dataset --method $method --seed $seed --model_class $model --scaler_type $scalar_type --ensemble_method $ensemble_method --research_question RQ_Ensemble"

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
done


#------------------------------------------#
#--------DeepRCA without ReasonRCA---------#
#------------------------------------------#
methods=(RMDnet)
model_class=("SFlexRCA") 
scalar_type_options=("Standard") 
ensemble_methods=("attention")
combine_baro_options=(false) # SET TO False to test without combining ReasonRCA
seeds=(1 2) 

for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        for method in "${methods[@]}"; do
            for model in "${model_class[@]}"; do
                for combine_baro in "${combine_baro_options[@]}"; do
                    for scalar_type in "${scalar_type_options[@]}"; do
                        for ensemble_method in "${ensemble_methods[@]}"; do
                            echo "ensemble_method: $ensemble_method, combine_baro: $combine_baro, method: $method, model: $model, scalar_type: $scalar_type, seed: $seed"
                            # skip baro + combine_baro=true
                            if [ "$method" = "baro" ] && [ "$combine_baro" = true ]; then
                                continue
                            fi

                            #activate conda environment called RCAEval
                            source ~/miniconda3/etc/profile.d/conda.sh
                            conda activate RCAEval
                            cmd="python main.py --dataset $dataset --method $method --seed $seed --model_class $model --scaler_type $scalar_type --ensemble_method $ensemble_method --research_question RQ_Ensemble"

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
done
