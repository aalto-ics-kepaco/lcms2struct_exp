for i in (seq 0 31)${BASE_DIR}/data/tools/cfmid4/${IMODE}/predicted_spectra__cv={0..9}.tar.gz
    for i in (seq 0 20)
        PYTHONPATH=../../publication/massbank/ python run_rt_prediction_approaches.py $i --n_jobs=12 --output_dir=../../../results_processed/comparison/massbank/ --no_plot --score_integration_approach=filtering__global
    end
end

for i in (seq 0 31)
    for j in (seq 0 20)
        echo $i$j
    end
end


for i in (seq 0 31)
    for j in (seq 0 20)
        set s (printf '%d%02d\n' $i $j)
        PYTHONPATH=../../publication/massbank/ python run_rt_prediction_approaches.py $s --n_jobs=12 --output_dir=../../../results_processed/comparison/massbank/ --no_plot --score_integration_approach=filtering__global --ms2scorer=sirius__sd__correct_mf__norm
    end
end

for scorer in "sirius__sd__correct_mf__norm" "metfrag__norm" "cfm-id__summed_up_sim__norm"
    for i in (seq 0 31)
        for j in (seq 0 20)
            set s (printf '%d%02d\n' $i $j)
            PYTHONPATH=../../publication/massbank/ python run_rt_prediction_approaches.py $s \
            --n_jobs=12 \
            --output_dir=../../../results_processed/comparison/massbank__exp_ver=2/ \
            --no_plot \
            --score_integration_approach=score_combination \
            --ms2scorer="$scorer"
        end
    end
end

for scorer in "sirius__sd__correct_mf__norm" "metfrag__norm" "cfm-id__summed_up_sim__norm"
    for i in (seq 0 31)
        for j in (seq 0 20)
            set s (printf '%d%02d\n' $i $j)
            PYTHONPATH="../../publication/massbank/" python run_msms_pl_rt_score_integration_approaches.py $s \
            --n_jobs=12 \
            --n_jobs_scoring_eval=3 \
            --output_dir="../../../results_processed/comparison/massbank__exp_ver=2/" \
            --no_plot \
            --score_integration_approach="msms_pl_rt_score_integration" \
            --ms2scorer="$scorer"
        end
    end
end