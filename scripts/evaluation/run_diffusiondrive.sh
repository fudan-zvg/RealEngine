TRAIN_TEST_SPLIT=mini
CHECKPOINT=$NAVSIM_DEVKIT_ROOT/model/diffusiondrive_navsim_88p1_PDMS.pth

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
        train_test_split=$TRAIN_TEST_SPLIT \
        agent=diffusiondrive_agent \
        worker=ray_distributed \
        agent.checkpoint_path=$CHECKPOINT \
        experiment_name=diffusiondrive_agent_eval \
        metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache/"

#train_test_split=mini agent=diffusiondrive_agent worker=single_machine_thread_pool
#agent.checkpoint_path=model/diffusiondrive_navsim_88p1_PDMS.pth
#experiment_name=diffusiondrive_agent_eval