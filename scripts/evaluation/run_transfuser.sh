# export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
# export OPENSCENE_DATA_ROOT="/SSD_DISK/data/openscene"
# export NUPLAN_MAPS_ROOT="/SSD_DISK/data/openscene/maps"
# export NAVSIM_EXP_ROOT="/SSD_DISK/users/lijingyu/workspace_simpvg/realengine/exp"
# export NAVSIM_DEVKIT_ROOT="/SSD_DISK/users/lijingyu/workspace_simpvg/realengine"

TRAIN_TEST_SPLIT=mini
CHECKPOINT=$NAVSIM_DEVKIT_ROOT/model/transfuser_seed_0.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_with_render_multi_agent.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=transfuser_agent \
worker=single_machine_thread_pool \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=transfuser_agent_eval

# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_with_render_edit_static.py \
# train_test_split=$TRAIN_TEST_SPLIT \
# agent=transfuser_agent \
# worker=single_machine_thread_pool \
# agent.checkpoint_path=$CHECKPOINT \
# experiment_name=transfuser_agent_eval

# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
# train_test_split=$TRAIN_TEST_SPLIT \
# agent=transfuser_agent \
# worker=single_machine_thread_pool \
# agent.checkpoint_path=$CHECKPOINT \
# experiment_name=transfuser_agent_eval 
