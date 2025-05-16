TRAIN_TEST_SPLIT=mini
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=$CACHE_PATH
# worker=single_machine_thread_pool