# PYTHONUNBUFFERED=1;CUDA_VISIBLE_DEVICES=7;PYTHONPATH=/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/DriveX/mesh;
# HF_ENDPOINT=https://hf-mirror.com;HF_HOME=/SSD_DISK_1/users/jiangjunzhe/huggingface;TORCH_HOME=/SSD_DISK_1/users/jiangjunzhe/torch

# base
# CUDA_VISIBLE_DEVICES=2 python navsim/planning/script/run_pdm_score_with_render_base.py \
# train_test_split=mini agent=diffusiondrive_agent worker=single_machine_thread_pool \
# agent.checkpoint_path=model/diffusiondrive_navsim_88p1_PDMS.pth \
# experiment_name=diffusiondrive_agent_eval

# CUDA_VISIBLE_DEVICES=2 nohup python navsim/planning/script/run_pdm_score_with_render_base.py \
# train_test_split=mini agent=transfuser_agent worker=single_machine_thread_pool \
# agent.checkpoint_path=model/transfuser_seed_0.ckpt \
# experiment_name=transfuser_agent_eval &

# CUDA_VISIBLE_DEVICES=3 nohup python navsim/planning/script/run_pdm_score_with_render_base.py \
# train_test_split=mini agent=constant_velocity_agent worker=single_machine_thread_pool \
# experiment_name=constant_velocity_agent_eval &

# CUDA_VISIBLE_DEVICES=4 nohup python navsim/planning/script/run_pdm_score_with_render_base.py \
# train_test_split=mini agent=stp3_transfuser_agent worker=single_machine_thread_pool \
# agent.checkpoint_path=model/stp3_transfuser_epoch-9.ckpt \
# experiment_name=stp3_transfuser_agent_eval &

# CUDA_VISIBLE_DEVICES=5 nohup python navsim/planning/script/run_pdm_score_with_render_base.py \
# train_test_split=mini agent=vad_agent worker=single_machine_thread_pool \
# agent.checkpoint_path=model/vad_epoch_99.ckpt \
# experiment_name=vad_agent_eval &

# edit
# CUDA_VISIBLE_DEVICES=3 nohup python navsim/planning/script/run_pdm_score_with_render_edit.py \
# train_test_split=mini agent=diffusiondrive_agent worker=single_machine_thread_pool \
# agent.checkpoint_path=model/diffusiondrive_navsim_88p1_PDMS.pth \
# experiment_name=diffusiondrive_agent_eval &

# CUDA_VISIBLE_DEVICES=4 nohup python navsim/planning/script/run_pdm_score_with_render_edit.py \
# train_test_split=mini agent=transfuser_agent worker=single_machine_thread_pool \
# agent.checkpoint_path=model/transfuser_seed_0.ckpt \
# experiment_name=transfuser_agent_eval &

# CUDA_VISIBLE_DEVICES=5 nohup python navsim/planning/script/run_pdm_score_with_render_edit.py \
# train_test_split=mini agent=constant_velocity_agent worker=single_machine_thread_pool \
# experiment_name=constant_velocity_agent_eval &

# CUDA_VISIBLE_DEVICES=6 nohup python navsim/planning/script/run_pdm_score_with_render_edit.py \
# train_test_split=mini agent=stp3_transfuser_agent worker=single_machine_thread_pool \
# agent.checkpoint_path=model/stp3_transfuser_epoch-9.ckpt \
# experiment_name=stp3_transfuser_agent_eval &

# CUDA_VISIBLE_DEVICES=7 nohup python navsim/planning/script/run_pdm_score_with_render_edit.py \
# train_test_split=mini agent=vad_agent worker=single_machine_thread_pool \
# agent.checkpoint_path=model/vad_epoch_99.ckpt \
# experiment_name=vad_agent_eval &

# multi agent
CUDA_VISIBLE_DEVICES=3 nohup python navsim/planning/script/run_pdm_score_with_render_multi_agent.py \
train_test_split=mini agent=diffusiondrive_agent worker=single_machine_thread_pool \
agent.checkpoint_path=model/diffusiondrive_navsim_88p1_PDMS.pth \
experiment_name=diffusiondrive_agent_eval &

CUDA_VISIBLE_DEVICES=4 nohup python navsim/planning/script/run_pdm_score_with_render_multi_agent.py \
train_test_split=mini agent=transfuser_agent worker=single_machine_thread_pool \
agent.checkpoint_path=model/transfuser_seed_0.ckpt \
experiment_name=transfuser_agent_eval &

CUDA_VISIBLE_DEVICES=5 nohup python navsim/planning/script/run_pdm_score_with_render_multi_agent.py \
train_test_split=mini agent=constant_velocity_agent worker=single_machine_thread_pool \
experiment_name=constant_velocity_agent_eval &

CUDA_VISIBLE_DEVICES=6 nohup python navsim/planning/script/run_pdm_score_with_render_multi_agent.py \
train_test_split=mini agent=stp3_transfuser_agent worker=single_machine_thread_pool \
agent.checkpoint_path=model/stp3_transfuser_epoch-9.ckpt \
experiment_name=stp3_transfuser_agent_eval &

CUDA_VISIBLE_DEVICES=7 nohup python navsim/planning/script/run_pdm_score_with_render_multi_agent.py \
train_test_split=mini agent=vad_agent worker=single_machine_thread_pool \
agent.checkpoint_path=model/vad_epoch_99.ckpt \
experiment_name=vad_agent_eval &