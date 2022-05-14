docker run \
       -it \
       --gpus all \
       -v /home/glebk/VSProjects/projects/data_fusion_matching/submission:/workspace \
       -v /home/glebk/VSProjects/projects/data_fusion_matching/test_data_small:/workspace/data \
       glebkaa/odsai:new \
       bash
