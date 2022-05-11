docker run \
       -it \
       --gpus all \
       -v /home/glebk/VSProjects/projects/data_fusion_matching/submission:/workspace \
       -v /home/glebk/VSProjects/projects/data_fusion_matching/test_data:/workspace/data \
       glebkaa/odsai \
       bash
