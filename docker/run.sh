docker run \
       -it \
       --gpus all \
       -v /home/glebk/VSProjects/projects/Matching/submission:/workspace \
       -v /home/glebk/VSProjects/projects/Matching/test_data:/workspace/data \
       glebkaa/odsai \
       bash
