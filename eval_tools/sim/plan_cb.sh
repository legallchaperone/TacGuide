#!/bin/bash
# plan_cb.sh <tag> <port>: official vision config on training seeds 0,1 then eval seeds 10000-10004
bash /root/autodl-tmp/run_seeds.sh $1 deploy_grasp_chip_vision $2 0 1
bash /root/autodl-tmp/run_seeds.sh $1 deploy_grasp_chip_vision $2 10000 10004
