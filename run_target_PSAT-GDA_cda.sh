# ########R2-------------------------------------
~/anaconda3/envs/vit_kd/bin/python target_PSAT-GDA_cda_oc.py --cls_par 0.6 --da uda --dset office --gpu_id 0 --s 0 --t 1 --output ckpspdtcda/target_PSAT-GDAcda_0113_1/ --seed 2020;
~/anaconda3/envs/vit_kd/bin/python target_PSAT-GDA_cda_oc.py --cls_par 0.6 --da uda --dset office --gpu_id 0 --s 1 --t 0 --output ckpspdtcda/target_PSAT-GDAcda_0113_1/ --seed 2020;
~/anaconda3/envs/vit_kd/bin/python target_PSAT-GDA_cda_oc.py --cls_par 0.6 --da uda --dset office --gpu_id 0 --s 2 --t 0 --output ckpspdtcda/target_PSAT-GDAcda_0113_1/ --seed 2020;

~/anaconda3/envs/vit_kd/bin/python target_PSAT-GDA_cda_oh.py --cls_par 0.6 --da uda --dset office-home --gpu_id 0 --s 0 --t 1 --output ckpspdtcda/target_PSAT-GDAcda_0113_1/;

~/anaconda3/envs/vit_kd/bin/python target_PSAT-GDA_cda_oh.py --cls_par 0.6 --da uda --dset office-home --gpu_id 0 --s 1 --t 0 --output ckpspdtcda/target_PSAT-GDAcda_0113_1/;

~/anaconda3/envs/vit_kd/bin/python target_PSAT-GDA_cda_oh.py --cls_par 0.6 --da uda --dset office-home --gpu_id 0 --s 2 --t 0 --output ckpspdtcda/target_PSAT-GDAcda_0113_2/;

 ~/anaconda3/envs/vit_kd/bin/python target_PSAT-GDA_cda_oh.py --cls_par 0.6 --da uda --dset office-home --gpu_id 0 --s 3 --t 0 --output ckpspdtcda/target_PSAT-GDAcda_0113_1/;
