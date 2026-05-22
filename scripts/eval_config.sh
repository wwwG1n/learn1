#!/bin/bash
# 由 eval_gen_eval.sh source：conda / 路径（生成超参均在 evaluation/gen_eval/geneval_lumina_dimoo.py 默认值）

python_ext="${python_ext:-python}"
lumina_dimoo_env="${lumina_dimoo_env:-lumina_dimoo}"
geneval_env="${geneval_env:-geneval}"

# 与 geneval_lumina_dimoo.py 中 output_root 默认一致，供 shell 解析生成目录
geneval_output_root="${geneval_output_root:-output/geneval_results}"

geneval_m2f_config="${geneval_m2f_config:-evaluation/gen_eval/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py}"
geneval_m2f_weights_dir="${geneval_m2f_weights_dir:-evaluation/gen_eval/mask2former}"
