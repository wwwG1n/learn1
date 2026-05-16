#!/bin/bash
#
# eval_gen_eval.sh — Lumina-DiMOO GenEval：lumina_dimoo 生成 + geneval 检测与汇总
#
# 命令示例（在仓库根目录执行）：
#
#   全流程（生成 + 评估）。默认生成目录前缀为 gen_eval（例如 gen_eval_lumina_dimoo_geneval_20260101_120000）：
#     CUDA_VISIBLE_DEVICES=0 ./scripts/eval_gen_eval.sh
#
#   自定义输出前缀（不设则默认为 gen_eval；若不要前缀可显式置空）：
#     GENEVAL_OUTPUT_DIR_PREFIX=exp42 CUDA_VISIBLE_DEVICES=0 ./scripts/eval_gen_eval.sh
#     GENEVAL_OUTPUT_DIR_PREFIX= ./scripts/eval_gen_eval.sh
#
#   仅评估已有生成目录（跳过生成）：
#     GENEVAL_IMAGE_DIR=output/geneval_results/gen_eval_lumina_dimoo_geneval_20260414_221351 ./scripts/eval_gen_eval.sh
#
#   透传非算法类运行参数给 evaluation/gen_eval/geneval_lumina_dimoo.py
#   （例如 metadata/output_root/n_samples；采样、EN、cache 等算法超参数请在 Python 脚本默认值中维护）：
#     GENEVAL_GEN_EXTRA_ARGS="--metadata_file prompts/evaluation_metadata.jsonl --output_root output/geneval_results" ./scripts/eval_gen_eval.sh
#

set -eo pipefail

conda_shell() {
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

source "${SCRIPT_DIR}/eval_config.sh"

# 生成目录：{GENEVAL_OUTPUT_DIR_PREFIX}_lumina_dimoo_geneval_<时间戳>；空字符串则无此前缀段
: "${GENEVAL_OUTPUT_DIR_PREFIX:=gen_eval}"

run_lumina_generation() {
    echo "步骤 1: conda 环境 ${lumina_dimoo_env} -> evaluation/gen_eval/geneval_lumina_dimoo.py (output_dir_prefix=${GENEVAL_OUTPUT_DIR_PREFIX})"
    echo "注意: 算法相关超参数统一由 evaluation/gen_eval/geneval_lumina_dimoo.py 默认值控制。"
    conda_shell
    conda activate "${lumina_dimoo_env}"

    local logf
    logf="$(mktemp)"
    # shellcheck disable=SC2086
    "${python_ext}" evaluation/gen_eval/geneval_lumina_dimoo.py \
        --output_dir_prefix "${GENEVAL_OUTPUT_DIR_PREFIX}" \
        ${GENEVAL_GEN_EXTRA_ARGS:-} 2>&1 | tee "${logf}"

    GENEVAL_IMAGE_DIR="$(grep '^OUTPUT_DIR_FINAL=' "${logf}" | tail -1 | sed 's/^OUTPUT_DIR_FINAL=//')"
    rm -f "${logf}"

    if [ -z "${GENEVAL_IMAGE_DIR}" ] || [ ! -d "${GENEVAL_IMAGE_DIR}" ]; then
        echo "错误: 未解析到 OUTPUT_DIR_FINAL 或目录不存在。"
        exit 1
    fi
    echo "生成结果目录: ${GENEVAL_IMAGE_DIR}"
    conda deactivate
}

resolve_image_dir() {
    if [ -n "${GENEVAL_IMAGE_DIR:-}" ] && [ -d "${GENEVAL_IMAGE_DIR}" ]; then
        echo "使用已有图像目录（跳过生成）: ${GENEVAL_IMAGE_DIR}"
        return
    fi
    run_lumina_generation
}

echo "仓库根目录: ${REPO_ROOT}"
echo "开始 GenEval（Lumina-DiMOO）..."

resolve_image_dir

RESULTS_DIR="${GENEVAL_IMAGE_DIR}/geneval_eval_results"
mkdir -p "${RESULTS_DIR}"
DET_JSONL="${RESULTS_DIR}/det.jsonl"
SUMMARY_TXT="${RESULTS_DIR}/res.txt"

echo "步骤 2: conda 环境 ${geneval_env} -> evaluate_images.py"
conda_shell
conda activate "${geneval_env}"

"${python_ext}" evaluation/gen_eval/evaluate_images.py "${GENEVAL_IMAGE_DIR}" \
    --outfile "${DET_JSONL}" \
    --model-config "${geneval_m2f_config}" \
    --model-path "${geneval_m2f_weights_dir}"

echo "步骤 3: summary_scores.py"
"${python_ext}" evaluation/gen_eval/summary_scores.py "${DET_JSONL}" > "${SUMMARY_TXT}"

conda deactivate

echo ""
echo "GenEval 评估完成。"
echo "检测明细: ${DET_JSONL}"
echo "汇总: ${SUMMARY_TXT}"
echo ""
cat "${SUMMARY_TXT}"
