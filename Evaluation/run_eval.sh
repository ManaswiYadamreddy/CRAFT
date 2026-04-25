#!/usr/bin/env bash
# run_eval.sh — end-to-end OSDFace-paper eval for CRAFT + OSDFace.
#
# For each dataset, runs 4 pipelines — {craft, osdface} × {stage1, stage2} —
# then scores each output directory and writes per-stage comparison tables.
#
# Usage:
#   bash Evaluation/run_eval.sh celeba          # only CelebA
#   bash Evaluation/run_eval.sh lfw             # only LFW
#   bash Evaluation/run_eval.sh both            # both (default)
#
#   # Skip inference if restored images already exist:
#   bash Evaluation/run_eval.sh both score_only
#
#   # Limit to one stage:
#   bash Evaluation/run_eval.sh both all stage1
#   bash Evaluation/run_eval.sh both all stage2
#
# Outputs (for $RESULTS_ROOT from the config):
#   $RESULTS_ROOT/
#       craft_s1/<ds>/restored/*.png, metrics.json, metrics.csv
#       craft_s2/<ds>/restored/*.png, metrics.json, metrics.csv
#       osdface_s1/<ds>/restored/*.png, metrics.json, metrics.csv
#       osdface_s2/<ds>/restored/*.png, metrics.json, metrics.csv
#       compare_<ds>_stage1.{json,md}
#       compare_<ds>_stage2.{json,md}

set -euo pipefail

MODE="${1:-both}"           # celeba | lfw | both
DO="${2:-all}"              # all | score_only
STAGES="${3:-both_stages}"  # both_stages | stage1 | stage2

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$EVAL_DIR")"
cd "$REPO_ROOT"

# Tiny YAML-reader helper (nested one level deep).
read_cfg () {
  local cfg="$1" key="$2"
  python - "$cfg" "$key" <<'PY'
import sys, yaml
cfg, key = sys.argv[1], sys.argv[2]
with open(cfg) as f:
    c = yaml.safe_load(f)
for part in key.split('.'):
    c = c.get(part, "") if isinstance(c, dict) else ""
print("" if c is None else c)
PY
}

has_stage () {
  [[ "$STAGES" == "both_stages" || "$STAGES" == "$1" ]]
}

run_one () {
  local DS="$1"
  local CFG="$EVAL_DIR/configs/eval_${DS}.yaml"
  [[ -f "$CFG" ]] || { echo "Config not found: $CFG" >&2; exit 1; }

  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  Dataset: $DS           config: $CFG"
  echo "════════════════════════════════════════════════════════════════"

  local LQ_DIR      ; LQ_DIR=$(read_cfg   "$CFG" lq_dir)
  local HQ_DIR      ; HQ_DIR=$(read_cfg   "$CFG" hq_dir)
  local FFHQ_DIR    ; FFHQ_DIR=$(read_cfg "$CFG" ffhq_dir)
  local PARSER_CKPT ; PARSER_CKPT=$(read_cfg "$CFG" parser_ckpt)
  local RES_ROOT    ; RES_ROOT=$(read_cfg    "$CFG" results_root)

  # CRAFT
  local CRAFT_S1    ; CRAFT_S1=$(read_cfg "$CFG" craft.stage1_ckpt)
  local CRAFT_S2    ; CRAFT_S2=$(read_cfg "$CFG" craft.stage2_ckpt)
  local CRAFT_EMBED ; CRAFT_EMBED=$(read_cfg "$CFG" craft.embed_dim)
  local CRAFT_RQ    ; CRAFT_RQ=$(read_cfg    "$CFG" craft.rq_levels)

  # OSDFace
  local OSD_S1      ; OSD_S1=$(read_cfg     "$CFG" osdface.stage1_ckpt)
  local OSD_S2      ; OSD_S2=$(read_cfg     "$CFG" osdface.stage2_ckpt)
  local OSD_EMBED   ; OSD_EMBED=$(read_cfg  "$CFG" osdface.embed_dim)
  local OSD_NCODE   ; OSD_NCODE=$(read_cfg  "$CFG" osdface.lq_n_codes)

  # Stage-2 common
  local SD_MODEL    ; SD_MODEL=$(read_cfg     "$CFG" stage2.pretrained_model)
  local SD_PREC     ; SD_PREC=$(read_cfg      "$CFG" stage2.mixed_precision)
  local SD_MERGE    ; SD_MERGE=$(read_cfg     "$CFG" stage2.merge_lora)
  local SD_PROMPTS  ; SD_PROMPTS=$(read_cfg   "$CFG" stage2.prompts_json)
  local SD_RANK     ; SD_RANK=$(read_cfg      "$CFG" stage2.lora_rank)
  local SD_ALPHA    ; SD_ALPHA=$(read_cfg     "$CFG" stage2.lora_alpha)
  local SD_TFIXED   ; SD_TFIXED=$(read_cfg    "$CFG" stage2.t_fixed)
  local SD_CTXDIM   ; SD_CTXDIM=$(read_cfg    "$CFG" stage2.context_dim)

  local MERGE_FLAG=""
  if [[ "$SD_MERGE" == "True" || "$SD_MERGE" == "true" ]]; then
    MERGE_FLAG="--merge_lora"
  fi
  local PROMPTS_FLAG=""
  if [[ -n "$SD_PROMPTS" && "$SD_PROMPTS" != "None" ]]; then
    PROMPTS_FLAG="--prompts_json $SD_PROMPTS"
  fi

  local HQ_FLAG=""   ; [[ -n "$HQ_DIR"   ]] && HQ_FLAG="--hq_dir $HQ_DIR"
  local FFHQ_FLAG="" ; [[ -n "$FFHQ_DIR" ]] && FFHQ_FLAG="--ffhq_dir $FFHQ_DIR"

  # Output dirs
  local DIR_CS1="$RES_ROOT/craft_s1/$DS/restored"
  local DIR_CS2="$RES_ROOT/craft_s2/$DS/restored"
  local DIR_OS1="$RES_ROOT/osdface_s1/$DS/restored"
  local DIR_OS2="$RES_ROOT/osdface_s2/$DS/restored"
  local JSON_CS1="$RES_ROOT/craft_s1/$DS/metrics.json"
  local JSON_CS2="$RES_ROOT/craft_s2/$DS/metrics.json"
  local JSON_OS1="$RES_ROOT/osdface_s1/$DS/metrics.json"
  local JSON_OS2="$RES_ROOT/osdface_s2/$DS/metrics.json"
  mkdir -p "$DIR_CS1" "$DIR_CS2" "$DIR_OS1" "$DIR_OS2"

  # ─── INFERENCE ──────────────────────────────────────────────────────
  if [[ "$DO" != "score_only" ]]; then

    if has_stage stage1; then
      echo ""
      echo "[inference] CRAFT Stage-1 → $DIR_CS1"
      python -m Evaluation.inference.infer_craft \
        --stage 1 \
        --stage1_ckpt "$CRAFT_S1" \
        --parser_ckpt "$PARSER_CKPT" \
        --input_dir   "$LQ_DIR" \
        --output_dir  "$DIR_CS1" \
        --embed_dim   "$CRAFT_EMBED" \
        --rq_levels   "$CRAFT_RQ"

      echo ""
      echo "[inference] OSDFace Stage-1 → $DIR_OS1"
      python -m Evaluation.inference.infer_osdface \
        --stage 1 \
        --stage1_ckpt "$OSD_S1" \
        --input_dir   "$LQ_DIR" \
        --output_dir  "$DIR_OS1" \
        --embed_dim   "$OSD_EMBED" \
        --lq_n_codes  "$OSD_NCODE"
    fi

    if has_stage stage2; then
      local TFIXED_FLAG=""  ; [[ -n "$SD_TFIXED" ]] && TFIXED_FLAG="--t_fixed $SD_TFIXED"
      local CTXDIM_FLAG=""  ; [[ -n "$SD_CTXDIM" ]] && CTXDIM_FLAG="--context_dim $SD_CTXDIM"

      echo ""
      echo "[inference] CRAFT Stage-2 → $DIR_CS2"
      # shellcheck disable=SC2086
      python -m Evaluation.inference.infer_craft \
        --stage 2 \
        --stage1_ckpt "$CRAFT_S1" \
        --stage2_ckpt "$CRAFT_S2" \
        --parser_ckpt "$PARSER_CKPT" \
        --input_dir   "$LQ_DIR" \
        --output_dir  "$DIR_CS2" \
        --embed_dim   "$CRAFT_EMBED" \
        --pretrained_model "$SD_MODEL" \
        --mixed_precision  "$SD_PREC" \
        --lora_rank  "$SD_RANK" --lora_alpha "$SD_ALPHA" \
        $TFIXED_FLAG $CTXDIM_FLAG $MERGE_FLAG $PROMPTS_FLAG

      echo ""
      echo "[inference] OSDFace Stage-2 → $DIR_OS2"
      # shellcheck disable=SC2086
      python -m Evaluation.inference.infer_osdface \
        --stage 2 \
        --stage1_ckpt "$OSD_S1" \
        --stage2_ckpt "$OSD_S2" \
        --input_dir   "$LQ_DIR" \
        --output_dir  "$DIR_OS2" \
        --embed_dim   "$OSD_EMBED" \
        --lq_n_codes  "$OSD_NCODE" \
        --pretrained_model "$SD_MODEL" \
        --mixed_precision  "$SD_PREC" \
        --lora_rank  "$SD_RANK" --lora_alpha "$SD_ALPHA" \
        $TFIXED_FLAG $CTXDIM_FLAG $MERGE_FLAG $PROMPTS_FLAG
    fi
  fi

  # ─── SCORING ────────────────────────────────────────────────────────
  score () {
    local DIR="$1" OUT="$2" TAG="$3"
    echo ""
    echo "[score] $TAG  → $OUT"
    # shellcheck disable=SC2086
    python -m Evaluation.evaluate \
      --restored_dir "$DIR" \
      --out_json     "$OUT" \
      $HQ_FLAG $FFHQ_FLAG
  }

  if has_stage stage1; then
    score "$DIR_CS1" "$JSON_CS1" "CRAFT-S1"
    score "$DIR_OS1" "$JSON_OS1" "OSDFace-S1"
    echo ""
    echo "[compare] stage 1 — $RES_ROOT/compare_${DS}_stage1.json"
    python -m Evaluation.compare \
      --craft_json   "$JSON_CS1" \
      --osdface_json "$JSON_OS1" \
      --out_json     "$RES_ROOT/compare_${DS}_stage1.json" \
      --dataset      "${DS}-stage1"
  fi

  if has_stage stage2; then
    score "$DIR_CS2" "$JSON_CS2" "CRAFT-S2"
    score "$DIR_OS2" "$JSON_OS2" "OSDFace-S2"
    echo ""
    echo "[compare] stage 2 — $RES_ROOT/compare_${DS}_stage2.json"
    python -m Evaluation.compare \
      --craft_json   "$JSON_CS2" \
      --osdface_json "$JSON_OS2" \
      --out_json     "$RES_ROOT/compare_${DS}_stage2.json" \
      --dataset      "${DS}-stage2"
  fi
}

case "$MODE" in
  celeba) run_one celeba ;;
  lfw)    run_one lfw    ;;
  both)   run_one celeba ; run_one lfw ;;
  *) echo "Usage: $0 {celeba|lfw|both} [all|score_only] [both_stages|stage1|stage2]" >&2; exit 1 ;;
esac

echo ""
echo "All done."
