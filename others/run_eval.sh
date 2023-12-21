#!/bin/bash

model=2023-09-17-144020_dp0.1_lr5e-5_sta3_ep3
qa_name=qa60
OPENAI_API_KEY="" python eval/eval.py \
    --question eval/"$qa_name"_questions.json \
    --context anno/scanrefer_val_content.json \
    --answer-list \
    eval/"$qa_name"_gpt4_answer.json \
    eval/"$qa_name"_"$model"_answer.json \
    --rule eval/rule.json \
    --output eval/review_"$qa_name"_"$model".jsonl