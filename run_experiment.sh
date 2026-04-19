#!/bin/bash

# 确保出错后继续执行（默认行为，显式声明更清晰）
set +e

echo "Starting experiments..."

# === Cricket 数据集 ===
# python UEA.py Cricket -lim=0.25 -de=0.25GB -b=8 || echo "⚠️ Failed: Cricket -lim=0.25"
# python UEA.py Cricket -lim=0.5 -de=0.5GB -b=8 || echo "⚠️ Failed: Cricket -lim=0.5"
# python UEA.py Cricket -lim=1 -de=1GB -b=8 || echo "⚠️ Failed: Cricket -lim=1"
# python UEA.py Cricket -lim=1.5 -de=1.5GB -b=8 || echo "⚠️ Failed: Cricket -lim=1.5"
# python UEA.py Cricket -lim=2 -de=2GB -b=8 || echo "⚠️ Failed: Cricket -lim=2"
# python UEA.py Cricket -lim=2.5 -de=2.5GB -b=8 || echo "⚠️ Failed: Cricket -lim=2.5"
# python UEA.py Cricket -lim=3 -de=3GB -b=8 || echo "⚠️ Failed: Cricket -lim=3"
# python UEA.py Cricket -lim=4 -de=4GB -b=8 || echo "⚠️ Failed: Cricket -lim=4"

# python UEA.py Cricket -lim=8 -de=8GB -b=64  -logdir=cricket_logs_monet -algo=monet|| echo "⚠️ Failed: Cricket -lim=0.5"
# python UEA.py Cricket -lim=10 -de=10GB -b=64 -logdir=cricket_logs_monet -algo=monet|| echo "⚠️ Failed: Cricket -lim=1"
# python UEA.py Cricket -lim=12 -de=12GB -b=64 -logdir=cricket_logs_monet -algo=monet|| echo "⚠️ Failed: Cricket -lim=1.5"
# python UEA.py Cricket -lim=14 -de=14GB -b=64 -logdir=cricket_logs_monet -algo=monet|| echo "⚠️ Failed: Cricket -lim=2"
# python UEA.py Cricket -lim=16 -de=16GB -b=64 -logdir=cricket_logs_monet -algo=monet|| echo "⚠️ Failed: Cricket -lim=2.5"
# python UEA.py Cricket -lim=18 -de=18GB -b=64 -logdir=cricket_logs_monet -algo=monet|| echo "⚠️ Failed: Cricket -lim=3"

 #python UEA.py Cricket -lim=18 -de=18GB -b=64  -logdir=./ -algo=oursILP|| echo "⚠️ Failed: Cricket -lim=3"
# 实验1 budget主实验
# 定义 lim 值（de 与 lim 数值相同，单位 GB）
# lim_values=(0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0)
# lim_values=(8 10 12 14 16)
lim_values=(16) #motor
# lim_values=(5 6 7 8 9 10 11 12 13 14 15)
# lim_values=(10)
# lim_values=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
# lim_values=(7 8)
logdir="trace_oursILP"
for lim in "${lim_values[@]}"; do
    # ✅ 实际运行时 lim × 0.9（浮点），用 awk 精确计算
    runtime_lim=$(awk "BEGIN {printf \"%.1f\", $lim * 1.0}")

    de="${lim}GB"
    cmd="python UEA.py MotorImagery -lim=${runtime_lim} -de=oursILP_bs2_${de} -b=2 -logdir=${logdir}"
    fail_msg="⚠️ Failed: MotorImagery -lim=${lim}"
    eval "$cmd" || echo "$fail_msg"
done



# # 实验二：不同长度
# algos=("oursILP") #18.2
# #algos=("monet" "checkmate" "mimose" "oursILP")
# limlen_values=(1300 1400 1500 1600)
# #limbs_values=(50 54 56 60 64 68 72 76 80)
# # limsc_values=(1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5)
# for algo in "${algos[@]}"; do
#     logdir="cricket_logs_${algo}"
#     for limlen in "${limlen_values[@]}"; do
#         # ✅ 实际运行时 lim × 0.9（浮点），用 awk 精确计算
#         runtime_lim=$(awk "BEGIN {printf \"%.1f\", $limsc * 1.0}")

#         de="${limlen}len"
#         cmd="python UEA.py Cricket -lim=18 -de=${algo}_bs64_${de} -b=64 -len=${limlen} -logdir=${logdir} -algo=${algo}"
#         fail_msg="⚠️ Failed: Cricket -algo=${algo} -lim=${limlen}"
#         eval "$cmd" || echo "$fail_msg"
#     done
# done


# #实验三，不同bs
# algos=("oursILP") #18.2
# #algos=("monet" "checkmate" "mimose" "oursILP")
# #limlen_values=(1300 1400 1500 1600)
# limbs_values=(68 72)
# # limsc_values=(1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5)
# for algo in "${algos[@]}"; do
#     logdir="cricket_logs_${algo}"
#     for limbs in "${limbs_values[@]}"; do
#         # ✅ 实际运行时 lim × 0.9（浮点），用 awk 精确计算
#         runtime_lim=$(awk "BEGIN {printf \"%.1f\", $limsc * 1.0}")

#         de="${limbs}bs"
#         cmd="python UEA.py Cricket -lim=18 -de=${algo}_bsvariable_${de} -b=${limbs} -logdir=${logdir} -algo=${algo}"
#         fail_msg="⚠️ Failed: Cricket -algo=${algo} -lim=${limbs}"
#         eval "$cmd" || echo "$fail_msg"
#     done
# done

# #实验四，不同dim
# algos=("oursILP") #18.2
# #algos=("monet" "checkmate" "mimose" "oursILP")
# #limlen_values=(1300 1400 1500 1600)
# limdim_values=(4 5 6 7 8)
# # limsc_values=(1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5)
# for algo in "${algos[@]}"; do
#     logdir="cricket_logs_${algo}"
#     for limdim in "${limdim_values[@]}"; do
#         # ✅ 实际运行时 lim × 0.9（浮点），用 awk 精确计算
#         #runtime_lim=$(awk "BEGIN {printf \"%.1f\", $limsc * 1.0}")

#         de="${limdim}dim"
#         cmd="python UEA.py Cricket -lim=18 -de=${algo}_bs64_${de} -b=64 -logdir=${logdir} -dim=${limdim} -algo=${algo}"
#         fail_msg="⚠️ Failed: Cricket -algo=${algo} -lim=${limdim}"
#         eval "$cmd" || echo "$fail_msg"
#     done
# done



# # # === PEMS-SF 数据集 ===

# # python UEA.py PEMS-SF -lim=1 -de=1GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=1"
# # python UEA.py PEMS-SF -lim=2 -de=2GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=2"
# # python UEA.py PEMS-SF -lim=3 -de=3GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=3"
# # python UEA.py PEMS-SF -lim=4 -de=4GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=4"
# # python UEA.py PEMS-SF -lim=5 -de=5GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=5"
# # python UEA.py PEMS-SF -lim=6 -de=6GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=6"
# # # python UEA.py PEMS-SF -lim=8 -de=8GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=8"
# # # python UEA.py PEMS-SF -lim=12 -de=12GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=12"
# # # python UEA.py PEMS-SF -lim=16 -de=16GB -b=8 || echo "⚠️ Failed: PEMS-SF -lim=16"



# # # === DuckDuckGeese 数据集 ===
# # # python UEA.py DuckDuckGeese -lim=0.5 -de=0.5GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=0.5"
# # # python UEA.py DuckDuckGeese -lim=1 -de=1GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=1"
# # # python UEA.py DuckDuckGeese -lim=2 -de=2GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=2"
# # # python UEA.py DuckDuckGeese -lim=4 -de=4GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=4"
# # # python UEA.py DuckDuckGeese -lim=8 -de=8GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=8"
# # python UEA.py DuckDuckGeese -lim=10 -de=10GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=10"
# # python UEA.py DuckDuckGeese -lim=12 -de=12GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=12"
# # python UEA.py DuckDuckGeese -lim=14 -de=14GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=14"
# # python UEA.py DuckDuckGeese -lim=16 -de=16GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=16"
# # python UEA.py DuckDuckGeese -lim=18 -de=18GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=18"
# # python UEA.py DuckDuckGeese -lim=20 -de=20GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=20"
# # python UEA.py DuckDuckGeese -lim=22 -de=22GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=22"
# # python UEA.py DuckDuckGeese -lim=24 -de=24GB -b=8 || echo "⚠️ Failed: DuckDuckGeese -lim=24"

# # # === MotorImagery 数据集 ===

# # # python UEA.py MotorImagery -lim=2 -de=2GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=2"
# # # python UEA.py MotorImagery -lim=4 -de=4GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=4"
# # # python UEA.py MotorImagery -lim=6 -de=6GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=6"
# # # python UEA.py MotorImagery -lim=8 -de=8GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=8"
# # python UEA.py MotorImagery -lim=10 -de=10GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=10"
# # python UEA.py MotorImagery -lim=12 -de=12GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=12"
# # python UEA.py MotorImagery -lim=14 -de=14GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=14"
# # python UEA.py MotorImagery -lim=16 -de=16GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=16"
# # python UEA.py MotorImagery -lim=18 -de=18GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=18"
# # python UEA.py MotorImagery -lim=20 -de=20GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=20"
# # python UEA.py MotorImagery -lim=22 -de=22GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=22"
# # python UEA.py MotorImagery -lim=24 -de=24GB -b=2 || echo "⚠️ Failed: MotorImagery -lim=24"
# # echo "All experiments completed (with possible failures)."