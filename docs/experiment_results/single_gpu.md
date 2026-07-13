# Agentrix Main Experiment Results

## Performance and Resource Metrics

| Model | Dataset | Prefix | Branches | Variant | Output tok/s | TTFT P50/P95/P99 ms | TPOT P50/P95/P99 ms | Latency P50/P95/P99 ms | GPU compute | Memory BW | Peak GPU KV GiB (%) | Peak total KV GiB | Logical KV read reduction |
|---|---|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|
| llama3.2-1b | agencybench | 16384 | 16 | flash_no_offload | 49.17 | 39515.05/77930.15/81392.82 | 5.53/5.68/5.74 | 39858.85/78275.70/81740.78 | 99.22% | 39.43% | 0.54 (65.16%) | 0.54 | 0.00% |
| llama3.2-1b | agencybench | 16384 | 16 | flash_ordinary_offload | 142.15 | 11829.33/23335.66/24418.53 | 5.57/5.60/6.36 | 12181.30/23686.13/24771.42 | 97.54% | 75.81% | 0.54 (65.16%) | 5.20 | 0.00% |
| llama3.2-1b | agencybench | 16384 | 16 | fork_no_offload | 49.65 | 38776.56/77009.19/80546.21 | 5.57/5.62/7.07 | 39127.83/77358.65/80895.08 | 98.41% | 40.00% | 0.54 (65.27%) | 0.54 | 96.22% |
| llama3.2-1b | agencybench | 16384 | 16 | fork_optimized_offload | 243.29 | 5090.40/10803.72/11729.04 | 20.07/23.35/23.84 | 6561.37/11638.28/12081.15 | 60.21% | 31.41% | 0.59 (71.51%) | 5.16 | 96.22% |
| llama3.2-1b | agencybench | 16384 | 16 | fork_ordinary_offload | 140.13 | 11795.23/23720.79/24794.27 | 5.61/5.81/7.27 | 12148.78/24077.99/25148.66 | 95.09% | 74.77% | 0.54 (65.27%) | 5.20 | 96.22% |
| llama3.2-1b | agentboard | 16384 | 16 | flash_no_offload | 48.07 | 37816.07/76375.57/79851.31 | 5.58/5.69/6.30 | 38015.06/76721.49/80193.90 | 99.40% | 38.97% | 0.54 (65.51%) | 0.54 | 0.00% |
| llama3.2-1b | agentboard | 16384 | 16 | flash_ordinary_offload | 139.58 | 11619.30/22753.36/24382.40 | 5.59/5.64/6.07 | 11967.83/23104.59/24532.98 | 97.73% | 75.69% | 0.54 (65.57%) | 5.22 | 0.00% |
| llama3.2-1b | agentboard | 16384 | 16 | fork_no_offload | 47.68 | 38522.28/77354.06/80708.94 | 5.61/5.64/5.83 | 38875.28/77708.49/80953.95 | 98.92% | 38.95% | 0.54 (65.39%) | 0.54 | 95.81% |
| llama3.2-1b | agentboard | 16384 | 16 | fork_optimized_offload | 246.42 | 5131.80/10273.79/11349.19 | 18.22/22.47/24.21 | 6619.44/11059.91/11740.50 | 58.39% | 29.73% | 0.59 (71.63%) | 5.19 | 95.81% |
| llama3.2-1b | agentboard | 16384 | 16 | fork_ordinary_offload | 138.53 | 11539.07/22825.75/24232.97 | 5.61/5.65/6.99 | 11891.03/23178.94/24585.08 | 95.45% | 73.93% | 0.54 (65.63%) | 5.23 | 95.81% |
| llama3.2-1b | appworld | 16384 | 16 | flash_no_offload | 48.82 | 35862.61/73269.19/76748.82 | 5.42/5.62/6.19 | 36205.63/73602.09/77089.82 | 99.15% | 38.39% | 0.54 (65.21%) | 0.54 | 0.00% |
| llama3.2-1b | appworld | 16384 | 16 | flash_ordinary_offload | 138.91 | 11313.81/22324.41/23551.03 | 5.56/5.59/5.61 | 11659.14/22540.54/23902.55 | 96.74% | 75.10% | 0.54 (64.92%) | 5.20 | 0.00% |
| llama3.2-1b | appworld | 16384 | 16 | fork_no_offload | 47.54 | 38110.72/75560.88/79128.97 | 5.46/5.60/6.49 | 38454.17/75911.34/79214.84 | 98.90% | 39.55% | 0.54 (64.80%) | 0.54 | 95.93% |
| llama3.2-1b | appworld | 16384 | 16 | fork_optimized_offload | 219.76 | 5556.49/11749.55/12539.10 | 20.28/22.84/23.11 | 6995.18/12581.95/13232.90 | 60.81% | 31.15% | 0.59 (70.92%) | 4.72 | 95.93% |
| llama3.2-1b | appworld | 16384 | 16 | fork_ordinary_offload | 138.63 | 11187.63/22435.93/23298.58 | 5.60/5.72/7.16 | 11541.94/22597.63/23650.14 | 96.03% | 73.62% | 0.54 (64.98%) | 5.20 | 95.93% |
| llama3.2-1b | swebench | 16384 | 16 | flash_no_offload | 50.15 | 38576.75/77043.47/80365.99 | 5.58/5.61/6.12 | 38928.46/77289.29/80714.45 | 99.68% | 40.30% | 0.53 (64.45%) | 0.53 | 0.00% |
| llama3.2-1b | swebench | 16384 | 16 | flash_ordinary_offload | 143.25 | 11693.60/23615.38/24648.91 | 5.50/5.51/5.53 | 12039.71/23961.60/24904.82 | 96.21% | 74.49% | 0.54 (64.63%) | 5.15 | 0.00% |
| llama3.2-1b | swebench | 16384 | 16 | fork_no_offload | 50.02 | 38667.53/76980.76/81467.44 | 5.56/5.68/6.56 | 39020.04/77334.90/81821.47 | 98.76% | 39.74% | 0.53 (64.45%) | 0.53 | 97.18% |
| llama3.2-1b | swebench | 16384 | 16 | fork_optimized_offload | 242.77 | 4944.89/10988.06/12521.18 | 19.91/23.55/23.56 | 6425.23/11948.58/13012.34 | 60.76% | 31.86% | 0.58 (69.98%) | 5.12 | 97.18% |
| llama3.2-1b | swebench | 16384 | 16 | fork_ordinary_offload | 140.01 | 11918.16/24055.07/25135.99 | 5.67/5.75/6.63 | 12271.12/24413.05/25492.67 | 95.88% | 74.61% | 0.54 (64.51%) | 5.16 | 97.18% |
| qwen3-1.7b | agencybench | 8192 | 8 | flash_no_offload | 88.14 | 9529.47/20935.32/22407.47 | 26.55/29.95/34.14 | 11289.08/22175.14/23341.09 | 99.60% | 45.79% | 2.90 (99.82%) | 2.90 | 0.00% |
| qwen3-1.7b | agencybench | 8192 | 8 | flash_ordinary_offload | 179.89 | 3850.68/8188.39/8725.81 | 12.84/24.96/28.34 | 4670.15/8827.07/9513.74 | 98.49% | 70.83% | 2.89 (99.41%) | 10.89 | 0.00% |
| qwen3-1.7b | agencybench | 8192 | 8 | fork_no_offload | 84.15 | 10019.46/22084.28/24119.89 | 27.60/31.47/32.42 | 11806.55/23331.72/25822.70 | 95.53% | 42.34% | 2.90 (99.94%) | 2.90 | 92.64% |
| qwen3-1.7b | agencybench | 8192 | 8 | fork_optimized_offload | 285.49 | 1744.13/3272.44/3745.90 | 21.51/35.56/48.73 | 3182.17/4324.67/4555.56 | 82.59% | 47.02% | 2.91 (100.00%) | 10.91 | 92.64% |
| qwen3-1.7b | agencybench | 8192 | 8 | fork_ordinary_offload | 160.40 | 4449.40/9570.49/10079.72 | 15.03/26.60/29.81 | 5397.87/10495.79/10788.73 | 90.42% | 63.47% | 2.89 (99.53%) | 10.89 | 92.64% |
| qwen3-1.7b | agencybench | 16384 | 16 | flash_no_offload | 30.87 | 62048.03/125000.89/132279.14 | 10.04/10.21/11.82 | 62664.55/125636.54/132910.28 | 99.47% | 40.85% | 1.90 (65.27%) | 1.90 | 0.00% |
| qwen3-1.7b | agencybench | 16384 | 16 | flash_ordinary_offload | 79.05 | 21784.80/43607.45/45526.34 | 10.07/10.24/12.35 | 22420.74/44242.67/46159.15 | 98.29% | 75.92% | 1.89 (65.16%) | 9.89 | 0.00% |
| qwen3-1.7b | agencybench | 16384 | 16 | fork_no_offload | 30.64 | 62604.98/126497.44/132078.22 | 10.07/10.24/11.46 | 63243.21/127120.56/132714.10 | 99.00% | 40.56% | 1.89 (65.16%) | 1.89 | 96.13% |
| qwen3-1.7b | agencybench | 16384 | 16 | fork_optimized_offload | 197.43 | 6311.35/11712.33/12674.99 | 19.88/22.77/22.80 | 7744.17/12640.41/13429.96 | 80.21% | 43.13% | 2.11 (72.75%) | 10.11 | 96.13% |
| qwen3-1.7b | agencybench | 16384 | 16 | fork_ordinary_offload | 78.79 | 21446.97/43610.42/45614.24 | 10.11/10.25/11.50 | 22082.59/44246.41/46255.11 | 97.71% | 75.06% | 1.89 (65.16%) | 9.89 | 96.13% |
| qwen3-1.7b | agentboard | 8192 | 8 | flash_no_offload | 84.19 | 10258.11/23025.27/24471.23 | 28.74/31.86/33.97 | 12160.80/24950.04/25716.59 | 99.25% | 44.47% | 2.89 (99.41%) | 2.89 | 0.00% |
| qwen3-1.7b | agentboard | 8192 | 8 | flash_ordinary_offload | 170.58 | 3902.82/8924.21/9008.98 | 14.02/23.60/29.48 | 4747.29/9806.83/9859.46 | 97.48% | 66.82% | 2.89 (99.59%) | 10.89 | 0.00% |
| qwen3-1.7b | agentboard | 8192 | 8 | fork_no_offload | 84.15 | 9854.28/23569.26/24592.49 | 28.57/32.29/34.63 | 11796.97/24976.68/25902.61 | 97.72% | 44.07% | 2.89 (99.59%) | 2.89 | 90.53% |
| qwen3-1.7b | agentboard | 8192 | 8 | fork_optimized_offload | 265.81 | 1536.90/4015.44/4053.72 | 21.68/30.12/53.69 | 2958.39/4936.72/4960.67 | 85.97% | 49.00% | 2.90 (99.76%) | 10.90 | 90.53% |
| qwen3-1.7b | agentboard | 8192 | 8 | fork_ordinary_offload | 169.08 | 3400.71/8963.36/9278.26 | 15.37/23.34/30.39 | 4387.52/9820.13/9997.21 | 93.38% | 64.51% | 2.89 (99.35%) | 10.89 | 90.53% |
| qwen3-1.7b | agentboard | 8192 | 16 | flash_no_offload | 86.18 | 19750.04/47180.61/49770.99 | 29.56/34.36/35.51 | 21718.64/49064.98/51385.96 | 99.22% | 44.76% | 2.89 (99.41%) | 2.89 | 0.00% |
| qwen3-1.7b | agentboard | 8192 | 16 | flash_ordinary_offload | 185.14 | 8024.09/18870.98/19821.13 | 13.79/33.70/35.19 | 8921.16/19737.36/20554.13 | 98.22% | 70.77% | 2.90 (99.88%) | 10.90 | 0.00% |
| qwen3-1.7b | agentboard | 8192 | 16 | fork_no_offload | 83.78 | 19811.10/49420.04/52101.22 | 29.79/32.95/34.46 | 21720.16/51439.99/53671.05 | 97.06% | 42.56% | 2.89 (99.35%) | 2.89 | 92.48% |
| qwen3-1.7b | agentboard | 8192 | 16 | fork_optimized_offload | 340.84 | 2667.22/7742.20/8340.77 | 19.74/55.47/81.44 | 4889.23/8768.05/9081.48 | 83.79% | 48.65% | 2.90 (99.82%) | 10.90 | 92.48% |
| qwen3-1.7b | agentboard | 8192 | 16 | fork_ordinary_offload | 184.64 | 8115.93/19418.61/21282.62 | 14.47/16.89/29.14 | 9052.68/20265.36/22254.48 | 93.10% | 68.60% | 2.89 (99.41%) | 10.89 | 92.48% |
| qwen3-1.7b | agentboard | 16384 | 8 | flash_no_offload | 31.12 | 27159.86/66860.39/71057.49 | 10.76/30.96/31.71 | 27839.97/67537.53/71734.33 | 99.31% | 41.13% | 2.07 (71.16%) | 2.07 | 0.00% |
| qwen3-1.7b | agentboard | 16384 | 8 | flash_ordinary_offload | 73.15 | 8713.83/21914.14/23364.04 | 10.65/30.23/30.91 | 9381.98/22586.29/24026.56 | 98.24% | 68.32% | 2.07 (71.28%) | 10.07 | 0.00% |
| qwen3-1.7b | agentboard | 16384 | 8 | fork_no_offload | 32.10 | 26970.14/65282.19/68792.62 | 10.74/14.83/15.82 | 27656.13/65964.70/69467.79 | 99.29% | 40.14% | 2.06 (70.92%) | 2.06 | 91.89% |
| qwen3-1.7b | agentboard | 16384 | 8 | fork_optimized_offload | 141.36 | 3186.66/6079.50/8180.03 | 16.83/21.63/23.21 | 4292.36/7111.58/8851.29 | 86.15% | 42.33% | 2.08 (71.51%) | 10.08 | 91.89% |
| qwen3-1.7b | agentboard | 16384 | 8 | fork_ordinary_offload | 74.84 | 8546.63/21624.01/23016.44 | 10.56/14.68/15.61 | 9209.66/22294.59/23675.76 | 96.96% | 66.78% | 2.07 (71.16%) | 10.07 | 91.89% |
| qwen3-1.7b | agentboard | 16384 | 16 | flash_no_offload | 33.30 | 53988.38/126612.65/136879.17 | 10.40/56.16/58.24 | 54657.09/127247.64/137539.62 | 99.73% | 40.19% | 2.24 (77.22%) | 2.24 | 0.00% |
| qwen3-1.7b | agentboard | 16384 | 16 | flash_ordinary_offload | 81.27 | 19045.97/44820.91/47100.03 | 10.46/56.59/58.75 | 19711.22/45487.31/47753.58 | 98.25% | 73.92% | 2.25 (77.40%) | 10.25 | 0.00% |
| qwen3-1.7b | agentboard | 16384 | 16 | fork_no_offload | 32.61 | 56206.93/130821.01/137543.45 | 10.43/14.59/18.68 | 56870.90/131478.24/138196.52 | 99.23% | 39.68% | 2.25 (77.52%) | 2.25 | 93.87% |
| qwen3-1.7b | agentboard | 16384 | 16 | fork_optimized_offload | 190.64 | 5591.64/12069.72/13216.67 | 20.87/26.30/29.59 | 7094.14/13027.44/14201.02 | 79.91% | 42.95% | 2.24 (77.22%) | 10.24 | 93.88% |
| qwen3-1.7b | agentboard | 16384 | 16 | fork_ordinary_offload | 83.12 | 19144.80/44769.15/47050.64 | 10.47/15.40/18.19 | 19794.73/45428.85/47705.65 | 97.67% | 75.27% | 2.24 (77.22%) | 10.24 | 93.88% |
| qwen3-1.7b | appworld | 8192 | 8 | flash_no_offload | 88.80 | 8743.81/21420.79/23304.45 | 26.93/30.35/31.08 | 10637.43/22643.06/24560.80 | 98.70% | 46.66% | 2.90 (99.94%) | 2.90 | 0.00% |
| qwen3-1.7b | appworld | 8192 | 8 | flash_ordinary_offload | 177.84 | 3798.94/8440.07/8794.31 | 13.31/25.78/28.27 | 4621.97/9218.03/9564.89 | 98.08% | 67.81% | 2.90 (99.88%) | 10.90 | 0.00% |
| qwen3-1.7b | appworld | 8192 | 8 | fork_no_offload | 82.16 | 9731.51/23453.96/25303.38 | 28.00/31.84/33.12 | 11706.82/25072.80/26285.42 | 95.21% | 43.35% | 2.89 (99.35%) | 2.89 | 91.95% |
| qwen3-1.7b | appworld | 8192 | 8 | fork_optimized_offload | 288.76 | 1301.45/3210.70/3227.44 | 21.05/45.35/56.08 | 2985.04/4066.47/4091.46 | 82.35% | 46.96% | 2.90 (99.82%) | 10.90 | 91.95% |
| qwen3-1.7b | appworld | 8192 | 8 | fork_ordinary_offload | 161.44 | 4043.91/9457.88/10353.09 | 14.47/22.91/29.12 | 4992.11/10339.92/11105.01 | 91.84% | 67.18% | 2.89 (99.35%) | 10.89 | 91.95% |
| qwen3-1.7b | appworld | 8192 | 16 | flash_no_offload | 84.82 | 20023.11/46000.00/49505.14 | 27.96/30.75/34.18 | 21895.00/47857.22/50988.79 | 99.57% | 43.84% | 2.89 (99.59%) | 2.89 | 0.00% |
| qwen3-1.7b | appworld | 8192 | 16 | flash_ordinary_offload | 188.85 | 7777.75/18121.01/19013.59 | 13.46/29.13/35.38 | 8626.98/18958.59/19793.09 | 99.23% | 74.60% | 2.89 (99.35%) | 10.89 | 0.00% |
| qwen3-1.7b | appworld | 8192 | 16 | fork_no_offload | 84.77 | 20029.27/46652.53/49352.66 | 29.11/32.44/33.76 | 21924.96/48577.93/51165.67 | 95.56% | 43.01% | 2.90 (99.71%) | 2.90 | 93.77% |
| qwen3-1.7b | appworld | 8192 | 16 | fork_optimized_offload | 353.64 | 2500.69/7060.94/8099.55 | 26.99/66.41/89.00 | 5038.32/8108.27/8865.91 | 78.45% | 44.77% | 2.91 (100.00%) | 10.91 | 93.77% |
| qwen3-1.7b | appworld | 8192 | 16 | fork_ordinary_offload | 174.83 | 8710.69/19896.05/21548.22 | 14.77/17.31/28.70 | 9643.38/20802.28/22546.99 | 90.53% | 66.95% | 2.91 (100.00%) | 10.91 | 93.77% |
| qwen3-1.7b | appworld | 16384 | 8 | flash_no_offload | 32.18 | 26462.21/63313.07/67383.43 | 10.42/28.44/30.23 | 27329.81/63968.77/68046.56 | 99.50% | 41.18% | 2.02 (69.63%) | 2.02 | 0.00% |
| qwen3-1.7b | appworld | 16384 | 8 | flash_ordinary_offload | 71.32 | 9491.39/22517.11/23724.25 | 10.72/28.53/30.85 | 10314.66/23192.30/24407.07 | 98.22% | 68.51% | 2.02 (69.69%) | 10.02 | 0.00% |
| qwen3-1.7b | appworld | 16384 | 8 | fork_no_offload | 30.66 | 28603.42/67165.63/70596.70 | 10.64/12.94/14.62 | 29275.71/67833.78/71261.03 | 99.05% | 40.50% | 2.03 (70.04%) | 2.03 | 93.40% |
| qwen3-1.7b | appworld | 16384 | 8 | fork_optimized_offload | 131.03 | 3963.11/8073.24/8278.45 | 17.84/21.80/21.85 | 4639.11/8871.42/9013.17 | 86.02% | 46.02% | 2.02 (69.69%) | 10.02 | 93.40% |
| qwen3-1.7b | appworld | 16384 | 8 | fork_ordinary_offload | 72.67 | 9444.80/22258.50/23640.31 | 10.71/13.39/14.81 | 10119.45/22932.48/24313.88 | 97.75% | 68.28% | 2.01 (69.16%) | 10.01 | 93.40% |
| qwen3-1.7b | appworld | 16384 | 16 | flash_no_offload | 31.19 | 59985.99/132211.16/138665.26 | 10.53/53.70/57.67 | 60657.03/132869.82/139326.56 | 99.73% | 40.39% | 2.20 (75.87%) | 2.20 | 0.00% |
| qwen3-1.7b | appworld | 16384 | 16 | flash_ordinary_offload | 80.12 | 19877.52/44621.03/46841.27 | 10.43/52.47/57.43 | 20531.50/45272.50/47497.45 | 98.74% | 72.94% | 2.22 (76.28%) | 10.22 | 0.00% |
| qwen3-1.7b | appworld | 16384 | 16 | fork_no_offload | 31.61 | 59244.62/130668.29/137883.12 | 10.51/14.89/17.75 | 59903.51/131323.27/138552.41 | 99.29% | 40.45% | 2.20 (75.63%) | 2.20 | 95.26% |
| qwen3-1.7b | appworld | 16384 | 16 | fork_optimized_offload | 186.82 | 5285.90/13171.44/14476.96 | 21.18/26.16/29.62 | 6733.41/13946.67/15314.37 | 78.32% | 42.94% | 2.20 (75.69%) | 10.20 | 95.26% |
| qwen3-1.7b | appworld | 16384 | 16 | fork_ordinary_offload | 81.71 | 19934.46/44621.08/46799.23 | 10.42/14.85/17.81 | 20597.46/45273.41/47457.03 | 97.38% | 73.18% | 2.17 (74.81%) | 10.17 | 95.26% |
| qwen3-1.7b | swebench | 16384 | 16 | flash_no_offload | 31.11 | 62840.97/124448.97/130008.08 | 9.51/10.21/12.24 | 63462.24/125041.33/130599.50 | 99.75% | 41.45% | 2.03 (69.75%) | 2.03 | 0.00% |
| qwen3-1.7b | swebench | 16384 | 16 | flash_ordinary_offload | 84.38 | 19905.75/40584.86/42450.86 | 9.43/9.75/12.64 | 20495.56/41180.62/43045.97 | 98.42% | 78.75% | 2.02 (69.63%) | 10.02 | 0.00% |
| qwen3-1.7b | swebench | 16384 | 16 | fork_no_offload | 31.61 | 61122.83/122676.67/128144.77 | 9.52/9.81/11.02 | 61719.20/123275.90/128743.95 | 98.91% | 38.54% | 2.02 (69.69%) | 2.02 | 94.23% |
| qwen3-1.7b | swebench | 16384 | 16 | fork_optimized_offload | 188.86 | 6464.57/12305.52/13465.40 | 20.57/25.64/25.65 | 7768.52/13323.29/14192.92 | 76.75% | 43.76% | 2.23 (76.93%) | 10.23 | 94.23% |
| qwen3-1.7b | swebench | 16384 | 16 | fork_ordinary_offload | 78.15 | 21885.58/43992.57/45941.59 | 10.18/10.47/11.88 | 22523.50/44630.10/46586.40 | 97.25% | 74.25% | 2.02 (69.69%) | 10.02 | 94.23% |

## Baseline Deltas

| Model | Dataset | Prefix | Branches | Variant | Baseline | Output throughput change | Peak GPU KV reduction | Peak total KV reduction |
|---|---|---:|---:|---|---|---:|---:|---:|
| llama3.2-1b | agencybench | 16384 | 16 | flash_ordinary_offload | flash_no_offload | 189.11% | 0.00% | -862.09% |
| llama3.2-1b | agencybench | 16384 | 16 | fork_no_offload | flash_no_offload | 0.98% | -0.18% | -0.18% |
| llama3.2-1b | agencybench | 16384 | 16 | fork_optimized_offload | fork_ordinary_offload | 73.62% | -9.56% | 0.63% |
| llama3.2-1b | agencybench | 16384 | 16 | fork_ordinary_offload | flash_ordinary_offload | -1.42% | -0.18% | 0.12% |
| llama3.2-1b | agentboard | 16384 | 16 | flash_ordinary_offload | flash_no_offload | 190.36% | -0.09% | -860.86% |
| llama3.2-1b | agentboard | 16384 | 16 | fork_no_offload | flash_no_offload | -0.80% | 0.18% | 0.18% |
| llama3.2-1b | agentboard | 16384 | 16 | fork_optimized_offload | fork_ordinary_offload | 77.88% | -9.15% | 0.76% |
| llama3.2-1b | agentboard | 16384 | 16 | fork_ordinary_offload | flash_ordinary_offload | -0.75% | -0.09% | -0.01% |
| llama3.2-1b | appworld | 16384 | 16 | flash_ordinary_offload | flash_no_offload | 184.50% | 0.45% | -859.78% |
| llama3.2-1b | appworld | 16384 | 16 | fork_no_offload | flash_no_offload | -2.62% | 0.63% | 0.63% |
| llama3.2-1b | appworld | 16384 | 16 | fork_optimized_offload | fork_ordinary_offload | 58.53% | -9.15% | 9.15% |
| llama3.2-1b | appworld | 16384 | 16 | fork_ordinary_offload | flash_ordinary_offload | -0.20% | -0.09% | -0.01% |
| llama3.2-1b | swebench | 16384 | 16 | flash_ordinary_offload | flash_no_offload | 185.65% | -0.27% | -863.15% |
| llama3.2-1b | swebench | 16384 | 16 | fork_no_offload | flash_no_offload | -0.25% | 0.00% | 0.00% |
| llama3.2-1b | swebench | 16384 | 16 | fork_optimized_offload | fork_ordinary_offload | 73.40% | -8.49% | 0.62% |
| llama3.2-1b | swebench | 16384 | 16 | fork_ordinary_offload | flash_ordinary_offload | -2.26% | 0.18% | -0.07% |
| qwen3-1.7b | agencybench | 8192 | 8 | flash_ordinary_offload | flash_no_offload | 104.09% | 0.41% | -275.44% |
| qwen3-1.7b | agencybench | 8192 | 8 | fork_no_offload | flash_no_offload | -4.53% | -0.12% | -0.12% |
| qwen3-1.7b | agencybench | 8192 | 8 | fork_optimized_offload | fork_ordinary_offload | 77.99% | -0.47% | -0.13% |
| qwen3-1.7b | agencybench | 8192 | 8 | fork_ordinary_offload | flash_ordinary_offload | -10.83% | -0.12% | -0.03% |
| qwen3-1.7b | agencybench | 16384 | 16 | flash_ordinary_offload | flash_no_offload | 156.04% | 0.18% | -421.68% |
| qwen3-1.7b | agencybench | 16384 | 16 | fork_no_offload | flash_no_offload | -0.76% | 0.18% | 0.18% |
| qwen3-1.7b | agencybench | 16384 | 16 | fork_optimized_offload | fork_ordinary_offload | 150.57% | -11.65% | -2.23% |
| qwen3-1.7b | agencybench | 16384 | 16 | fork_ordinary_offload | flash_ordinary_offload | -0.33% | 0.00% | 0.00% |
| qwen3-1.7b | agentboard | 8192 | 8 | flash_ordinary_offload | flash_no_offload | 102.60% | -0.18% | -277.17% |
| qwen3-1.7b | agentboard | 8192 | 8 | fork_no_offload | flash_no_offload | -0.06% | -0.18% | -0.18% |
| qwen3-1.7b | agentboard | 8192 | 8 | fork_optimized_offload | fork_ordinary_offload | 57.21% | -0.41% | -0.11% |
| qwen3-1.7b | agentboard | 8192 | 8 | fork_ordinary_offload | flash_ordinary_offload | -0.88% | 0.24% | 0.06% |
| qwen3-1.7b | agentboard | 8192 | 16 | flash_ordinary_offload | flash_no_offload | 114.82% | -0.47% | -277.47% |
| qwen3-1.7b | agentboard | 8192 | 16 | fork_no_offload | flash_no_offload | -2.79% | 0.06% | 0.06% |
| qwen3-1.7b | agentboard | 8192 | 16 | fork_optimized_offload | fork_ordinary_offload | 84.59% | -0.41% | -0.11% |
| qwen3-1.7b | agentboard | 8192 | 16 | fork_ordinary_offload | flash_ordinary_offload | -0.27% | 0.47% | 0.13% |
| qwen3-1.7b | agentboard | 16384 | 8 | flash_ordinary_offload | flash_no_offload | 135.08% | -0.17% | -387.13% |
| qwen3-1.7b | agentboard | 16384 | 8 | fork_no_offload | flash_no_offload | 3.17% | 0.33% | 0.33% |
| qwen3-1.7b | agentboard | 16384 | 8 | fork_optimized_offload | fork_ordinary_offload | 88.88% | -0.50% | -0.10% |
| qwen3-1.7b | agentboard | 16384 | 8 | fork_ordinary_offload | flash_ordinary_offload | 2.31% | 0.17% | 0.03% |
| qwen3-1.7b | agentboard | 16384 | 16 | flash_ordinary_offload | flash_no_offload | 144.08% | -0.23% | -356.81% |
| qwen3-1.7b | agentboard | 16384 | 16 | fork_no_offload | flash_no_offload | -2.05% | -0.38% | -0.38% |
| qwen3-1.7b | agentboard | 16384 | 16 | fork_optimized_offload | fork_ordinary_offload | 129.35% | 0.00% | 0.00% |
| qwen3-1.7b | agentboard | 16384 | 16 | fork_ordinary_offload | flash_ordinary_offload | 2.28% | 0.23% | 0.05% |
| qwen3-1.7b | appworld | 8192 | 8 | flash_ordinary_offload | flash_no_offload | 100.28% | 0.06% | -275.46% |
| qwen3-1.7b | appworld | 8192 | 8 | fork_no_offload | flash_no_offload | -7.48% | 0.59% | 0.59% |
| qwen3-1.7b | appworld | 8192 | 8 | fork_optimized_offload | fork_ordinary_offload | 78.87% | -0.47% | -0.13% |
| qwen3-1.7b | appworld | 8192 | 8 | fork_ordinary_offload | flash_ordinary_offload | -9.22% | 0.53% | 0.14% |
| qwen3-1.7b | appworld | 8192 | 16 | flash_ordinary_offload | flash_no_offload | 122.64% | 0.24% | -276.26% |
| qwen3-1.7b | appworld | 8192 | 16 | fork_no_offload | flash_no_offload | -0.06% | -0.12% | -0.12% |
| qwen3-1.7b | appworld | 8192 | 16 | fork_optimized_offload | fork_ordinary_offload | 102.28% | 0.00% | 0.00% |
| qwen3-1.7b | appworld | 8192 | 16 | fork_ordinary_offload | flash_ordinary_offload | -7.42% | -0.65% | -0.17% |
| qwen3-1.7b | appworld | 16384 | 8 | flash_ordinary_offload | flash_no_offload | 121.64% | -0.08% | -395.55% |
| qwen3-1.7b | appworld | 16384 | 8 | fork_no_offload | flash_no_offload | -4.73% | -0.59% | -0.59% |
| qwen3-1.7b | appworld | 16384 | 8 | fork_optimized_offload | fork_ordinary_offload | 80.30% | -0.77% | -0.15% |
| qwen3-1.7b | appworld | 16384 | 8 | fork_ordinary_offload | flash_ordinary_offload | 1.89% | 0.76% | 0.15% |
| qwen3-1.7b | appworld | 16384 | 16 | flash_ordinary_offload | flash_no_offload | 156.84% | -0.54% | -363.49% |
| qwen3-1.7b | appworld | 16384 | 16 | fork_no_offload | flash_no_offload | 1.33% | 0.31% | 0.31% |
| qwen3-1.7b | appworld | 16384 | 16 | fork_optimized_offload | fork_ordinary_offload | 128.63% | -1.18% | -0.25% |
| qwen3-1.7b | appworld | 16384 | 16 | fork_ordinary_offload | flash_ordinary_offload | 1.99% | 1.93% | 0.42% |
| qwen3-1.7b | swebench | 16384 | 16 | flash_ordinary_offload | flash_no_offload | 171.23% | 0.17% | -394.63% |
| qwen3-1.7b | swebench | 16384 | 16 | fork_no_offload | flash_no_offload | 1.59% | 0.08% | 0.08% |
| qwen3-1.7b | swebench | 16384 | 16 | fork_optimized_offload | fork_ordinary_offload | 141.66% | -10.39% | -2.10% |
| qwen3-1.7b | swebench | 16384 | 16 | fork_ordinary_offload | flash_ordinary_offload | -7.39% | -0.08% | -0.02% |

## Offload Traffic

| Model | Dataset | Prefix | Branches | Variant | Peak CPU KV GiB (%) | Load GiB | Load ops (avg MiB) | Store GiB | Store ops (avg MiB) | Load reduction vs baseline |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| llama3.2-1b | agencybench | 16384 | 16 | flash_ordinary_offload | 4.66 (58.28%) | 63.53 | 127 (512.25) | 4.66 | 197 (24.24) | - |
| llama3.2-1b | agencybench | 16384 | 16 | fork_optimized_offload | 4.57 (57.14%) | 9.56 | 30 (326.37) | 4.58 | 267 (17.56) | 85.03% |
| llama3.2-1b | agencybench | 16384 | 16 | fork_ordinary_offload | 4.66 (58.19%) | 63.85 | 128 (510.82) | 4.66 | 197 (24.24) | -0.51% |
| llama3.2-1b | agentboard | 16384 | 16 | flash_ordinary_offload | 4.68 (58.51%) | 63.76 | 127 (514.12) | 4.68 | 199 (24.09) | - |
| llama3.2-1b | agentboard | 16384 | 16 | fork_optimized_offload | 4.59 (57.39%) | 9.08 | 28 (332.16) | 4.60 | 266 (17.70) | 85.96% |
| llama3.2-1b | agentboard | 16384 | 16 | fork_ordinary_offload | 4.68 (58.51%) | 64.70 | 127 (521.65) | 4.68 | 199 (24.09) | -1.46% |
| llama3.2-1b | appworld | 16384 | 16 | flash_ordinary_offload | 4.66 (58.21%) | 62.86 | 129 (498.95) | 4.66 | 197 (24.21) | - |
| llama3.2-1b | appworld | 16384 | 16 | fork_optimized_offload | 4.13 (51.65%) | 8.34 | 33 (258.71) | 4.13 | 249 (16.99) | 86.66% |
| llama3.2-1b | appworld | 16384 | 16 | fork_ordinary_offload | 4.66 (58.21%) | 62.49 | 127 (503.86) | 4.66 | 197 (24.21) | 0.58% |
| llama3.2-1b | swebench | 16384 | 16 | flash_ordinary_offload | 4.62 (57.70%) | 62.42 | 127 (503.27) | 4.62 | 194 (24.40) | - |
| llama3.2-1b | swebench | 16384 | 16 | fork_optimized_offload | 4.54 (56.79%) | 9.46 | 33 (293.59) | 4.54 | 265 (17.56) | 85.15% |
| llama3.2-1b | swebench | 16384 | 16 | fork_ordinary_offload | 4.62 (57.76%) | 63.70 | 128 (509.57) | 4.62 | 193 (24.52) | -2.05% |
| qwen3-1.7b | agencybench | 8192 | 8 | flash_ordinary_offload | 8.00 (100.00%) | 166.52 | 229 (744.62) | 33.33 | 412 (82.83) | - |
| qwen3-1.7b | agencybench | 8192 | 8 | fork_optimized_offload | 8.00 (100.00%) | 30.23 | 46 (672.95) | 32.51 | 555 (59.98) | 82.14% |
| qwen3-1.7b | agencybench | 8192 | 8 | fork_ordinary_offload | 8.00 (100.00%) | 169.22 | 233 (743.68) | 33.33 | 412 (82.83) | -1.62% |
| qwen3-1.7b | agencybench | 16384 | 16 | flash_ordinary_offload | 8.00 (100.00%) | 225.92 | 127 (1821.61) | 16.39 | 198 (84.74) | - |
| qwen3-1.7b | agencybench | 16384 | 16 | fork_optimized_offload | 8.00 (100.00%) | 30.64 | 30 (1045.80) | 16.08 | 270 (61.00) | 86.10% |
| qwen3-1.7b | agencybench | 16384 | 16 | fork_ordinary_offload | 8.00 (100.00%) | 220.44 | 127 (1777.41) | 16.39 | 198 (84.74) | 2.43% |
| qwen3-1.7b | agentboard | 8192 | 8 | flash_ordinary_offload | 8.00 (100.00%) | 38.75 | 54 (734.87) | 9.52 | 117 (83.33) | - |
| qwen3-1.7b | agentboard | 8192 | 8 | fork_optimized_offload | 8.00 (100.00%) | 8.64 | 12 (737.62) | 9.28 | 155 (61.33) | 75.49% |
| qwen3-1.7b | agentboard | 8192 | 8 | fork_ordinary_offload | 8.00 (100.00%) | 35.26 | 47 (768.29) | 9.52 | 117 (83.33) | 9.01% |
| qwen3-1.7b | agentboard | 8192 | 16 | flash_ordinary_offload | 8.00 (100.00%) | 96.13 | 121 (813.52) | 10.65 | 189 (57.69) | - |
| qwen3-1.7b | agentboard | 8192 | 16 | fork_optimized_offload | 8.00 (100.00%) | 12.11 | 23 (539.23) | 10.37 | 249 (42.63) | 87.25% |
| qwen3-1.7b | agentboard | 8192 | 16 | fork_ordinary_offload | 8.00 (100.00%) | 94.97 | 122 (797.15) | 10.65 | 189 (57.69) | 1.20% |
| qwen3-1.7b | agentboard | 16384 | 8 | flash_ordinary_offload | 8.00 (100.00%) | 110.02 | 62 (1817.12) | 17.61 | 153 (117.87) | - |
| qwen3-1.7b | agentboard | 16384 | 8 | fork_optimized_offload | 8.00 (100.00%) | 19.63 | 11 (1827.64) | 17.38 | 228 (78.06) | 82.09% |
| qwen3-1.7b | agentboard | 16384 | 8 | fork_ordinary_offload | 8.00 (100.00%) | 109.65 | 63 (1782.22) | 17.61 | 153 (117.87) | 0.34% |
| qwen3-1.7b | agentboard | 16384 | 16 | flash_ordinary_offload | 8.00 (100.00%) | 226.16 | 127 (1823.56) | 18.72 | 225 (85.19) | - |
| qwen3-1.7b | agentboard | 16384 | 16 | fork_optimized_offload | 8.00 (100.00%) | 34.13 | 29 (1205.15) | 18.37 | 303 (62.09) | 85.12% |
| qwen3-1.7b | agentboard | 16384 | 16 | fork_ordinary_offload | 8.00 (100.00%) | 229.33 | 127 (1849.12) | 18.72 | 225 (85.21) | -1.40% |
| qwen3-1.7b | appworld | 8192 | 8 | flash_ordinary_offload | 8.00 (100.00%) | 60.11 | 85 (724.15) | 13.55 | 169 (82.11) | - |
| qwen3-1.7b | appworld | 8192 | 8 | fork_optimized_offload | 8.00 (100.00%) | 12.11 | 18 (688.92) | 13.24 | 239 (56.71) | 80.24% |
| qwen3-1.7b | appworld | 8192 | 8 | fork_ordinary_offload | 8.00 (100.00%) | 61.28 | 85 (738.19) | 13.55 | 168 (82.59) | -1.94% |
| qwen3-1.7b | appworld | 8192 | 16 | flash_ordinary_offload | 8.00 (100.00%) | 129.68 | 176 (754.53) | 15.18 | 272 (57.15) | - |
| qwen3-1.7b | appworld | 8192 | 16 | fork_optimized_offload | 8.00 (100.00%) | 17.96 | 37 (496.95) | 14.66 | 334 (44.94) | 86.58% |
| qwen3-1.7b | appworld | 8192 | 16 | fork_ordinary_offload | 8.00 (100.00%) | 133.77 | 180 (760.99) | 15.18 | 273 (56.94) | -3.15% |
| qwen3-1.7b | appworld | 16384 | 8 | flash_ordinary_offload | 8.00 (100.00%) | 163.62 | 95 (1763.67) | 25.03 | 218 (117.60) | - |
| qwen3-1.7b | appworld | 16384 | 8 | fork_optimized_offload | 8.00 (100.00%) | 31.44 | 26 (1238.06) | 24.75 | 329 (77.02) | 80.61% |
| qwen3-1.7b | appworld | 16384 | 8 | fork_ordinary_offload | 8.00 (100.00%) | 162.12 | 97 (1711.45) | 25.03 | 218 (117.60) | 0.92% |
| qwen3-1.7b | appworld | 16384 | 16 | flash_ordinary_offload | 8.00 (100.00%) | 334.33 | 192 (1783.10) | 26.69 | 322 (84.87) | - |
| qwen3-1.7b | appworld | 16384 | 16 | fork_optimized_offload | 8.00 (100.00%) | 50.81 | 48 (1083.87) | 26.22 | 436 (61.57) | 84.89% |
| qwen3-1.7b | appworld | 16384 | 16 | fork_ordinary_offload | 8.00 (100.00%) | 336.17 | 190 (1811.79) | 26.69 | 322 (84.87) | -0.55% |
| qwen3-1.7b | swebench | 16384 | 16 | flash_ordinary_offload | 8.00 (100.00%) | 223.58 | 126 (1817.06) | 16.69 | 199 (85.86) | - |
| qwen3-1.7b | swebench | 16384 | 16 | fork_optimized_offload | 8.00 (100.00%) | 33.03 | 32 (1057.11) | 16.39 | 266 (63.11) | 85.53% |
| qwen3-1.7b | swebench | 16384 | 16 | fork_ordinary_offload | 8.00 (100.00%) | 228.30 | 127 (1840.77) | 16.69 | 199 (85.87) | -2.11% |

## Provenance

| Agentrix commit | Dirty | vLLM commit | Dirty | GPU blocks override | FlashInfer sampler | Prefix-aware policy | Admission window | Record cap | Full dataset | Runs |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 26f63d93787f | no | 287304ad68ce | no | 1700 | no | no | 0 | 32 | no | 16 |
| 26f63d93787f | no | 287304ad68ce | no | 1700 | no | yes | 16 | 32 | no | 4 |
| a6db65e6aa68 | no | 287304ad68ce | no | 1700 | no | no | 0 | - | yes | 16 |
| a6db65e6aa68 | no | 287304ad68ce | no | 1700 | no | yes | 16 | - | yes | 4 |
| c95508e11624 | no | 287304ad68ce | no | 1700 | no | no | 0 | 32 | no | 4 |
| c95508e11624 | no | 287304ad68ce | no | 1700 | no | no | 0 | 8 | no | 24 |
| c95508e11624 | no | 287304ad68ce | no | 1700 | no | yes | 16 | 32 | no | 1 |
| c95508e11624 | no | 287304ad68ce | no | 1700 | no | yes | 16 | 8 | no | 6 |

## Metric Notes

- Memory BW is NVIDIA `utilization.memory`, a memory-controller activity proxy rather than measured HBM GB/s.
- Peak GPU KV is sampled from `vllm:kv_cache_usage_perc` during the request phase.
- Logical KV read reduction estimates repeated prefix KV read volume avoided by ForkAttention; it is not physical cache capacity.
- Peak GPU KV reduction uses sampled physical GPU KV occupancy relative to the baseline named in the delta table.
- Throughput and request latency include both common-analysis and branch requests; branch-only distributions remain in the CSV.
- Record cap is the deterministic maximum number of source records per dataset; `-` with Full dataset=yes means every available record was used.
- Accuracy is deterministic output agreement against FlashAttention, not environment-level task accuracy.
- Repeat exact match compares Fork TP run 2 directly with Fork TP run 1.
- Experimental KV reload rebalance is disabled for this experiment matrix.
