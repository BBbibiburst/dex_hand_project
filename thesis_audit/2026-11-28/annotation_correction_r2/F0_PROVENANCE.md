# F0来源专项纠正

审核意见中“F0就是未训练的C3初始化”的判断不符合现有调用链。初始修正项零初始化与最终比较用F0是不同状态；本轮不为纠正这一点执行任何模型加载或推理。

1. `outputs/tactile_p45_mask_ablation_2026-11-18/A/training_summary.json`登记A组训练20轮、G_steps=1152、D_steps=1280，VAE_bit_identical=true。
2. `outputs/tactile_p48_unified_four_route_audit_2026-11-19/code/generate.py:9`对A组加载`P45/A/generator.pt`，不是generator_initial.pt。K=1是该训练终点的单步积分读出。
3. `outputs/tactile_p49_round_frozen_adversarial_2026-11-19/code/train.py:9`同样加载`P45/A/generator.pt`；第14行断言F0验证输出等于P45 A组保存的K1_validation，并以P45Aendpoint标记来源。
4. P49中F0的optimizer_steps=0表示尚未执行P49新增更新，不能消除P45已经发生的训练。
5. P48 `metrics_swd.csv`的A_K1行登记SWD=0.20660925217182965、Original_SWD=0.41134301198750495、R_W=0.4977202817338595；没有在本轮重算该指标。

因此摘要改用“训练后的共享向量场单步基线”。49.77%是原合同下整条方案相对未转换Natural Sim的距离下降，不能从中单独分离C3初始化、向量场修正或训练损失的因果贡献。Original已经提供未转换输入参照，但完整外部基线、同预算MLP/GRU及干净的单因素消融仍然缺失。

本项为源代码及历史记录静态核对（L2），不是新实验。GMM坐标原点不一致问题依然成立，不因F0来源澄清而消失。
