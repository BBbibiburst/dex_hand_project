# 术语、符号和实际公式档案

版本：2026-11-19。只记录已核验实现，不以模型名称推断功能。代码路径默认相对项目根目录；正文尚未写入。

## 1. 数据与物理语义

|术语/符号|本文统一含义|单位/限制|依据|
|---|---|---|---|
|Real / \(X_r\)|实机既有processed触觉观测|无已认证N/Pa标定；不能称压力真值|P40 sensor_semantics_report；P41源码缺口|
|Natural Sim / \(X_s\)|既有任务策略真实推进MuJoCo后记录的processed触觉|区别于Real引导探针数据及重复读取索引|P43 integrated dataset_manifest/windows_manifest|
|Paired Sim / \(X_p\)|Real统计目标指导球形探针、经MuJoCo执行生成的统计伪配对触觉|不是同步实机—仿真接触，不是直接写Gaussian数组|P43-C3报告、matching.py、配对清单|
|native touch raw|当前仿真原生touch传感器法向力累加|项目SI约定下为N，不除面积；感知区域重叠会重复计力|P40语义报告及712行映射|
|Real raw|真实采集侧存储的原始观测|不等于MuJoCo raw；原驱动/标定源码缺口仍在|P21.1、P41|
|processed|各自采集/后处理链输出|两域处理合同和尺度不完全等同；归一化后无量纲|signal_processing.py；P40|
|712D|每帧712个标量通道|不是712个三维力向量|P40通道导出；P44模型输入|
|Patch|固定索引通道组，共16组|仿真内部几何可查；Real对应物理方向未认证|P40、P41；P49 evaluate.py OFF|
|Session|真实采集会话/分组单位|不是窗口，多个Trial属于一个Session|Real index及P37数据表|
|Trial|Session内完整采集试次|不自动视为跨Session独立重复|P37/P43 real索引|
|Root|仿真采集初态/来源的分组键|需看具体指纹定义；不同seed不自动独立|P35.2/P43独立性说明|
|Episode|一次执行记录|不同窗口可来自同一Episode；Root与Episode不总是一一对应|各阶段manifest|
|Window / \(X\)|四帧有序触觉，单样本\(X\in\mathbb R_{\ge0}^{4\times712}\)|批量\([N,4,712]\)，不等于N个独立采集主体|P37及P43窗口清单|
|Stage|CLOSE/HOLD/LIFT等源记录阶段|同名不保证两域物理状态、任务或载荷相同|P34/P37|

Patch长度为`[32,32,56]*5+[112]`，总计712。仿真site布局有4×8、4×8、7×8及掌心7×16，不能据此认证Real电气映射或局部朝向。

Natural按20Hz源记录提取的相邻帧通常间隔0.05s，四帧首末跨度0.15s；Real保留原样本索引且间隔不均匀。\(t\)用于四帧时写“帧索引”，用于Euler场时另记\(\tau\in[0,1]\)，不能将归一化积分时间称为物理秒。

## 2. 数据版本（不得混用）

|版本|Real Train / Validation|Sim Train / Validation|用途|
|---|---|---|---|
|P24全保存池|全库11110窗，77有效Session，1645完整资产Trial；非该实验训练量|6359窗，119注册初态组|历史全库，不等于所有模型都用了全库|
|P33/P35小开发集|每域训练128、验证48窗；Real16/6 Trial|P35实际贡献35/11 Root，允许清单38/13|同校准Session开发；旧转换器曾用验证Trial|
|P37旧开发|5153/1408窗，32/7 Session|1824/406窗，36/8 Root|旧Lift-derived来源；非服务器全五任务|
|P43及P44–P51|5153/780窗，32/4 Session；Train755、Val120Trial|4049/905窗，50/10 Root、五任务|本轮主线数据；另有历史测试，当前不打开|
|P49/P50 D内部|Real24fit/8inner Session；4217/936窗|40fit/10inner Root；3249/800窗|仅D分组泛化；不是F0新盲测|

P43将旧P37七个Validation Session分为4模型选择及3历史测试角色；原P37文件未改。名称改变不消除先前开发暴露。

## 3. 独立Dual-VAE与Translator

核验源码：`outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18/code/vae.py`；上游为P43-C3 models.py。

- \(E_r,E_s\)：各域独立Encoder，同构但不共享权重。每帧拼接硬Mask和归一化log响应，1424→128、GELU/LayerNorm、GRU128，分别输出64维mean/logvar；\(\mu,\log\sigma^2\in\mathbb R^{4\times64}\)。logvar夹在[-8,4]是既有模型合同，不新增搜索。
- \(D_r,D_s\)：各域独立Decoder。32维前帧Mask编码、当前64维latent→GRUCell128；Mask head输出712 logits；幅值分支以隐状态及当前Mask编码为条件，每Taxel输出两分量Gamma混合的logit/shape/rate。
- \(T\)：Translator，2层GRU，输入64、隐藏128、输出投影64；逐帧映射\(\mathbb R^{4\times64}\to\mathbb R^{4\times64}\)。两独立空间不能在未经验证时直接解释为公共物理坐标。

训练重参数化：\(z=\mu+\exp(\tfrac12\log\sigma^2)\odot\epsilon\)，\(\epsilon\sim N(0,I)\)。推理使用posterior mean。VAE实际损失为每窗Mask BCE + 正值Gamma混合NLL + 既定权重KL；权重及训练尺度须读取具体训练config，不能把后期冻结模型都说成联合训练。

确定性读出：\(m_{tj}=\mathbf1[a_{tj}\ge0]\)，\(y_{tj}=m_{tj}s_+\sum_{q=1}^2\pi_{tjq}k_{tjq}/r_{tjq}\)。shape/rate通过softplus加机器epsilon为正；MAP0.5与混合均值不同于自由随机采样。Teacher forcing用真实上一帧Mask，部署用生成Mask，两种成绩不可混报。

原始Translator配对目标为latent Huber加冻结Real Decoder NLL，监督源只为统计伪配对；Natural参与Sim VAE训练不等于Natural具有一一Real目标。部署不需要Real样本、GMM或探针。

## 4. GMM与监督语义

\(p(u\mid c)=\sum_{k=1}^{K}\pi_k\mathcal N(u;\mu_k,\Sigma_k)\)仅概括已拟合空间统计混合模型；\(u\)为具体实现的Patch内固定索引坐标代理，\(c\)仅指对应实现中已核实的Patch分组等拟合条件，不代表已经建立跨域语义条件或实现Conditional OT。K=1/2/3、BIC与Session稳定性来源见原GMM日志，不泛称所有实验都做过选择。

P41前置审计本身没有拟合GMM；后续`outputs/tactile_gmm_dual_vae_2026-11-16/`有独立授权的代理索引GMM，P43-C3继承。原任务阶段名称相近不代表同一已完成模型。

GMM统计→字典检索/加载条件→MuJoCo执行→统计伪配对。它是训练监督工具，不是最终Natural推理必须执行的前置模块。匹配阈值/空间代理不能当硬件标定。

## 5. 共享参数向量场与对抗损失

源码P44 model.py与P49 train.py。\(z_s=E_s(X_s)\)的posterior mean；\(h_0=0\)，**不是\(h_0=z_s\)**。

\[
 v_\theta(h,z_s,\tau)=T(z_s)+c_\theta(h,z_s,\tau),\quad
 h_{k+1}=h_k+K^{-1}v_\theta(h_k,z_s,k/K),\quad Y=D_r(h_K).
\]

条件时间特征为\([\tau,1-\tau,\sin\pi\tau,\cos\pi\tau]\)，与\(h,z_s\)拼接132维，投影128、GRU128、线性64。K∈{1,2,4,8}为同一[0,1]积分区间的Euler步数；更多K不是更多对抗轮数，不等于训练了K个独立网络。“连续场参数化/离散积分”不能直接写成已证明连续物理迁移。

\(D_\phi\)表示训练用域判别器，\(\ell_\phi\)为logit，\(p_\phi=\sigma(\ell_\phi)\)为Real概率。P44/P49全局D特征是Mask2848与log响应2848拼接，5696→192→64→1。P45 A仅用连续2848维；三组参数量不完全相同，不能称容量完全控制。

P49 K1生成器目标：
\[
 L_G=\mathbb E_w\operatorname{softplus}(-\ell_\phi(Y))
 +0.5295653698327406L_p+0.001L_v.
\]

定义Patch份额\(a_{tp}(X)=\sum_{j\in I_p}X_{tj}/\max(\sum_jX_{tj},10^{-12})\)，相对帧差\(r_t(X)=\|X_{t+1}-X_t\|_1/\max(\|X_{t+1}\|_1+\|X_t\|_1,10^{-12})\)。

\(L_p=\mathbb E_w[\frac14\sum_t\|a_t(Y)-a_t(X_s)\|_2^2+\frac13\sum_t|r_t(Y)-r_t(X_s)|]\)。\(L_v\)为修正场按Real Train latent std（下限0.1）归一化后的均方；P49 K1实际使用c[0]，不含Translator基项。w为域内组等权的窗口权重。

判别损失按Real/Sim各0.5加权BCE，加lazy R1（gamma0.1、每16步一次）。冻结D参数不等于切断对输入的梯度。Decoder原有hard forward/ST backward与D输入处Mask的零VJP是两处不同机制，必须分别解释。

P50三支连续log响应视角：Global2848维；Spatial为每帧16Patch均值/二阶矩/最大值，共192维；Temporal为相邻三帧差，共2136维。每支128→64→1。Ladv为三支BCE平均。Refiner另有120968参数，实际为log响应残差非负参数化，不是无条件生成。其flow不包含新增残差；preserve对整体幅值缩放不敏感，属于已验证约束缺口。

## 6. 三种评价协议必须分开

|协议|分类器训练对象|冻结什么|可支持结论|
|---|---|---|---|
|迁移前原11C0|Real Train与原Natural Train|模型、log1p、合并Train Scaler、通道和0.5阈值|原边界读出改变；联合门槛另查|
|迁移后独立Probe|Real与对应Generated Train|生成器后重训的新Probe|是否仍有可学习域信息；不能替换旧主协议|
|迭代Dk下FR|Real与上一轮生成Train|本轮G更新期间Dk固定|该边界下前后接受率；不是对所有域差异的证明|

C0标签已核验C3 code/c0.py为Real=1、Sim=0；早期初始规划文本曾有相反表述，不能照抄。P49/P50同样用Real=1。不同历史评估仍应逐文件确认。

\(BA=(TPR+TNR)/2\)，\(D_{BA}=1-2|BA-0.5|\)。BA∈[0.45,0.55]与D≥0.90等价；历史联合要求还含AUC∈[0.475,0.525]。AUC为Real正类概率的排序能力，不根据小于0.5就翻转正式值。

\(FR_g=|g|^{-1}\sum_{i\in g}\mathbf1[p_i\ge0.5]\)，\(FR_{macro}=G^{-1}\sum_gFR_g\)；Micro为全部窗口均值。Real Acceptance按Session宏平均。多视角Joint逐窗三支AND，相应连续分数min(logits)。宏均值≥90%不是每Root都≥90%，也不是Root通过比例。

Real score envelope：8个历史inner Session组等权logit2.5%/97.5%区间，4外层Session评价；是P51辅助描述，不改变分类阈值，不等于物理真实分布置信域。

## 7. 分布、结构与依赖指标

- 固定距离表示\(\varphi(X)=[\log(1+X)-\mu_{r,train}]/\max(\sigma_{r,train},0.05)\)，展平2848维。它与原C0合并Train拟合的Scaler不同，不能混用。
- P48–P51项目名SWD实际为1024固定投影下一维**Wasserstein-1绝对CDF差积分**的均值，Real按Session、Sim按Root等权。不是未经核实的SW2平方距离，也不是原物理单位。P46某些窗口边际距离与此估计目标不同。
- MMD采用RealTrain固定sigma的0.5/1/2倍RBF核平均，带组权重V统计，最后取非负平方根。写结果要注明这一实现，不套用无偏U统计区间。
- PRDC为特征邻域Precision/Recall/Density/Coverage，k3/5/10/20、P47固定100次采样。P/R/C∈[0,1]，Density可大于1；100次采样不等于100次独立采集。不能识别保持输出集合的PairShuffle。
- Root内归一化HSIC采用中心化RBF Gram核对齐：\(\operatorname{tr}(K_cL_c)/\sqrt{\operatorname{tr}(K_c^2)\operatorname{tr}(L_c^2)}\)，先每Root再宏平均；零变化按实现标记。高值仅支持统计依赖，不证明接触语义正确。
- Mask为精确x>0；幅值分位数只用正值。P39 amplitude-only有效特征为**每帧全体正值5分位数，共4×5**，不是每Patch分位数；恒零补齐只是复用网络接口，不能当2848有效特征。
- P(0→1)、P(1→0)、连续激活长度、lag1和相邻帧变化描述索引时序；Root/Session Bootstrap单位、次数300或1000随阶段而异，不统一篡改。
- Constant、RealDonor、PairShuffle分别检验常量退化、无输入依赖的真实样本和对应损坏；不能要求边际分数区间单独拒绝所有负对照。
- \(\log2-CE\)为历史冻结分类器的JSD代理，分类器不最优时不等于真实JS散度，可能为负。

## 8. 待核实项

真实Taxel物理映射、Raw处理与标定源码、跨域相同物体/载荷/控制条件、任务真值保持、跨Session泛化充分性仍有限。本文不创造名为“90%迁移等价性”的新综合分数。

## V2协作同步（2026-11-19）

当前研究范围与七章候选目录见THESIS_PLAN_V2.md，机器人平台、抓取、示教及数据证据见ROBOTICS_EVIDENCE.md；THESIS_PLAN.md为历史六章V1。本文既有技术证据及范本阅读记录保持原范围，不据目录变更回填实验。后续遵守ChatGPT任务书→Codex材料核验并获通过→仅写指定tex→用户审核→用户确认后编译并冻结，Codex不编写或修改任务书、不自动下一节。当前目录尚待审核，无正文冻结。
