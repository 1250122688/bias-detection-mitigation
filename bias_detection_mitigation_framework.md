# Model Bias Detection & Mitigation Framework

## 0. Scope & Governance
- **适用范围**：所有用于影响 member / group / provider 决策的预测模型（cost prediction, risk scoring, HICC 等）
- **触发时机**：上线前；重大改版（数据源、feature、目标变量变化）；定期复查（建议年度）；监控告警
- **角色**：Model owner 执行评估；独立 reviewer（analytics governance / compliance）复核；业务 owner 确认 impact 判断和 materiality
- **输出**：Bias Assessment Report + sign-off；未通过则不得上线或需明确 mitigation 计划

---

## 1. Bias Detection

### 1.1 Decision Context（理解业务与决策链）
- **Output**：类型（score / 金额 / 分类 / 排序）及取值范围
- **Unit**：预测对象（member / group / claim / provider）
- **Use**：output 如何转化为决策 —— 阈值？排序取 top N？人工参考？自动执行？（决定 1.3 用哪类 metric）
- **Prohibited factors**：法律、合规、公司政策上不得参与决策的因素（按适用法规列出，如 race, ethnicity, sex, age, disability, national origin, genetic information 等）；另列"可用但需谨慎"的因素
- **Impact**：outcome 对个人 / 群体的影响方向、大小、可逆性；谁受益、谁受损

### 1.2 Prohibited & Proxy Variable Check
- **直接检查**：input 中是否含 prohibited 变量 → 有则 **Critical finding**，停止流程并上报
- **Proxy 检查**：用全部 input 预测 protected attribute，或计算相关性；高相关变量（zip, language, plan type, provider 等）标记为 proxy risk，需业务解释其必要性
- **Label 检查**：label 是否已 encode 历史差异（如用 cost 代表 need，则 access 差异会进入 label）

### 1.3 Subgroup 定义与 Metric 选择
- **Subgroup 来源**：protected attributes（若无直接数据，说明是否用 imputation 如 BISG，并记录不确定性）+ 业务相关 subgroup（region, plan type, tenure, industry）+ 关键交叉组
- **最小样本量**：n 低于阈值的 group 不单独出结论，仅标注
- **Metric 与 Use 对应**：

| 决策方式 | 建议 metric |
|---|---|
| 阈值分类 | selection rate, FPR / FNR, precision by group |
| 排序 / top-N | group representation in top N, precision@k by group |
| 连续预测（金额） | residual / bias by group, over-/under-prediction rate, calibration slope |
| 通用 | calibration by group, AUC / error by group |

### 1.4 Business Impact Measurement
- 将 metric 差异换算为业务量：受影响人数、金额、被改变的决策数、下游后果
- Materiality 判断：小 metric 差异 × 大人群 = 大问题；反之亦然

### 1.5 判定：Threshold 与分级
**两道门**：统计门（差异是否真实）+ 业务门（差异是否重要），两道都过才升级。

**统计门（先过滞噪音）**
- 最小样本量：rate 类指标该 group 至少 30 个 positive event（或 n≥100），否则只标注不判定
- Bootstrap 置信区间不包含 parity 值（ratio=1 / 差异=0）
- Pattern：跨 ≥2 个时间窗或 cohort 方向一致；交叉组只要求方向一致，不单独设 act 线

**业务门：第一版起点 threshold（三档）**

| Metric | Pass | Watch | Investigate |
|---|---|---|---|
| Selection rate / top-N representation ratio | 0.8–1.25 | 0.8–0.9 或 1.1–1.25 | <0.8 或 >1.25 |
| FPR / FNR / TPR / precision 绝对差 | <5 pp | 5–10 pp | >10 pp |
| AUC 差异 | <0.02 | 0.02–0.05 | >0.05 |
| Calibration（E/O ratio 或 slope）by group | 0.9–1.1 | 0.85–0.9 或 1.1–1.15 | 超出 |
| 连续预测相对偏差（group 平均残差 / group 平均实际值） | <5% | 5–10% | >10% |

- **方向不对称**：按 1.1 Impact 判断哪一侧后果更重（如资源分配场景中的低估），该侧阈值收紧
- 监管规则（如适用的 4/5 rule）作为不可低于的 floor，不作为目标

**用自己的数据校准（替换起点值）**
1. 在最近 12 个月数据上跑一遍，得到各 metric 的实际水平和自然波动范围
2. 反推：业务 owner 定决策层面的容忍量（如"某 group 多错分的 member 不超过 X 人 / X%"），模拟 metric 差多少会产生该量级，以此替换起点值
3. Baseline：算现有流程（旧模型或人工规则）的 disparity，act 线不得差于现状，watch 线可定在现状水平
4. 每个 threshold 记录来源和假设；上线 3–6 个月后按 watch / investigate 触发频率复审

**分级**：Critical（prohibited 变量在 input，或 investigate 级差异且 impact 重大）/ Major（investigate 级）/ Minor（watch 级）/ No finding

### 1.6 Detection Report
决策链描述、subgroup 与 metric 定义、结果表、impact 估算、判定与分级、数据局限、下一步（进入 RCA 或结案）

---

## 2. Root Cause Analysis
每条 finding 对应到以下来源，输出 root cause map：

- **2.1 Data**
  - Representation：各 group 样本量、分布差异
  - Quality：missingness、错误率、数据完整性是否随 group 变化
  - Feature：proxy 变量、同一 feature 在不同 group 的测量含义是否一致
  - Label：定义是否公平；历史决策 / access 差异是否已进入 label
- **2.2 Model**：目标函数是否放大差异、对小群体欠拟合、feature selection、正则化、调参只看整体指标
- **2.3 Post-processing**：recalibration、threshold、business rule、override 是否引入或放大差异
- **2.4 Decision process**：模型之外的规则和人工环节

---

## 3. Mitigation
- **3.1 Data-level（pre-processing）**：补数据、reweighting、重采样、修复或重定义 label、移除 / 替换 proxy
- **3.2 Model-level（in-processing）**：fairness constraint / regularization、调整 loss weight、group-aware 训练、更换模型
- **3.3 Output-level（post-processing）**：group-wise calibration、threshold 调整、score 校正
- **3.4 Process-level**：调整决策规则、增加人工审核、限制 output 使用范围
- **选择原则**：优先修上游；量化 accuracy–fairness tradeoff；合规检查（部分场景不允许按 protected attribute 做 group-specific 处理，需 legal review）；记录未采用方案及原因

---

## 4. Post-Mitigation Validation
- **4.1 重跑 1.3–1.5**：目标 subgroup 差异是否降至阈值内
- **4.2 副作用检查**：其他 subgroup 是否恶化；整体性能损失是否在可接受范围；是否引入新 proxy
- **4.3 Business impact 复测**：换算后的影响是否真实减少（不只是 metric 变好）
- **4.4 稳定性**：out-of-sample / 跨时间验证
- **4.5 Sign-off 与文档**：更新 model documentation；设定监控指标、阈值和复查周期

---

## 5. Ongoing Monitoring
定期重跑 1.3 核心指标；数据 drift、人群结构变化或业务规则变更触发重新评估，回到 Section 1。
