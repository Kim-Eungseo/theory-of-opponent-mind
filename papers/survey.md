# 협력 MARL 의 Theory of Mind & Opponent Modeling — 서베이

본 저장소의 연구 (cooperative OM aux + Overcooked / Hanabi) 와 직접 관련된
**파트너 / 상대 모델링, belief 추론, zero-shot coordination** 분야의 핵심
논문 21편 정리. 각 논문 PDF 는 `papers/pdf/` 안 (gitignored — 용량 큼).

---

## 0. 본 서베이의 위치

본 저장소의 목표:
> "내 정책 표현으로 상대를 모델링하는 게 cooperative MARL 에 도움이 되는가?"

이 질문은 **20여 년의 ToM-MARL 연구 흐름** 과 직결됩니다. 단계적 전개:

```
1990s: Game-theoretic OM (Bayesian opponent identification)
2010s: Deep MARL + 단순 partner aux head 접근
2018:  ToMnet (meta-learning ToM)
2019:  BAD (Bayesian belief MDP, Hanabi 첫 강한 결과)
2020:  Other-Play (self-play 의 한계 정의 → ZSC 등장)
2021:  FCP / TrajeDi / MEP (population diversity 가 ZSC 의 정답)
2022:  Equivariant / SED (구조적 / convention 명료화)
2023:  COLE / OBL 정착 (open-ended + grounded 해석)
2024:  ZSC-Eval (벤치 표준화) / Noisy ZSC (현실적 가정)
2025:  Cross-env / R3D2 / AHCC (일반화 frontier)
```

본 연구의 negative finding ("vanilla aux + BAD-routing = capacity 효과") 은
이 흐름의 *2018-2019 단계 가설* 을 다시 검증한 것 — 이미 학계에서 *그 정도
manipulation 만으로는 부족함* 이 알려져 있고, 그래서 분야가 **population
diversity (FCP/TrajeDi/MEP)** 와 **explicit belief (BAD/SAD/OBL)** 로 진화한
것임. 우리는 그 진화의 동기를 재검증한 셈.

---

## 1. 핵심 용어 사전

| 용어 | 한 줄 정의 |
|------|---------|
| **ToM** (Theory of Mind) | 다른 agent 의 내적 상태 (belief, goal, intent) 를 추론하는 능력 |
| **OM** (Opponent / Other-agent Modeling) | ToM 의 RL 구현 — 상대 policy/state 예측 |
| **ZSC** (Zero-Shot Coordination) | 학습 시 본 적 없는 partner 와 즉시 협력 |
| **Self-Play** (SP) | 같은 정책을 자기 자신과 학습 (대부분 baseline) |
| **Cross-Play** (XP) | 독립 학습된 두 정책 간 평가 |
| **Other-Play** (OP) | 환경 symmetry 무작위화로 convention lock-in 방지 |
| **BAD** (Bayesian Action Decoder) | public belief 위에서 deterministic policy |
| **SAD** (Simplified Action Decoder) | greedy action 을 partner input 으로 — BAD 의 단순화 |
| **OBL** (Off-Belief Learning) | counterfactual belief 로 fixed-point 학습 |
| **FCP** (Fictitious Co-Play) | seed × checkpoint 풀에 best-response 학습 |
| **TrajeDi** | trajectory-level JS divergence 로 다양성 강제 |
| **MEP** (Maximum Entropy Population) | population entropy bonus 로 diverse pool |
| **COLE** | open-ended graph game 으로 population 자동 생성 |
| **AHCC** (Ad-Hoc Human-AI Coordination Challenge) | 새 표준 평가 벤치 |
| **ToMnet** | meta-learn 으로 unseen agent 의 mental state 예측 |
| **Convention** | 임의로 합의된 행동 의미 (예: "왼쪽 hint = 빨강 카드 play") |

---

## 2. 카테고리

```
[1] Foundational ToM / Opponent-Modeling 아키텍처    (3편)
[2] Hanabi-specific belief & convention 연구         (6편)
[3] Overcooked / Cooperative coordination 메서드      (7편)
[4] Zero-Shot Coordination 평가 프레임워크            (2편)
[5] 최신 (2024–2025) 벤치마크, 일반화, 서베이         (3편)
```

각 entry 표기: **(year) Authors** — 제목, *venue*, 핵심 주장, 본 연구와의 관계.

---

## [1] Foundational architectures

### 🌱 Rabinowitz et al. (2018) — *Machine Theory of Mind* (ToMnet)
*ICML 2018* — `papers/pdf/rabinowitz_2018_tomnet.pdf` (arxiv 1802.07740)

**한 줄**: 다른 agent 의 *과거 행동만* 을 보고 그들의 mental state (목표,
믿음, false belief) 를 예측하는 meta-learning 네트워크.

**메서드**: 세 모듈 파이프라인
- *Character net* `e_char(τ)`: 과거 episode 데이터로 agent 의 character 임베딩
- *Mental net* `e_mental(o, e_char)`: 현 obs 와 character 합쳐 mental state
- *Prediction net* `f_pred(state, e_mental)`: action / goal / belief 예측

False-belief task (Sally-Anne 같은 cognitive task) 에서 인간 ToM 패턴 재현.

**환경**: gridworld 기반 toy POMDP. 정량 score 보다 *질적 ToM 능력* 입증이
목적.

**본 연구와의 관계**: probing aux head 의 **이론적 출발점**. 우리는 ToMnet
의 simplified 버전 (single-step partner action prediction) 을 BAD-routing
형태로 시도. ToMnet 자체는 *meta-learning* 인데 우리는 *online RL aux*
로 단순화 — 그래서 효과가 약했을 수도.

**한계**: meta-learning 에 strong prior (수백 episode obs) 필요. 실시간 ZSC
에는 부적합.

---

### 🌱 Foerster et al. (2019) — *Bayesian Action Decoder* (BAD)
*ICML 2019* — `papers/pdf/foerster_2019_bad.pdf` (arxiv 1811.01458)

**한 줄**: **Public belief MDP** 정의 + 그 위에서 deterministic policy 학습.
2-player Hanabi 첫 강한 결과 (24 score).

**메서드**:
- 매 시점 *public belief* `B_t = P(s_t^private | h_public_t)` 정의
- Policy 는 deterministic function `π: B_t → a_t` (확률 정책 X)
- 모든 agent 가 같은 belief tracker 공유 → 결정론적 행동이 *그 자체로*
  belief 정보를 전달함

```
공식적: B_{t+1}(s) ∝ B_t(s') · 1[π(B_t(s')) = a_t observed] · P(s|s', a_t)
```

**환경**: 2-player Hanabi-Full. **24.0 / 25** average (당시 SOTA).

**본 연구와의 관계**: "explicit belief in policy input" 의 **이론적 원형**.
우리 *BAD-routing* 변형 (TOM 출력을 policy input 에 concat) 은 BAD 의 단순
구현체. 하지만 BAD 는 **확률 정책 X, deterministic on belief space** —
우리는 그 핵심을 빼먹어서 효과가 약했을 가능성.

**한계**: belief tracker 가 partial observability 에서 폭발적으로 커짐.
계산량 issue 로 후속 SAD 로 단순화됨.

---

### 🌱 Hu et al. (2020) — *"Other-Play" for Zero-Shot Coordination*
*ICML 2020* — `papers/pdf/hu_2020_other_play.pdf` (arxiv 2003.02979)

**한 줄**: Self-play 가 *임의의 symmetry-breaking convention* 으로 수렴
한다는 문제 정의 + 환경 symmetry 무작위화로 해결.

**메서드**:
- 환경의 **automorphism group** `G` 정의 (예: 카드 색 permutation)
- 매 episode 마다 partner 시점에 random `g ∈ G` 적용
- ego policy 는 *symmetry-invariant* solution 만 학습 가능

```
ZSC objective:  max_π  E_{g ∼ G}[ J(π, g·π) ]
```

**환경**:
- Toy lever coordination: random partner 와 4.0 / 10.0 (SP), OP 8.5 / 10
- Hanabi 2-player: SP 22.7 (cross-play 하면 9.5), OP 22.0 (cross-play 11.7)

**본 연구와의 관계**: **"self-play 점수 ≠ 협력 능력"** 의 학문적 정의.
본 저장소의 capacity-confound finding 도 이 framing 에 정확히 들어맞음 —
self-play 점수만 보고 OM 효과 주장하면 ZSC 학자들이 "그래서 cross-play
는?" 하고 즉시 반박할 것.

**한계**: 환경 symmetry 가 명시 가능해야 함. Overcooked 처럼 명확한 symmetry
없는 환경에선 적용 어려움.

---

## [2] Hanabi belief, conventions, decoding

### 🃏 Bard et al. (2020) — *The Hanabi Challenge*
*Artificial Intelligence Journal, 2020* — `papers/pdf/bard_2019_hanabi_challenge.pdf`
(arxiv 1902.00506)

**한 줄**: Hanabi 를 ToM 의 **canonical benchmark** 로 정착시킨 proposal
paper. HLE (Hanabi Learning Environment) 공개.

**기여**:
- 학습 환경 (HLE — 본 저장소에서 사용 중) + 통합 API
- 규칙 기반 bot 풀 (MC, SimpleAgent, SmartAgent, …) — held-out partner
  evaluation 의 표준
- Self-play vs ad-hoc team-play 의 명확한 구분
- 점수 기준: 24+ = 강함, 25 = 완벽 (이론상)

**환경**: 2-5 player Hanabi-Full / Small / VeryAbbreviated 등.

**본 연구와의 관계**: 우리 Hanabi 실험의 환경 정의. "vanilla self-play 가
~5 plateau" 라는 사실은 이 paper 가 이미 보고함 (당시 deep RL baseline 도
~3-5).

---

### 🃏 Hu & Foerster (2020) — *Simplified Action Decoder* (SAD)
*ICLR 2020* — `papers/pdf/hu_2020_sad.pdf` (arxiv 1912.02288)

**한 줄**: BAD 의 단순화 — agent 가 sampled action 외에 **greedy action**
도 함께 emit, partner 가 greedy 를 input 으로 받음.

**메서드**:
- 매 step 두 action: `a_t ∼ π(·|h_t)` (실제 실행), `a_t^greedy = argmax π`
- Partner observation 에 `a^greedy_t-1` concat
- Greedy 는 *deterministic intent signal* — partial belief 정보 전달

```
공식적: π_partner(o_t, a^greedy_{t-1})  # 추가 입력으로 greedy 수신
```

**환경**: 2-player Hanabi-Full. **22.06 / 25** average self-play.

**본 연구와의 관계**: 우리 **TOM+BAD** 의 가장 가까운 prior work. 차이:
- SAD: partner 의 *greedy intent* 를 input 으로 (정확히 원하는 행동)
- 우리 TOM: partner 의 *예측된* action 분포 (학습된 추정)

SAD 가 더 강한 이유는 *ground truth* 를 partner 에게 직접 전달하기 때문.
우리 TOM 은 추정만 가능 — 정확도 33% 정도. 이 차이가 효과 차이를 설명.

**한계**: action space 좁아야 (Hanabi 20 actions). Continuous / 큰 action
space 에선 greedy concat 이 noisy.

---

### 🃏 Hu et al. (2021) — *Off-Belief Learning* (OBL)
*ICML 2021* — `papers/pdf/hu_2021_obl.pdf` (arxiv 2103.04000)

**한 줄**: 현 Hanabi self-play SOTA (~24.1). **counterfactual belief** 로
fixed-point 학습 → grounded convention 만 emerge.

**메서드**:
- "level-0" partner: random 정책
- ego 는 *level-0 partner 가 했을 행동* 을 가정한 belief 로 학습
- 그 결과 ego policy 가 새 partner 가 됨 → "level-1"
- 반복하면서 fixed point 에 수렴

```
π_{k+1} = argmax_π  E_{partner ∼ π_k}[J(π, partner) | belief grounded on π_k]
```

핵심 통찰: 보통 self-play 는 *임의 convention* 에 lock-in 됨. OBL 의 fixed
point 는 **arbitrary convention 이 쓸모없는** policy — hint 가 *literal*
의미만 갖는 정책으로 수렴.

**환경**:
- 2-player Hanabi: **24.10** (당시 self-play SOTA)
- Cross-play with OBL-trained partners: 23+ (vs SP 의 12)

**본 연구와의 관계**: **"왜 우리 aux head 가 안 통했나"** 의 답. OBL 은 *명시적
counterfactual reasoning* 을 알고리즘 레벨에 박아넣어야 grounded convention
이 나옴을 보임. 단순 aux loss 로는 임의 convention 에 lock-in 되는 SP 와
같은 운명.

**한계**: fixed-point 수렴까지 1.5B+ env steps. 컴퓨트 무거움.

---

### 🃏 Hu, Sokota et al. (2022) — *Self-Explaining Deviations* (SED)
*NeurIPS 2022* — `papers/pdf/sokota_2022_self_explaining_deviations.pdf`
(arxiv 2207.12322)

**한 줄**: OBL 의 fixed-point 위에 "행동의 *의미* 가 명확한" 행동만 채택.
Hanabi 의 *finesse play* (간접 hint via play) 처음 학습.

**메서드**: agent 가 deviation 을 할 때, deviation 의 의미가 partner 에게
*명료하게 추론 가능* 한 경우만 reward 받게 설계. self-explaining 의
formal 정의 통해 OBL+SED loss 추가.

**환경**: 2-player Hanabi. OBL 보다 약간 낮지만 더 *해석 가능* 한 정책.

**본 연구와의 관계**: convention quality 와 raw score 가 *별개 축* 임을
보임. 우리 자료 분석 시 "OM 정확도 ↑ 인데 score ↑ 아님" 이 SED 의 framing
과 정확히 일치 — convention 의 *grounding* 자체가 문제였을 수.

---

### 🃏 Stechbarth et al. (2025) — *Augmenting Action Space with Conventions*
*JAAMAS (Springer), 2025* — `papers/pdf/conventions_2024_hanabi.pdf`
(arxiv 2412.06333)

**한 줄**: Convention 을 *암묵* 이 아닌 *명시적 macro-action* 으로 격상.
2-3 player Hanabi 에서 convention 학습 가속.

**메서드**: 게임의 macro-action 사전 (예: "color hint → next-turn play
slot 1") 을 action space 에 추가. agent 는 atomic + macro action 모두 선택
가능. Reward shaping 으로 macro 의 *use it or lose it* 강제.

**환경**: 2-3 player Hanabi-Full. SAD 대비 sample efficiency ~3배.

**본 연구와의 관계**: convention 을 *학습* 하지 말고 *주입* 하는 방향.
우리가 OM aux 로 *implicit* 시도한 것의 정반대. 향후 우리 연구가 macro 를
explicit 도입하는 형태로 확장될 수 있음.

---

### 🃏 Nekoei et al. (2025) — *A Generalist Hanabi Agent* (R3D2)
*ICLR 2025* — `papers/pdf/nekoei_2025_generalist_hanabi.pdf` (arxiv 2503.14555)

**한 줄**: **2-, 3-, 4-, 5-player 동시 핸들** 하는 단일 agent. 텍스트
기반 표현 + dynamic action-space transformer.

**메서드**:
- 게임 상태를 자연어 sequence 로 인코딩
- Transformer policy 가 action 을 *이름* 으로 출력 (action ID 가 아님)
- 그래서 player 수에 따라 변하는 action space 를 한 모델로 처리

**환경**: Hanabi 2/3/4/5 player. 모든 player count 에서 sub-optimal 이지만
하나의 agent — *cross-config zero-shot transfer* 가 가능한 첫 사례.

**본 연구와의 관계**: ToM 의 *configuration generalization*. 우리 multi-env
일반화 (Overcooked + Hanabi + ViZDoom) 의 단일-environment-multi-config
버전.

---

## [3] Overcooked / Cooperative coordination

### 🍳 Carroll et al. (2019) — *Utility of Learning about Humans*
*NeurIPS 2019* — `papers/pdf/carroll_2019_overcooked.pdf` (arxiv 1910.05789)

**한 줄**: **Overcooked-AI 환경 출시** + PPO+BC (behavior cloning) 가
self-play 보다 인간과 더 잘 맞음을 입증.

**메서드**:
- 5 layout (cramped / asymmetric / forced / coord-ring / counter-circuit)
- 인간 데이터로 BC partner 학습 → PPO ego 가 BC partner 와 학습
- 비교: SP / BC / PPO+BC

**환경 점수** (asymmetric_advantages, episode 400 step):
| 방법 | Self-play | Human |
|------|---------|-------|
| SP (vanilla PPO) | ~140 | ~50 |
| BC | ~80 | ~80 |
| **PPO+BC** | **~140** | **~120** |

**본 연구와의 관계**: 우리 환경 그 자체. *우리가 재현 시도한* 결과 — 그들의
default config 로 vanilla SP 도 sparse=0 으로 collapse 했음. 따라서 그들
publication 의 ~140 은 PPO+BC 였음을 우리가 직접 확인. 우리 vanilla 82
점이 사실 그들 reproducible 결과보다 강함.

**한계**: human data 의존도 높음. ZSC 와는 별개 setting.

---

### 🍳 Strouse et al. (2021) — *Fictitious Co-Play* (FCP)
*NeurIPS 2021* — `papers/pdf/strouse_2021_fcp.pdf` (arxiv 2110.08176)

**한 줄**: Population-based ZSC 의 정수 — N seed 의 self-play + 각 seed 의
training-time 체크포인트 = pool, 그 pool 에 best-response 학습.

**메서드**:
1. N=32 self-play agents 학습 (다른 seed)
2. 각 agent 의 학습 중 K=20 체크포인트 저장 → pool size N×K = 640
3. ego agent 를 pool 무작위 sampling 한 partner 와 학습 (best-response)

```
ego = argmax_π  E_{partner ∼ Pool}[ J(π, partner) ]
```

**환경**:
- Overcooked: SP 대비 cross-play +30~+80 sparse (layout 따라)
- 인간 평가 (114명): FCP 가 BCP, SP, BC 모두 통계 유의 우세 (*p < 0.05*)

**본 연구와의 관계**: 우리가 self-play 에서 머무는 동안, ZSC 분야는 이미
2021 에 이 답을 냈음. *aux head 같은 representation trick 은 cross-play
에 약하고, population diversity 가 강함*. 본 연구가 publishable 되려면 FCP
같은 방법 위에서 추가 향상을 보여야.

**한계**: 학습 비용 N×K 배. 환경 의존도 높음 (대부분 보고는 Overcooked).

---

### 🍳 Lupu et al. (2021) — *TrajeDi*
*ICML 2021* — `papers/pdf/lupu_2021_trajedi.pdf` (PMLR v139)

**한 줄**: Trajectory-level Jensen-Shannon divergence 로 population
*행동* 다양성 명시 강제. Action-distribution diversity 보다 강함.

**메서드**:
- Population members `{π_1, ..., π_N}` 의 trajectory 분포 `p_i(τ) = ∏ π_i(a_t|s_t)`
- 목적: `J(π_i) - λ * mean_{j≠i} JSD(p_i || p_j)`
- 미분 가능 → SGD 로 학습

**환경**: Overcooked / Hanabi. FCP 와 비슷하거나 약간 우세 (특히 Hanabi).

**본 연구와의 관계**: "diversity = 핵심" 의 또 다른 증거. 우리는 6 시드
*세는* 수준이지만 TrajeDi 는 *수렴 시 행동 다양* 을 강제 — 그 차이가
ZSC robustness 의 차이.

---

### 🍳 Zhao et al. (2021) — *Maximum Entropy Population* (MEP)
*NeurIPS 2021 spotlight* — `papers/pdf/zhao_2021_mep.pdf` (arxiv 2112.11701)

**한 줄**: Population 학습 시 *집단 entropy bonus* + prioritized sampling
으로 challenging partner 우선 학습.

**메서드**:
- Pool entropy: `H(π_pool) = -∑_i (1/N) log[(1/N) ∑_j π_j(a|s)]`
- Stage 1: 모든 N 정책을 H 최대화 + reward 로 동시 학습
- Stage 2: ego 가 pool sampling → "어려운" partner 에 prioritize

**환경**: Overcooked. Carroll layout 평균 sparse:
| 방법 | cramped | asym | forced | coord | counter |
|------|--------|-----|------|------|--------|
| SP | 80 | 130 | 0 | 30 | 5 |
| FCP | 110 | 175 | 30 | 110 | 80 |
| **MEP** | **130** | **180** | **35** | **140** | **120** |

**본 연구와의 관계**: counter / forced 같은 *우리가 5M 으로 못 푼* 어려운
layout 도 MEP 면 잘 됨. 기본 RL 부족이 아니라 *population diversity* 가
관건이었음을 시사.

---

### 🍳 Li et al. (2023) — *COLE* (Tackling Cooperative Incompatibility)
*ICML 2023 → JAIR 2024 (extended)* — `papers/pdf/li_2023_cole.pdf`
(arxiv 2306.03034)

**한 줄**: ZSC 를 **open-ended graph game** 으로 — 매 generation 에서 ego
약점을 exploit 하는 partner 가 emerge, ego 가 그에 대응.

**메서드**:
- Generation t: pool `{π_1^t, ..., π_K^t}` 에서 ego 선정
- Adversary: ego 와 협력 *못하는* partner 학습 (intentional
  miscoordination)
- Pool 갱신: 새 adversary 합류
- 반복 → robust ego

**환경**: Overcooked + 인간 실험 (자체 platform). FCP/MEP 와 비슷하거나
약간 우세 + *humans 와 더 잘 맞음*.

**본 연구와의 관계**: ZSC 의 frontier — 단순 diversity (FCP/MEP) 를 넘어
*adaptive curriculum*. 본 연구의 next step 후보.

---

### 🍳 Muglich et al. (2022) — *Equivariant Networks for ZSC*
*NeurIPS 2022* — `papers/pdf/muglich_2022_equivariant_zsc.pdf` (arxiv 2210.12124)

**한 줄**: ZSC 를 *알고리즘* 이 아닌 *아키텍처* 로 — 환경 symmetry 에
equivariant 한 네트워크는 자동으로 OP 효과.

**메서드**: 네트워크 weight 공유로 환경 symmetry group `G` 에 equivariant.
임의의 input transformation `g·x` 에 대해 output 도 `g·f(x)` 로 변함.

**환경**: Hanabi-like grid game. Other-Play 와 동등하지만 *학습 시간 동일*
(OP 는 partner 시점 augment 로 비용 ↑).

**본 연구와의 관계**: 가장 *cheap* 한 ZSC 방법 — 우리 trainer 도 적용
가능. 단 Overcooked 에는 명확한 symmetry 가 적어 효과 제한적.

---

### 🍳 Yu, Mao et al. (2024) — *Mastering Zero-Shot Interactions*
*ICML 2024* — `papers/pdf/yu_2024_zsc_evolutionary.pdf` (arxiv 2402.03136)

**한 줄**: *cooperative + competitive* 동시 simultaneous game 의 ZSC.
HSP 의 evolutionary 확장.

**메서드**: hidden-utility self-play (Yu 2023 ICLR) 위에 evolutionary
population. 매 generation 에서 utility-shifted partner 와 cross-play 평가.

**환경**: Overcooked + Coin Game (zero-sum) + Iterated PD. 다양 setting.

**본 연구와의 관계**: 우리가 *cooperative only* 만 다루지만, 같은 OM
representation 이 adversarial 에서도 통하는지 확인할 때 reference.

---

## [4] ZSC 평가 도구

### 🛠️ Ruan et al. (2024) — *ZSC-Eval*
*NeurIPS 2024 (Datasets & Benchmarks)* — `papers/pdf/ruan_2024_zsc_eval.pdf`
(arxiv 2310.05208)

**한 줄**: ZSC 의 **표준 벤치마크 + 도구** — SP/FCP/MEP/TrajeDi/HSP/COLE/E3T
모두 통합된 평가.

**기여**:
- Pre-trained 정책 풀 (HuggingFace `Leoxxxxh/ZSC-Eval-policy_pool`)
- 환경: Overcooked + LBF + GRF
- Metric: cross-play score, robust seed selection, Bonferroni correction

**본 연구와의 관계**: 우리 향후 cross-play 비교의 *표준 도구*. 자체 partner
풀 만들 필요 없이 그들 풀에 우리 method 평가 가능.

---

### 🛠️ Wang et al. (2024) — *Noisy Zero-Shot Coordination*
*arxiv preprint, 2024-11* — `papers/pdf/noisy_zsc_2024.pdf` (arxiv 2411.04976)

**한 줄**: 기존 ZSC 의 *common knowledge* 가정 깨짐 — partner 가 noisy /
miscalibrated 일 때 robust ZSC 알고리즘.

**메서드**: partial common knowledge 를 모델 — partner 가 어떤 prior 를
가지는지 자체가 uncertain. Bayesian belief over belief 추가.

**환경**: Toy noisy lever + Hanabi noisy variant.

**본 연구와의 관계**: *현실적인* ZSC 의 step. 인간 partner 는 항상 noisy.

---

## [5] 최신 (2024-2025) 벤치 / 일반화

### 🆕 Jha et al. (2025) — *Cross-environment Cooperation*
*arxiv 2025-04* — `papers/pdf/jha_2025_cross_env_coop.pdf` (arxiv 2504.12714)

**한 줄**: 여러 협력 환경 (Overcooked + Hanabi + others) **동시 학습** →
unseen 환경으로 zero-shot transfer.

**메서드**: multi-task PPO + environment-conditioned policy. 환경 별 task
embedding 으로 policy 가 *환경 정체성* 을 input 으로 받음. transfer 시
빠른 fine-tune 또는 zero-shot.

**환경**: Overcooked / Hanabi / Cleanup / Harvest 동시.

**본 연구와의 관계**: 우리 multi-env 인프라 의 *next phase*. ToM 표현이
환경 invariant 면 cross-env transfer 가능.

---

### 🆕 OvercookedV2 (2025)
*arxiv 2025-03* — `papers/pdf/overcookedv2_2025.pdf` (arxiv 2503.17821)

**한 줄**: 기존 Overcooked layout 이 *너무 풀린다* 는 비판 + 새 V2 layout
공개. 강한 ZSC + ToM 요구.

**기여**: 9 layout (기존 5 → 새 4). 기존 SOTA (FCP/MEP) 가 새 layout 에선
SP 와 거의 차이 없음. 즉 기존 벤치는 ZSC 차이를 noise 안에 묻고 있었음.

**본 연구와의 관계**: 우리 향후 평가의 *대체 벤치*. asym 같은 쉬운 layout
에 머물지 말고 V2 의 어려운 layout 에서 시도.

---

### 🆕 Ad-Hoc Human-AI Coordination Challenge (2025)
*arxiv 2025-06* — `papers/pdf/ahcc_2025_ad_hoc_challenge.pdf` (arxiv 2506.21490)

**한 줄**: held-out 인간 / AI partner 와의 ZSC 를 위한 **공식 대회 + 벤치**
플랫폼. 2026 NeurIPS 동반 가능성.

**기여**: 표준 평가 protocol, leaderboard, partner 풀, eval 인프라. 학계가
ZSC 결과를 *통일된 방식* 으로 비교 가능.

**본 연구와의 관계**: 미래 벤치 — 우리 method 가 진짜 publishable level
이면 여기서 score 내야.

---

## Venue 분포

```
ICML        7  ToMnet, BAD, Other-Play, OBL, TrajeDi, COLE, ZSC-evolutionary
NeurIPS     6  Overcooked, FCP, MEP (spotlight), Equivariant ZSC, SED, ZSC-Eval (D&B)
ICLR        2  SAD, R3D2 / Generalist Hanabi
Journal     3  Hanabi Challenge (AIJ 2020), COLE-extended (JAIR 2024), Conventions (JAAMAS 2025)
arxiv only  4  Noisy ZSC, Cross-env Coop, OvercookedV2, AHCC
```

→ ToM/ZSC 의 메인 venue 는 **ICML + NeurIPS**. ICLR 은 비주류, AAAI/RLC 는
거의 없음. AAAI/RLC 노린다면 분야의 *주류 venue 가 아님* 인지하고 시작.

---

## 메서드 비교표

| 메서드 | 종류 | self-play | cross-play | 학습량 | 핵심 idea |
|------|------|---------|---------|------|---------|
| Vanilla SP | baseline | mid-high | low | 1× | self-play |
| BAD | belief | high (Hanabi 24) | n/a | 1× | public belief MDP |
| SAD | belief | mid-high (Hanabi 22) | n/a | 1× | greedy intent share |
| OBL | belief | **highest** (Hanabi 24+) | high | 5-10× | counterfactual fixed-point |
| Other-Play | symmetry | mid | high | 1.2× | symmetry randomization |
| Equivariant | symmetry | mid | high | 1× | architecture |
| FCP | population | mid-high | high | N×K | seed×ckpt pool BR |
| TrajeDi | population | mid | high | N× | trajectory diversity |
| MEP | population | mid-high | **highest** | N× | entropy bonus |
| COLE | population | mid-high | **highest** | N× generations | open-ended adversary |
| ToMnet | meta-RL | n/a | n/a | meta | predict from obs |
| **본 TOM+BAD** | aux | mid (capacity 효과) | ❓ untested | 1× | trajectory aux + BAD route |

---

## 본 연구의 함의

negative finding (TOM+BAD = capacity 효과) 은 학계 합의에 부합:

1. **Self-play 점수만으로는 OM 효과 측정 불가**: Hu 2020 (OP), Strouse 2021
   (FCP), Bard 2019 모두 강조. 우리 capacity-confound finding 은 그 합의의
   재확인.
2. **Aux loss 만으로는 부족**: 분야가 BAD → SAD → OBL 로 *알고리즘 자체* 를
   복잡화한 것은 단순 aux 가 약함을 인정한 결과.
3. **Population diversity 가 ZSC 의 정답**: FCP/TrajeDi/MEP/COLE 모두
   같은 메시지. 본 연구가 publishable 이려면 이 흐름 위에서 *추가 향상*
   필요.
4. **현재 frontier 는 cross-env / cross-config 일반화**: Jha 2025, R3D2 2025,
   AHCC 2025. 단일 환경 self-play 향상은 더 이상 새 결과 아님.

---

## 추천 reading order

### 입문 (반드시 읽어야 할 6편)

1. **Rabinowitz 2018 (ToMnet)** — ToM 정의
2. **Carroll 2019 (Overcooked)** — 메인 환경
3. **Hu 2020 (Other-Play)** — self-play 의 한계
4. **Strouse 2021 (FCP)** — population 방법론
5. **Hu 2021 (OBL)** — Hanabi belief 의 정점
6. **Ruan 2024 (ZSC-Eval)** — 정직한 평가법

### 심화 (분야별)

**Hanabi belief 흐름**:
- Bard 2019 → BAD → SAD → OBL → SED → Stechbarth 2025 → R3D2

**Overcooked 협력 흐름**:
- Carroll 2019 → FCP → MEP / TrajeDi → COLE → OvercookedV2

**ZSC 인프라 / 일반화**:
- Other-Play → Equivariant → ZSC-Eval → Noisy ZSC → Cross-env → AHCC

---

## 미해결 문제

1. **ToM 의 *그라운딩***: aux head 의 belief 가 실제 의미와 grounded 인가?
   현재는 self-supervised 라 임의 mapping 가능. 인간 인지의 ToM 처럼
   referential 한 belief 가 가능한가?
2. **계산 효율**: OBL/FCP 같은 강한 방법은 SP 의 5-30배 컴퓨트. 학생/연구실
   레벨에서 재현 어려움. 효율적 변형이 미해결.
3. **Cross-env 일반화**: Jha 2025 가 시작했지만 환경 4-5 개 수준. 100 개
   환경에 동시 학습한 ToM agent 는 아직 없음.
4. **인간 partner 와의 진짜 ZSC**: AHCC 2025 가 시도. 인간이 진짜로 임의
   convention 을 깨트리며 협력하는 능력을 AI 가 흉내낼 수 있는가?
5. **Convention 의 *해석 가능성***: SED 가 시도했지만, agent 의 의도가
   인간에게 *납득* 되는 형태로 emerge 시키는 일반 방법은 미해결.

---

## 참고: 코드 / 재현

| 논문 | 공식 repo |
|------|---------|
| Carroll 2019 | github.com/HumanCompatibleAI/overcooked_ai (본 저장소 `external/`) |
| Strouse 2021 (FCP) | 비공개 (DeepMind) |
| Lupu 2021 (TrajeDi) | github.com/ALupu/TrajeDi |
| Zhao 2021 (MEP) | github.com/SamuelZRG/maximum_entropy_population_zsc |
| Li 2023 (COLE) | github.com/liyang619/COLE-Source-Code |
| Ruan 2024 (ZSC-Eval) | github.com/sjtu-marl/ZSC-Eval |
| Hu 2021 (OBL) | github.com/facebookresearch/off-belief-learning |
| Hu 2020 (SAD) | github.com/facebookresearch/hanabi_SAD |
| Nekoei 2025 (R3D2) | github.com/chandar-lab/R3D2 (예상) |

PantheonRL (Stanford) 의 `external/PantheonRL` 도 SB3 기반의 깔끔한 구현
참고 가능.
