# OnCOM 1차 파일럿 (2026-05-12)

**Online Continual Opponent Modeling** 2-phase 파이프라인 첫 end-to-end 실험.
구현 + 학습 + 평가 + latent 분석까지 한 사이클 완료.

> 코드: `src/tom/world_model/`, `src/tom/opponent_pool/`, `src/tom/envs/overcooked_solo.py`, `src/tom/training/{world_model_trainer,oncom_trainer}.py`, `scripts/{train_world_model,train_oncom,eval_oncom}.py`

---

## 실험 셋업

- **환경**: Overcooked `asymmetric_advantages`, partial obs (`view_radius=2`, 5×5 window), horizon 400
- **Partner pool (train)**: 10종 scripted — NOOP / Dir×4 / Wander×3 / Random×2
- **Partner pool (held-out)**: 4종 disjoint — `wander_held_a/b`, `random_held`, `dir_N_held`
- **OnCOM 아키텍처**:
  - Encoder φ (Phase 1 init **or** scratch)
  - Trajectory encoder η: GRU over partner (obs, action) 시퀀스 (K=16)
  - Conditional policy π(a | φ(o), z_opp) + 보조 OM head π̂(a^opp | …)
  - Contrastive: InfoNCE + momentum target (MoCo style)

## 학습 3건

| 단계 | env steps | 시간 | 핵심 메트릭 |
|------|---------|------|-----------|
| Phase 1: World Model | 1.0 M | ~3분 | `l_dyn=0.0008`, `l_rew=0.0005` |
| Phase 2: OnCOM-scratch (encoder full trainable) | 1.5 M | ~20분 | training ret **75.8**, om_acc 0.80, c_loss 0.40 |
| Phase 2: OnCOM-LoRA (Phase 1 init + LoRA r=8) | 1.5 M | ~20분 | training ret **0.4** (collapse), om_acc 0.83, c_loss 0.42 |

### 학습 곡선 (training return)

```
step       scratch     lora
100K       12.1        18.5
300K       22.2        26.1
500K       19.1        17.2
800K        9.6         3.6     ← shaping anneal 끝나는 지점
1100K      30.8         0.8
1400K      70.8         0.4
```

→ shaping 제거 직후 lora 정책 회복 못 함, scratch 는 late ramp.

## 평가

### Protocol P1 — held-out partner 점수 (10 episodes 평균 ± std)

| 파트너 | scratch | lora |
|------|---------|------|
| `wander_held_a` | **107.5 ± 49.1** | 7.6 ± 4.0 |
| `wander_held_b` | **117.8 ± 32.5** | 10.1 ± 3.0 |
| `random_held` | **93.9 ± 31.8** | 8.0 ± 2.7 |
| `dir_N_held` | **138.0 ± 44.1** | 7.3 ± 5.0 |

→ scratch 가 모든 held-out 에 10× 이상 압승.  
*caveat*: eval shaped coef=0.5 (training 끝에는 0). 따라서 절대 점수는 *shaping 포함*. 비교는 같은 조건이라 유효.

### Protocol P3 — z_opp latent 공간 (cosine 유사도, scratch ckpt)

**같은 class 안 강한 cluster**:
```
random_0  ↔ random_1            +0.78
random_0  ↔ random_held         +0.88   ★ train→heldout 전이
random_1  ↔ random_held         +0.91   ★
dir_N     ↔ dir_N_held          +0.78   ★ direction 전이
wander_0  ↔ wander_1            +0.49
wander_2  ↔ wander_held_a       +0.63   ★
```

**다른 class 간 negative**:
```
dir_N     ↔ wander_0            -0.33
dir_W     ↔ random_1            -0.30
wander_2  ↔ dir_N_held          -0.38
```

→ **본 적 없는 held-out partner 도 같은 class 면 train pool 의 같은 class 와 latent 가 일치**. Contrastive 가 *명시적인 opponent ID* 없이도 class-aware 표현을 자동 학습.

---

## 결과 해석

### Claim 별 검증

| Claim | 가설 | 결과 |
|------|-----|------|
| **C1** | World model pretrain 이 OM 학습 보조 | ❌ **반증** — scratch >> lora |
| **C2** | Continual 적응 (held-out partner 잘 다룸) | 🟡 부분 — held-out 100+ 점수 도달, 단 sequential continual eval 미수행 |
| **C3** | z_opp latent 가 opponent-semantic | ✅ **강한 증거** — train→heldout class 전이 검증 |

### 왜 WM pretrain 이 hurt 했나? (가설)

1. **Phase 1 데이터 분포 미스매치**: WM 은 *random learner policy* 로 학습됨 → encoder 가 "random rollout dynamics" 만 알게 됨
2. **LoRA bottleneck 한계**: r=8 으로는 그 mismatched encoder 를 *policy-useful* 표현으로 못 재형성
3. **Shaping anneal 직후 회복 실패**: scratch 는 encoder 전체 재학습으로 회복, lora 는 frozen base 위에 작은 delta 만이라 막힘

### 가장 paper-worthy 한 발견

원래 plan 의 *model-based* 측면은 약하지만, **contrastive trajectory encoder** 의 emergent 표현 학습이 강함:

> *"Episode 내 짧은 partner trajectory 만 보고도 opponent class 를 자동으로 식별하는 latent 가 contrastive InfoNCE 만으로 emerge. Train pool 에 없는 새 partner 도 같은 class 면 그 class 의 train partner 와 latent 가 일치 (cosine 0.78-0.91)."*

이게 paper backbone 가능.

---

## Refined paper framing

원안: *"Model-based + Continual + OM"* — model-based 부분 reject  
신안: *"Contrastive Opponent Representation Learning for Online Adaptation"*

핵심 contribution:
1. Scripted partner pool 로 short trajectory contrastive 학습
2. Held-out partner 와도 latent 일치 → zero-shot class-level adaptation
3. Online: episode 시작 시 K-step partial trajectory 만으로도 추정 가능

---

## 다음 작업 후보 (우선순위)

1. **🥇 Sequential continual eval (Protocol P2)** — held-out partner 순차 노출, forgetting 측정
2. **🥈 Capacity-matched 통제** — same-param vanilla 와 비교 (지난번 confound 우회 확인)
3. **🥉 WM pretrain 변종**:
   - encoder_mode=`full` + WM init (warm start 만)
   - WM 학습 데이터를 trained policy rollout 으로
4. **t-SNE 시각화** — z_opp 2D embedding 을 paper figure 로
5. **글 작업** — intro / related work / method draft

## 산출물

```
runs_world_model/v0/
  wm_000505600.pt
  wm_001004800.pt
  wm_final.pt

runs_oncom/v0_scratch/
  ckpt_*, ckpt_final.pt
  p1.json     (held-out P1 결과)
  p3.json     (latent space 분석)

runs_oncom/v0_lora/
  ckpt_*, ckpt_final.pt
  p1.json     (collapsed)
```

## 환경 정보

- conda env: `tom-coop` (python 3.10, torch 2.x, overcooked-ai 1.1.0, numpy<2)
- GPU: Blackwell sm_120 (PyTorch nightly cu128)
- 학습 wall-clock 총: ~50분 (병렬 + 순차 혼합)
