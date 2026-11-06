# ComfyUI-FlowMomentum

Custom samplers for rectified-flow models (Anima / Cosmos-style),
built around one question: how far can you get by changing the
*stepping rule* instead of the model?

All implementations are original clean-room code, written for
ComfyUI 0.44's sampler contract. This pack is Anima-family only,
by design.

## Nodes

### Flow Sampler (Unified)

One sampler node with an in-node `algorithm` dropdown — pick the
stepping rule the way KSampler picks a sampler name:

- `momentum` — flow Euler plus **velocity momentum**: an exponential
  moving average of the step velocity.

  ```
  v_t = (x_t - denoised_t) / sigma_t
  m_t = beta * m_{t-1} + (1 - beta) * v_t
  x_{t+1} = x_t + m_t * (sigma_{t+1} - sigma_t)
  ```

  `strength` is beta (0 = plain Euler, so the node A/B-tests itself).
- `ab2` — variable-step Adams–Bashforth 2 extrapolation.
- `euler` — plain flow Euler.

An optional `sigmas` input accepts an external grid (e.g. Flow
Sigma Grid below); unconnected, it uses the built-in linear-shift
grid (`sigma = shift * t / (1 + (shift - 1) * t)`, shift 3.0 — do
not stack with an AuraFlow model patch, the shift would apply
twice).

### Flow Momentum Sampler

The momentum sampler as a standalone node (same math as Unified's
`momentum` mode, with its own `momentum` knob).

### Flow Extrap Sampler (AB2)

Variable-step Adams–Bashforth multistep sampler. `order 1` is plain
Euler; `order 2` extrapolates the velocity from the previous step,
with coefficients adjusted for the local step-size ratio
((1 + r/2), −r/2). Accepts an optional external `sigmas` input.

### Flow Sigma Grid

A standalone SIGMAS node: the flow linear-shift grid with a
warpable parameterisation.

- `linear` — the standard grid.
- `detail` — more steps land in the low-sigma detail region.
- `smooth` — denser at both ends, sparser in the middle.
- `curve_strength` blends the warp in (0 = linear).

## Verified on real hardware

Tested on an RTX PRO 6000 (MIG 24 GB) pod,mac studio m3 ultra 512gb, ComfyUI 0.44,
Chiarezza 2.9B hybrid checkpoint:

- Two-stage chain: Anima Flow Corrective (UniPC) → Flow Momentum
  (beta 0.5, denoise 0.35) — visibly cleaner lines, shading and
  eye detail than the first stage alone.
- Unified (`ab2`) + Flow Sigma Grid (`detail` 0.5), standalone
  from EmptyLatentImage — clean generation.

Implementation note for sampler authors: an EmptyLatentImage
latent has 4 channels while Anima expects 16, so a standalone
sampler node must call
`comfy.sample.fix_empty_latent_channels(model, latent["samples"])`
on the samples tensor before sampling — the standard KSampler
procedure that custom samplers are easy to skip.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/seren-alegro/ComfyUI-FlowMomentum
# restart ComfyUI
```

Or copy this folder as `ComfyUI-FlowMomentum` into `ComfyUI/custom_nodes`
manually, then restart ComfyUI.

## Tests

`tests/` — sigma-grid properties, momentum exactness on a
constant-velocity field for any beta, beta = 0 equivalence with
hand-rolled Euler, AB2 error vs Euler on a curved field. Needs
torch (run inside a ComfyUI environment).

## License

MIT

---

## 한국어 요약

Anima 계열 rectified-flow 모델용 커스텀 샘플러 팩이에요. 속도
모멘텀(EMA), AB2 외삽, 시그마 그리드, 그리고 셋을 하나로 합친
Unified 노드(노드 안에서 알고리즘 선택)를 담고 있어요. RTX PRO
6000 포드와 mac studio m3 ultra 512gb에서 2단계 체인과 단독 실행 모두 실측을 마쳤어요.
