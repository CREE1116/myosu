# 묘수 — 고양이와 두는 오목

오목을 좋아하는 고양이 까망이, 하양이와 렌주 룰로 한 판 두는 웹 앱입니다.
AlphaZero 방식으로 직접 학습시킨 작은 신경망이 브라우저 안에서 MCTS로 수를 읽습니다.

**바로 두기: https://myosu-mks.pages.dev** (홈 화면에 설치하면 오프라인에서도 됩니다)

## 무엇을 할 수 있나

- **대국** — 까망이(보통)나 하양이(쉬움)를 골라 흑/백으로 둡니다. 렌주 금수(3·3, 4·4, 장목)를 지킵니다.
- **훈수** — 막히면 다른 고양이가 와서 한 수 추천이나 필승 수를 한 번씩 알려 줍니다.
- **복기** — 끝난 판에서 고양이가 승부처를 짚어 줍니다. 놓친 필승(넷 수순을 끝까지 탐색해 확인)과,
  엔진이 다시 읽어 "이 수 다음부터 진 판"이라고 확인한 수만 보여 줍니다. 그 자리부터 같은 고양이와 다시 둘 수도 있습니다.
- **묘수풀이** — 넷을 이어서 이기는 문제 30개(넷 이어 가기 · 금수 함정 · 긴 수순). 복기에서 놓친 필승도 문제로 모입니다.

## 구조

| 경로 | 내용 |
|---|---|
| `src/` | 웹 앱 (Vite + TypeScript, 프레임워크 없음). 렌주 판정, VCF 탐색, 브라우저 MCTS, 고양이 대사와 UI |
| `src/assets/` | 까망이·하양이 스프라이트 ([별도 라이선스](src/assets/LICENSE.md)) |
| `models/` | 배포 중인 모델 가중치 하나 (`gen_314.bin`, 37만 파라미터) |
| `static/` | PWA 파일 (아이콘, manifest, service worker) |
| `arch-bench/` | 학습 코드 (Python): 자가대국 리그, 챔피언 판정, 학습 루프 |
| `engine-rs/` | 학습용 Rust MCTS 엔진 (Python에서 동적 라이브러리로 부름) |
| `scripts/gen-puzzles.ts` | 묘수풀이 문제 생성기 |

## 실행

```bash
npm install
npm run dev      # http://localhost:5173
npm test         # vitest
npm run build    # dist/ 에 앱과 모델 하나를 묶음
```

빌드는 `arch-bench/checkpoints/`에 학습 결과가 있으면 그 챔피언을, 없으면 `models/`의 모델을 묶습니다.
특정 세대를 고르려면 `GOMOKU_DEPLOY_GEN=314 npm run build`.

## 학습

학습 루프(`arch-bench/train_az_daemon.py`)는 자가대국 → 리플레이 버퍼 → 학습 → 챔피언과 40판 대국으로 승급을 판정합니다.
신경망 학습에는 작성자의 별도 라이브러리 `rg`(Rust 자동미분, Metal 가속)를 쓰기 때문에, 이 저장소만으로는 학습을 그대로 재현할 수 없습니다.
Rust MCTS 엔진은 `cd engine-rs && cargo build --release`로 빌드합니다.

## 라이선스

- 코드: [MIT](LICENSE)
- 까망이·하양이 캐릭터와 그림(`src/assets/`, `static/icons/`, `docs/`의 캐릭터 설정): 모든 권리 보유. 자세한 내용은 [src/assets/LICENSE.md](src/assets/LICENSE.md).
