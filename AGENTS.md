<!-- oh-my-harness:start:always-commit -->
## 커밋 필수 규칙
- 작업으로 생성하거나 수정한 모든 추적 대상 파일은 작업 종료 전에 커밋한다.
- 작업 종료 전 `git status`를 확인하고 의도하지 않은 미커밋 변경사항을 남기지 않는다.
- 임시 파일, 비밀정보, 빌드 산출물은 커밋하지 않는다.
- 검증에 실패한 변경사항은 커밋하지 않는다.

<!-- oh-my-harness:end:always-commit -->


<!-- oh-my-harness:start:atomic-commits -->
## 원자적 커밋 규칙
- 각 커밋은 하나의 독립적인 논리적 변경만 포함해야 한다.
- 서로 관련 없는 기능, 리팩터링, 포맷 변경, 문서 변경을 하나의 커밋에 섞지 않는다.
- 구현과 해당 구현에 직접 대응하는 테스트는 같은 커밋에 포함할 수 있다.
- 커밋 전에 staged diff를 검토하고 부분적으로 완료된 변경이나 디버깅 코드를 제거한다.
- 커밋 메시지는 변경 목적을 명확하고 간결하게 설명해야 한다.

<!-- oh-my-harness:end:atomic-commits -->


<!-- oh-my-harness:start:omh-loop-protocol -->
## Autonomous Loop Protocol

The autonomous loop runs one work order per fresh session. `WORKPLAN.md` is the single source of truth.

- Read `WORKPLAN.md` first; it is the single source of truth.
- Pick the next unchecked task and implement it exactly as its work order in `docs/work-orders/<ID>.md` says. Make no design decisions.
- No work order, no work: mark the task "BLOCKED: no work order" and stop. Never write your own work order.
- Run the work order's acceptance commands before ticking any checkbox.
- Architect-only, never edited by the loop: none declared — add paths to `loop.architectOnly` in harness.yaml.
- If a task needs a human, or after three failed attempts, mark it "BLOCKED: <reason>" and move on. Never idle waiting for a person.
- One task, one commit; update the checkbox and the progress log in the same commit.
- Print exactly OMH_GOAL_COMPLETE as the final line ONLY when every task is done and verified; otherwise never mention that string in any form.

Start: `omh loop start`
Watch: `omh loop status`, then `tail -f .omh/state/loop/runs/<runId>/events.jsonl`
Stop: `omh loop stop` (`--now` to skip the grace period)
<!-- oh-my-harness:end:omh-loop-protocol -->
