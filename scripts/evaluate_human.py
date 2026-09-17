import argparse
import hashlib
import json
import mimetypes
import random
import re
import sys
import threading
from collections import OrderedDict, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts.evaluate_dataset import build_episode_content, content_sha256  # noqa: E402
from source.dataset_io import (  # noqa: E402
    DATASET_SCHEMA_VERSION,
    atomic_write_json,
    load_episode,
    load_json,
    load_manifest,
    utc_timestamp,
    validate_dataset,
)
from source.difficulty import DIFFICULTY_LEVELS  # noqa: E402
from source.multimodal import format_command, get_control_labels  # noqa: E402
from source.preprocess import TEST_TYPES  # noqa: E402
from source.prompts import PROMPT_PREFIX  # noqa: E402


HUMAN_PROTOCOL_VERSION = "human_eval_v1"
HUMAN_PROMPT_VERSION = "model_prompt_v1_human_response_ui"
RECENT_PAIR_WINDOW = 36


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run a blinded, resumable local web interface for human evaluation "
            "of a frozen VisionRecBench dataset."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "results" / "human",
    )
    parser.add_argument("--scenario", nargs="+", default=None)
    parser.add_argument(
        "--level",
        type=int,
        nargs="+",
        choices=DIFFICULTY_LEVELS,
        default=None,
    )
    parser.add_argument(
        "--test-type",
        dest="test_types",
        nargs="+",
        choices=TEST_TYPES,
        default=None,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--session-size",
        type=int,
        default=36,
        help="Show a break reminder after this many submitted trials.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.participant_id):
        parser.error(
            "--participant-id must contain only letters, numbers, '.', '_', "
            "or '-', and must start with a letter or number"
        )
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.session_size < 1:
        parser.error("--session-size must be positive")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    return args


def select_manifest_rows(rows, scenarios=None, levels=None, test_types=None):
    selected = list(rows)
    if scenarios:
        allowed = set(scenarios)
        selected = [row for row in selected if row["scenario"] in allowed]
    if levels:
        allowed = {int(level) for level in levels}
        selected = [
            row for row in selected
            if int(row["difficulty_level"]) in allowed
        ]
    if test_types:
        allowed = set(test_types)
        selected = [row for row in selected if row["test_type"] in allowed]
    return selected


def build_balanced_plan(rows, seed=0, recent_pair_window=RECENT_PAIR_WINDOW):
    """Interleave experimental cells while separating repeated nuisance pairs."""
    rng = random.Random(int(seed))
    groups = defaultdict(list)
    for row in rows:
        key = (
            int(row["scene"]),
            int(row["difficulty_level"]),
            row["test_type"],
        )
        groups[key].append(dict(row))
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    plan = []
    recent_pairs = []
    while any(groups.values()):
        active_keys = [key for key, values in groups.items() if values]
        rng.shuffle(active_keys)
        for key in active_keys:
            candidates = groups[key]
            selected_index = len(candidates) - 1
            recent = set(recent_pairs[-recent_pair_window:])
            for index in range(len(candidates) - 1, -1, -1):
                pair_id = candidates[index].get("nuisance_pair_id")
                if not pair_id or pair_id not in recent:
                    selected_index = index
                    break
            row = candidates.pop(selected_index)
            plan.append(row)
            pair_id = row.get("nuisance_pair_id")
            if pair_id:
                recent_pairs.append(pair_id)
    return plan


def _canonical_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def plan_configuration(args, metadata, episode_ids):
    return {
        "protocol_version": HUMAN_PROTOCOL_VERSION,
        "dataset_name": metadata["dataset_name"],
        "dataset_content_sha256": metadata["content_sha256"],
        "participant_id": args.participant_id,
        "random_seed": int(args.random_seed),
        "session_size": int(args.session_size),
        "filters": {
            "scenarios": sorted(args.scenario) if args.scenario else None,
            "levels": sorted(args.level) if args.level else None,
            "test_types": sorted(args.test_types) if args.test_types else None,
            "limit": args.limit,
        },
        "episode_ids": list(episode_ids),
    }


def study_root(output_root, metadata, participant_id):
    dataset_dir = metadata["dataset_name"].replace("/", "-")
    return (
        Path(output_root)
        / dataset_dir
        / participant_id
        / HUMAN_PROTOCOL_VERSION
    )


def human_result_path(root, record):
    return (
        Path(root)
        / "responses"
        / f"difficulty_level{record['difficulty_level']}"
        / record["test_type"]
        / record["scenario"]
        / f"{record['episode_id']}.json"
    )


def load_or_create_plan(args, metadata, selected_rows, root):
    plan_path = Path(root) / "plan.json"
    ordered_rows = build_balanced_plan(selected_rows, seed=args.random_seed)
    if args.limit is not None:
        ordered_rows = ordered_rows[: args.limit]
    expected = plan_configuration(
        args,
        metadata,
        [row["episode_id"] for row in ordered_rows],
    )

    if plan_path.is_file():
        if not args.resume:
            raise FileExistsError(
                f"Human evaluation plan already exists: {plan_path}. Use --resume."
            )
        existing = load_json(plan_path)
        mismatches = [
            key for key, value in expected.items()
            if existing.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "Existing human plan does not match this invocation "
                f"({', '.join(mismatches)}): {plan_path}"
            )
        return existing

    document = {
        **expected,
        "created_at": utc_timestamp(),
    }
    document["plan_id"] = _canonical_sha256(expected)
    atomic_write_json(plan_path, document)
    return document


def trial_token(plan_id, trial_order, episode_id):
    raw = f"{plan_id}:{trial_order}:{episode_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def public_trial_payload(record, token, completed, total, session_size):
    control_labels = record.get("control_labels") or get_control_labels(record["task"])
    commands = [
        format_command(step["command"], control_labels)
        for step in record["steps"]
    ]
    evidence = [
        {
            "step": index,
            "command": commands[index - 1],
            "image_url": f"/media/{token}/evidence/{index}",
        }
        for index in range(1, len(record["steps"]) + 1)
    ]
    return {
        "complete": False,
        "trial_token": token,
        "progress": {
            "completed": completed,
            "total": total,
            "current": completed + 1,
        },
        "instructions": PROMPT_PREFIX.strip(),
        "command_trace": commands,
        "evidence": evidence,
        "final_observation_url": f"/media/{token}/final",
        "answer_options": [
            {"index": index, "text": text}
            for index, text in enumerate(record["answer_options"], start=1)
        ],
        "session_size": session_size,
    }


def create_human_result(
    record,
    metadata,
    participant_id,
    plan_id,
    trial_order,
    session_size,
    choice,
    response_time_ms,
    active_response_time_ms,
    input_content_hash,
):
    correct = int(choice) == int(record["answer_index"])
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_name": metadata["dataset_name"],
        "dataset_content_sha256": metadata["content_sha256"],
        "episode_id": record["episode_id"],
        "episode_signature": record["episode_signature"],
        "scenario": record["scenario"],
        "scene": record["scene"],
        "seed": record["seed"],
        "model": f"human:{participant_id}",
        "evaluator_type": "human",
        "participant_id": participant_id,
        "evaluation_protocol_version": HUMAN_PROTOCOL_VERSION,
        "human_prompt_version": HUMAN_PROMPT_VERSION,
        "plan_id": plan_id,
        "trial_order": int(trial_order),
        "session_index": (int(trial_order) - 1) // int(session_size) + 1,
        "evaluated_at": utc_timestamp(),
        "input_content_sha256": input_content_hash,
        "response_time_ms": int(response_time_ms),
        "active_response_time_ms": int(active_response_time_ms),
        "difficulty_level": int(record["difficulty_level"]),
        "difficulty_name": record["difficulty_name"],
        "test_type": record["test_type"],
        "nuisance_pair_id": record.get("nuisance_pair_id"),
        "nuisance_signature": record.get("nuisance_signature"),
        "environment_template": record.get("environment_template"),
        "arm_type": record.get("arm_type"),
        "camera_view": record.get("camera_view"),
        "target_present": record["target_present"],
        "target_index": record["target_index"],
        "answer_index": record["answer_index"],
        "answer_text": record["answer_text"],
        "answer_options": record["answer_options"],
        "choice": int(choice),
        "valid": True,
        "correct": correct,
        "bad_response": 0,
    }


class HumanEvaluationStudy:
    def __init__(
        self,
        dataset_root,
        metadata,
        manifest_rows,
        plan,
        root,
        input_hash_fn=None,
    ):
        self.dataset_root = Path(dataset_root).resolve()
        self.metadata = metadata
        self.manifest_by_id = {
            row["episode_id"]: row for row in manifest_rows
        }
        self.plan = plan
        self.root = Path(root)
        self.participant_id = plan["participant_id"]
        self.session_size = int(plan["session_size"])
        self._lock = threading.Lock()
        self._input_hash_fn = input_hash_fn or self._default_input_hash
        self._records = OrderedDict()
        self._tokens = {}
        self._token_to_order = {}
        self._completed_orders = set()
        self._next_pending_order = 1
        for order, episode_id in enumerate(plan["episode_ids"], start=1):
            if episode_id not in self.manifest_by_id:
                raise ValueError(f"Plan episode is absent from manifest: {episode_id}")
            token = trial_token(plan["plan_id"], order, episode_id)
            self._tokens[order] = token
            self._token_to_order[token] = order
        self._validate_existing_results()

    def _record(self, order):
        if order in self._records:
            self._records.move_to_end(order)
        else:
            episode_id = self.plan["episode_ids"][order - 1]
            self._records[order] = load_episode(
                self.dataset_root,
                self.manifest_by_id[episode_id],
            )
            if len(self._records) > 4:
                self._records.popitem(last=False)
        return self._records[order]

    def _result_path_for_order(self, order):
        episode_id = self.plan["episode_ids"][order - 1]
        return human_result_path(self.root, self.manifest_by_id[episode_id])

    def _validate_existing_results(self):
        expected_common = {
            "dataset_content_sha256": self.metadata["content_sha256"],
            "participant_id": self.participant_id,
            "evaluation_protocol_version": HUMAN_PROTOCOL_VERSION,
            "plan_id": self.plan["plan_id"],
        }
        for order in range(1, len(self.plan["episode_ids"]) + 1):
            path = self._result_path_for_order(order)
            if not path.is_file():
                continue
            row = load_json(path)
            expected = {
                **expected_common,
                "episode_id": self.plan["episode_ids"][order - 1],
                "trial_order": order,
            }
            mismatches = [
                key for key, value in expected.items()
                if row.get(key) != value
            ]
            if mismatches:
                raise ValueError(
                    f"Existing human result metadata mismatch "
                    f"({', '.join(mismatches)}): {path}"
                )
            self._completed_orders.add(order)
        self._advance_next_pending_order()

    def _advance_next_pending_order(self):
        total = len(self.plan["episode_ids"])
        while (
            self._next_pending_order <= total
            and self._next_pending_order in self._completed_orders
        ):
            self._next_pending_order += 1

    def _default_input_hash(self, record):
        content = build_episode_content(record, -1, self.dataset_root)
        return content_sha256(content)

    def completed_count(self):
        return len(self._completed_orders)

    def next_order(self):
        if self._next_pending_order > len(self.plan["episode_ids"]):
            return None
        return self._next_pending_order

    def current_payload(self):
        with self._lock:
            order = self.next_order()
            total = len(self.plan["episode_ids"])
            if order is None:
                return {
                    "complete": True,
                    "progress": {"completed": total, "total": total},
                }
            completed = self.completed_count()
            return public_trial_payload(
                self._record(order),
                self._tokens[order],
                completed,
                total,
                self.session_size,
            )

    def submit(
        self,
        token,
        choice,
        response_time_ms,
        active_response_time_ms,
    ):
        with self._lock:
            current_order = self.next_order()
            if current_order is None:
                raise ValueError("The study is already complete.")
            if token != self._tokens[current_order]:
                raise ValueError("This trial is no longer current; reload the page.")
            record = self._record(current_order)
            if not isinstance(choice, int) or not 1 <= choice <= len(
                record["answer_options"]
            ):
                raise ValueError("Choice is outside the available answer options.")
            for name, value in (
                ("response_time_ms", response_time_ms),
                ("active_response_time_ms", active_response_time_ms),
            ):
                if not isinstance(value, int) or not 0 <= value <= 86_400_000:
                    raise ValueError(f"{name} must be an integer between 0 and 86400000.")
            result = create_human_result(
                record,
                self.metadata,
                self.participant_id,
                self.plan["plan_id"],
                current_order,
                self.session_size,
                choice,
                response_time_ms,
                active_response_time_ms,
                self._input_hash_fn(record),
            )
            path = self._result_path_for_order(current_order)
            if path.exists():
                raise ValueError("A response for this trial already exists.")
            atomic_write_json(path, result)
            self._completed_orders.add(current_order)
            self._advance_next_pending_order()
            completed = self.completed_count()
            return {
                "saved": True,
                "completed": completed,
                "total": len(self.plan["episode_ids"]),
                "break_due": completed % self.session_size == 0,
            }

    def media_path(self, token, kind, index=None):
        with self._lock:
            try:
                order = self._token_to_order[token]
            except KeyError as exc:
                raise ValueError("Unknown trial token.") from exc
            record = self._record(order)
            if kind == "final":
                descriptor = record["steps"][-1]["observation"]
            elif kind == "evidence":
                if index is None or not 1 <= index <= len(record["steps"]):
                    raise ValueError("Evidence index is out of range.")
                descriptor = record["steps"][index - 1]["evidence"]
            else:
                raise ValueError("Unsupported media kind.")
        path = (self.dataset_root / descriptor["path"]).resolve()
        try:
            path.relative_to(self.dataset_root)
        except ValueError as exc:
            raise ValueError("Media path escapes the dataset root.") from exc
        if not path.is_file():
            raise ValueError("Media file is missing.")
        return path


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VisionRecBench Human Evaluation</title>
  <style>
    :root { color-scheme: dark; --bg:#0d1117; --panel:#161b22; --line:#30363d; --text:#e6edf3; --muted:#9da7b3; --accent:#58a6ff; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,sans-serif; line-height:1.45; }
    header { position:sticky; top:0; z-index:5; display:flex; justify-content:space-between; gap:16px; padding:12px 20px; background:rgba(13,17,23,.96); border-bottom:1px solid var(--line); }
    main { max-width:1580px; margin:0 auto; padding:20px 20px 150px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:18px; }
    .muted { color:var(--muted); }
    pre { white-space:pre-wrap; margin:0; font:14px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace; }
    .evidence { margin:22px 0; }
    .evidence h3 { font-size:15px; font-weight:600; margin:0 0 8px; }
    img { display:block; width:100%; height:auto; margin:auto; background:#000; border:1px solid var(--line); border-radius:6px; }
    .final img { max-width:1024px; }
    #answerBar { position:fixed; z-index:10; left:0; right:0; bottom:0; padding:12px 20px; background:rgba(13,17,23,.97); border-top:1px solid var(--line); }
    #answerInner { max-width:1540px; margin:auto; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
    .option, #submit { border:1px solid var(--line); border-radius:7px; color:var(--text); background:var(--panel); padding:10px 14px; cursor:pointer; font-size:15px; }
    .option.selected { border-color:var(--accent); box-shadow:0 0 0 2px rgba(88,166,255,.25); }
    #submit { margin-left:auto; background:#238636; border-color:#2ea043; font-weight:650; }
    #submit:disabled { opacity:.45; cursor:not-allowed; }
    #message { padding:60px 20px; text-align:center; font-size:20px; }
    @media (max-width:700px) { main { padding-left:8px; padding-right:8px; } header { padding:10px; } #submit { width:100%; margin-left:0; } }
  </style>
</head>
<body>
  <header><strong>VisionRecBench Human Evaluation</strong><span id="progress" class="muted">Loading…</span></header>
  <main id="content"><div id="message">Loading trial…</div></main>
  <div id="answerBar" hidden><div id="answerInner"><span class="muted">Select an answer:</span><div id="options"></div><button id="submit" disabled>Submit (Enter)</button></div></div>
  <script>
    let trial = null, selected = null, submitting = false, startedAt = 0, activeStartedAt = 0, activeMs = 0;
    const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    function activeNow() { return activeMs + (document.hidden ? 0 : performance.now() - activeStartedAt); }
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) activeMs += performance.now() - activeStartedAt;
      else activeStartedAt = performance.now();
    });
    async function loadTrial() {
      selected = null;
      submitting = false;
      const response = await fetch('/api/trial', {cache:'no-store'});
      trial = await response.json();
      if (trial.complete) {
        document.getElementById('progress').textContent = `${trial.progress.completed} / ${trial.progress.total}`;
        document.getElementById('content').innerHTML = '<div id="message"><h2>Evaluation complete</h2><p class="muted">All responses have been saved. You may close this page.</p></div>';
        document.getElementById('answerBar').hidden = true;
        return;
      }
      const p = trial.progress;
      document.getElementById('progress').textContent = `Trial ${p.current} of ${p.total} · ${p.completed} saved`;
      const commands = trial.command_trace.map(x => `- ${esc(x)}`).join('\n');
      const evidence = trial.evidence.map(item => `<section class="evidence"><h3>Visual evidence after ${esc(item.command)}</h3><img src="${item.image_url}" alt="Time-ordered visual evidence for step ${item.step}"></section>`).join('');
      document.getElementById('content').innerHTML = `<section class="panel"><pre>${esc(trial.instructions)}</pre></section><section class="panel"><h2>Motor-command trace</h2><pre>${commands}</pre></section><section><h2>Time-ordered visual evidence</h2>${evidence}</section><section class="panel final"><h2>Final camera view after the complete command trace</h2><img src="${trial.final_observation_url}" alt="Final camera view"></section>`;
      const optionRoot = document.getElementById('options');
      optionRoot.innerHTML = '';
      trial.answer_options.forEach(option => {
        const button = document.createElement('button');
        button.className = 'option';
        button.dataset.index = option.index;
        button.textContent = `${option.index}. ${option.text}`;
        button.onclick = () => choose(option.index);
        optionRoot.appendChild(button);
      });
      document.getElementById('submit').disabled = true;
      document.getElementById('answerBar').hidden = false;
      window.scrollTo(0, 0);
      startedAt = performance.now(); activeStartedAt = startedAt; activeMs = 0;
    }
    function choose(index) {
      selected = index;
      document.querySelectorAll('.option').forEach(button => button.classList.toggle('selected', Number(button.dataset.index) === index));
      document.getElementById('submit').disabled = false;
    }
    async function submit() {
      if (!trial || selected === null || submitting) return;
      submitting = true;
      const button = document.getElementById('submit'); button.disabled = true;
      const payload = {trial_token:trial.trial_token, choice:selected, response_time_ms:Math.round(performance.now()-startedAt), active_response_time_ms:Math.round(activeNow())};
      let response;
      try {
        response = await fetch('/api/respond', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
      } catch (error) {
        submitting = false; button.disabled = false;
        alert(`Could not reach the local server: ${error}`);
        return;
      }
      const result = await response.json();
      if (!response.ok) { alert(result.error || 'Could not save response. Reloading current trial.'); await loadTrial(); return; }
      if (result.break_due && result.completed < result.total) alert(`Break point reached (${result.completed}/${result.total}). Please rest before continuing.`);
      await loadTrial();
    }
    document.getElementById('submit').onclick = submit;
    document.addEventListener('keydown', event => {
      if (/^[1-9]$/.test(event.key) && trial) {
        const index = Number(event.key);
        if (trial.answer_options.some(option => option.index === index)) choose(index);
      } else if (event.key === 'Enter' && selected !== null) submit();
    });
    loadTrial().catch(error => { document.getElementById('content').innerHTML = `<div id="message">Failed to load: ${esc(error)}</div>`; });
  </script>
</body>
</html>
"""


def make_handler(study):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/trial":
                self._json(200, study.current_payload())
                return
            parts = [unquote(part) for part in parsed.path.split("/") if part]
            if len(parts) in (3, 4) and parts[0] == "media":
                token, kind = parts[1], parts[2]
                try:
                    index = int(parts[3]) if len(parts) == 4 else None
                    path = study.media_path(token, kind, index)
                    body = path.read_bytes()
                except (ValueError, OSError) as exc:
                    self._json(404, {"error": str(exc)})
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                )
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "private, max-age=3600")
                self.end_headers()
                self.wfile.write(body)
                return
            self._json(404, {"error": "Not found"})

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/api/respond":
                self._json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size.")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = study.submit(
                    payload.get("trial_token"),
                    payload.get("choice"),
                    payload.get("response_time_ms"),
                    payload.get("active_response_time_ms"),
                )
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(409, {"error": str(exc)})
                return
            self._json(200, result)

        def log_message(self, format_string, *args):
            if args and str(args[1]).startswith("4"):
                super().log_message(format_string, *args)

    return Handler


def main(argv=None):
    args = parse_args(argv)
    dataset_root = args.dataset.resolve()
    report = validate_dataset(
        dataset_root,
        verify_checksums=args.verify_checksums,
    )
    if not report["valid"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(1)
    metadata = load_json(dataset_root / "metadata.json")
    if metadata.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise SystemExit(
            f"Human evaluation requires dataset schema {DATASET_SCHEMA_VERSION}."
        )
    if not metadata.get("paired_nuisance", False):
        raise SystemExit("Human evaluation requires a paired-nuisance dataset.")

    manifest_rows = load_manifest(dataset_root)
    selected_rows = select_manifest_rows(
        manifest_rows,
        scenarios=args.scenario,
        levels=args.level,
        test_types=args.test_types,
    )
    if not selected_rows:
        raise SystemExit("No episodes match the requested filters.")
    root = study_root(args.output, metadata, args.participant_id)
    plan = load_or_create_plan(args, metadata, selected_rows, root)
    study = HumanEvaluationStudy(
        dataset_root,
        metadata,
        manifest_rows,
        plan,
        root,
    )
    completed = study.completed_count()
    total = len(plan["episode_ids"])
    print(
        f"Human evaluation: participant={args.participant_id!r}, "
        f"selected={total}, completed={completed}, pending={total - completed}",
        flush=True,
    )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(study))
    host, port = server.server_address[:2]
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"Open http://{display_host}:{port} in a browser. Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Submitted responses are already saved; use --resume to continue.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
