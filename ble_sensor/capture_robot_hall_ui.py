#!/usr/bin/env python3
"""Pygame operator console for real dual-foot Hall data collection.

The UI never opens a BLE connection itself.  It runs capture_robot_hall.py,
which owns hci0/hci1 only after its safety preflight succeeds.  All displayed
measurements remain raw Hall Bx/By/Bz counts, temperature and timing/health
metadata; no force, pressure, contact-force or friction value is inferred.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Iterable

import numpy as np

from dual_foot_bridge.capture_ipc import read_packet as read_capture_packet
from dual_foot_bridge.hall_display import (
    DualHallDisplayFilter,
    HallDisplayLayout,
    load_display_layout,
)


LEFT_ADDRESS = "98:A3:16:A1:BF:CA"
RIGHT_ADDRESS = "98:A3:16:A1:C1:2E"
REQUIRED_ADAPTERS = {"left": "hci0", "right": "hci1"}
CAPTURE_IPC = Path("/tmp/g1_foot_hall_capture.bin")
SAFETY_LABELS = {
    "area_clear": "周围人员和障碍物已清空",
    "unloaded": "双足完全卸力，未触碰 TPU",
    "robot_safe": "机器人断使能或刚性固定",
    "harness": "顶置吊架已承载并检查",
    "estop": "急停操作员已就位",
    "surface_safe": "低摩擦表面已固定且边缘防护完成",
}


@dataclass(frozen=True)
class Phase:
    key: str
    title: str
    duration_s: int
    instruction: str
    safety: tuple[str, ...]
    group: str


PHASES = (
    Phase(
        "suspended_unloaded",
        "01 悬空无载",
        180,
        "顶置吊架使双脚完全悬空、TPU 无任何接触；等待温漂/回弹稳定并记录原始 Hall。",
        ("area_clear", "unloaded", "robot_safe", "harness", "estop"),
        "无载稳定性",
    ),
    Phase(
        "dual_standing",
        "02 双足站立",
        90,
        "顶置吊架保护下双足静态站立，保持姿态稳定，不主动迈步。",
        ("area_clear", "harness", "estop"),
        "静态承载",
    ),
    Phase(
        "walking_straight",
        "03 低速直走",
        120,
        "顶置吊架保护下低速直线行走；记录机器人日志和同机单调时间戳用于离线对齐。",
        ("area_clear", "harness", "estop"),
        "动态行走",
    ),
)


def extract_json_document(text: str) -> dict[str, Any] | None:
    """Return the first complete JSON object embedded in command output."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "capture"


def required_checks_complete(phase: Phase, checks: dict[str, bool]) -> bool:
    return all(bool(checks.get(key, False)) for key in phase.safety)


def assess_capture_quality(manifest: dict[str, Any]) -> tuple[str, list[str]]:
    """Apply fixed acceptance thresholds without interpreting Hall as mechanics."""
    reasons: list[str] = []
    status = str(manifest.get("status", "missing"))
    if status != "complete":
        return "FAIL", [f"采集状态为 {status}，不是 complete"]

    rows = manifest.get("paired_rows", {})
    post_ready = int(rows.get("rows_after_ready", 0) or 0)
    both_valid = int(rows.get("both_valid_after_ready", 0) or 0)
    valid_ratio = both_valid / post_ready if post_ready else 0.0
    if valid_ratio < 0.95:
        reasons.append(f"双脚有效率仅 {valid_ratio * 100:.1f}%（FAIL <95%）")

    health = manifest.get("final_health", {}).get("feet", {})
    raw_timing = manifest.get("raw_frame_timing", {})
    rates: list[float] = []
    for side in ("left", "right"):
        foot = health.get(side, {})
        rejected = int(foot.get("rejected_frames", 0) or 0)
        rate = float(foot.get("sample_rate_hz", 0.0) or 0.0)
        rates.append(rate)
        if rejected > 0:
            reasons.append(f"{side} 存在 {rejected} 个坏帧（FAIL）")
        if rate < 80.0:
            reasons.append(f"{side} 仅 {rate:.1f} Hz（FAIL <80 Hz 原始采集下限）")
        raw = raw_timing.get(side, {})
        raw_rate = float(raw.get("mean_rate_hz", 0.0) or 0.0)
        raw_p95_ms = float(raw.get("interval_ms_p95", 0.0) or 0.0)
        if raw_rate and raw_rate < 80.0:
            reasons.append(
                f"{side} 全程平均 Notify 仅 {raw_rate:.1f} Hz（FAIL <80 Hz）"
            )
        if raw_p95_ms > 50.0:
            reasons.append(
                f"{side} Notify 间隔 P95={raw_p95_ms:.1f} ms（FAIL >50 ms）"
            )

    skew_ms = float(rows.get("abs_frame_skew_ns_p95", 0) or 0) / 1.0e6
    if skew_ms > 50.0:
        reasons.append(f"左右帧时间戳偏差 P95={skew_ms:.1f} ms（FAIL >50 ms）")

    if reasons:
        return "FAIL", reasons

    reviews: list[str] = []
    if valid_ratio < 0.99:
        reviews.append(f"双脚有效率 {valid_ratio * 100:.1f}%（复核 <99%）")
    for side, rate in zip(("left", "right"), rates):
        if rate < 95.0:
            reviews.append(f"{side} {rate:.1f} Hz（复核 <95 Hz 目标）")
        raw = raw_timing.get(side, {})
        raw_rate = float(raw.get("mean_rate_hz", 0.0) or 0.0)
        raw_p95_ms = float(raw.get("interval_ms_p95", 0.0) or 0.0)
        if raw_rate and raw_rate < 95.0:
            reviews.append(f"{side} 全程平均 Notify {raw_rate:.1f} Hz（复核 <95 Hz）")
        if raw_p95_ms > 20.0:
            reviews.append(f"{side} Notify 间隔 P95={raw_p95_ms:.1f} ms（复核 >20 ms）")
    if skew_ms > 20.0:
        reviews.append(f"左右帧时间戳偏差 P95={skew_ms:.1f} ms（复核 >20 ms）")
    return ("REVIEW", reviews) if reviews else ("PASS", ["固定质量门槛全部满足"])


def preflight_blockers(document: dict[str, Any] | None) -> list[str]:
    if not document:
        return ["尚未获得预检结果"]
    blockers: list[str] = []
    for adapter in document.get("missing_adapters", []):
        blockers.append(f"缺少蓝牙适配器 {adapter}")
    for process in document.get("competing_ble_processes", []):
        blockers.append(
            f"BLE 被 PID {process.get('pid', '?')} / {process.get('script', '?')} 占用"
        )
    if not blockers and not document.get("ready", False):
        blockers.append("预检未通过（原因未报告）")
    return blockers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.magnetic.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("logs/robot_capture_sessions")
    )
    parser.add_argument("--left-adapter", default="hci0")
    parser.add_argument("--right-adapter", default="hci1")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=920)
    return parser.parse_args()


class ProcessController:
    def __init__(self, project_dir: Path, args: argparse.Namespace) -> None:
        self.project_dir = project_dir
        self.args = args
        self.messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.capture: subprocess.Popen[str] | None = None
        self.capture_session: Path | None = None
        self.capture_started = 0.0
        self._capture_thread: threading.Thread | None = None
        self._preflight_running = False

    def command_base(self) -> list[str]:
        python = self.project_dir / ".venv/bin/python"
        return [
            str(python),
            str(self.project_dir / "capture_robot_hall.py"),
            "--config",
            str(self.args.config.resolve()),
            "--left-adapter",
            self.args.left_adapter,
            "--right-adapter",
            self.args.right_adapter,
        ]

    def request_preflight(self) -> bool:
        if self._preflight_running or self.capture is not None:
            return False
        self._preflight_running = True

        def worker() -> None:
            try:
                result = subprocess.run(
                    [*self.command_base(), "--preflight-only"],
                    cwd=self.project_dir,
                    capture_output=True,
                    text=True,
                    timeout=30.0,
                    check=False,
                )
                merged = result.stdout + ("\n" + result.stderr if result.stderr else "")
                self.messages.put(
                    ("preflight", (result.returncode, extract_json_document(merged), merged))
                )
            except Exception as error:
                self.messages.put(("preflight_error", str(error)))
            finally:
                self._preflight_running = False

        threading.Thread(target=worker, name="hall-preflight", daemon=True).start()
        return True

    def start_capture(self, phase: Phase, note: str, batch_id: str) -> Path:
        if self.capture is not None:
            raise RuntimeError("已有采集进程正在运行")
        attempt_stamp = datetime.now().strftime("%H%M%S")
        session_name = safe_component(
            f"{phase.title[:2]}_{phase.key}_{attempt_stamp}"
        )
        output_root = self.args.output_root.resolve() / batch_id
        session = output_root / session_name
        command = [
            *self.command_base(),
            "--output-root",
            str(output_root),
            "--session-name",
            session_name,
            "--duration",
            str(phase.duration_s),
            "--note",
            f"UI阶段={phase.title}; 指令={phase.instruction}; 操作员备注={note.strip()}",
        ]
        self.capture = subprocess.Popen(
            command,
            cwd=self.project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        process = self.capture
        self.capture_session = session
        self.capture_started = time.monotonic()

        def worker() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                self.messages.put(("log", line.rstrip()))
            return_code = process.wait()
            self.messages.put(("capture_done", (return_code, session)))

        self._capture_thread = threading.Thread(
            target=worker, name="hall-capture-output", daemon=True
        )
        self._capture_thread.start()
        return session

    def stop_capture(self) -> bool:
        if self.capture is None or self.capture.poll() is not None:
            return False
        os.killpg(self.capture.pid, signal.SIGINT)
        return True

    def clear_finished_capture(self) -> None:
        if self.capture is not None and self.capture.poll() is not None:
            self.capture = None

    @property
    def preflight_running(self) -> bool:
        return self._preflight_running


class HallCaptureUI:
    BG = (10, 15, 27)
    PANEL = (22, 31, 49)
    PANEL_2 = (29, 40, 62)
    TEXT = (232, 238, 247)
    MUTED = (151, 164, 184)
    CYAN = (56, 189, 248)
    GREEN = (52, 211, 153)
    YELLOW = (250, 204, 21)
    RED = (248, 113, 113)
    LINE = (54, 69, 96)

    def __init__(self, args: argparse.Namespace) -> None:
        import pygame

        self.pg = pygame
        pygame.init()
        pygame.key.start_text_input()
        self.project_dir = Path(__file__).resolve().parent
        self.args = args
        self.screen = pygame.display.set_mode(
            (max(1280, args.width), max(760, args.height)), pygame.RESIZABLE
        )
        pygame.display.set_caption("G1 双足 Hall 数据采集控制台")
        self.clock = pygame.time.Clock()
        self.fonts = self._load_fonts()
        self.controller = ProcessController(self.project_dir, args)
        self.hall_layout: HallDisplayLayout = load_display_layout(
            self.project_dir / "config/sensor_layout_a4_15.json"
        )
        self.hall_display = DualHallDisplayFilter()
        self.live_sample = None
        self.last_capture_sequence = -1
        self.viz_geometry_cache: dict[tuple[str, int, int], tuple[Any, ...]] = {}
        self.phase_index = 0
        self.checks = {key: False for key in SAFETY_LABELS}
        self.preflight: dict[str, Any] | None = None
        self.preflight_code: int | None = None
        self.preflight_time = 0.0
        self.note = "最终装配双足，顶置吊架保护"
        self.note_active = False
        self.logs: list[str] = []
        self.health: dict[str, Any] | None = None
        self.last_health_read = 0.0
        self.batch_id = ""
        self.last_result: tuple[str, list[str]] | None = None
        self.phase_status = ["pending" for _ in PHASES]
        self.notice = "正在检查远端适配器和 BLE 占用状态……"
        self.notice_color = self.YELLOW
        self.click_targets: list[tuple[Any, tuple[str, Any]]] = []
        self.running = True
        self.close_requested = False
        self.controller.request_preflight()

    def _load_fonts(self) -> dict[str, Any]:
        candidates = (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        )
        bold_candidates = (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Bold.otf",
        )
        regular = next((path for path in candidates if Path(path).exists()), None)
        bold = next((path for path in bold_candidates if Path(path).exists()), regular)
        font = self.pg.font.Font
        return {
            "tiny": font(regular, 11),
            "xs": font(regular, 15),
            "sm": font(regular, 18),
            "md": font(regular, 21),
            "lg": font(bold, 28),
            "xl": font(bold, 35),
        }

    def _text(self, value: str, pos: tuple[int, int], font: str = "sm", color=None) -> Any:
        surface = self.fonts[font].render(value, True, color or self.TEXT)
        self.screen.blit(surface, pos)
        return surface.get_rect(topleft=pos)

    def _wrap(self, value: str, width: int, font: str = "sm") -> list[str]:
        result: list[str] = []
        line = ""
        for char in value:
            trial = line + char
            if line and self.fonts[font].size(trial)[0] > width:
                result.append(line)
                line = char
            else:
                line = trial
        if line:
            result.append(line)
        return result or [""]

    def _card(self, rect: Any, title: str) -> None:
        self.pg.draw.rect(self.screen, self.PANEL, rect, border_radius=12)
        self.pg.draw.rect(self.screen, self.LINE, rect, width=1, border_radius=12)
        self._text(title, (rect.x + 16, rect.y + 12), "md")

    def _button(
        self, rect: Any, label: str, action: tuple[str, Any], enabled: bool = True, color=None
    ) -> None:
        fill = color or self.PANEL_2
        if not enabled:
            fill = (38, 44, 56)
        elif rect.collidepoint(self.pg.mouse.get_pos()):
            fill = tuple(min(channel + 18, 255) for channel in fill)
        self.pg.draw.rect(self.screen, fill, rect, border_radius=8)
        self.pg.draw.rect(self.screen, self.LINE, rect, width=1, border_radius=8)
        label_surface = self.fonts["sm"].render(
            label, True, self.TEXT if enabled else (103, 114, 132)
        )
        self.screen.blit(label_surface, label_surface.get_rect(center=rect.center))
        if enabled:
            self.click_targets.append((rect, action))

    def _poll_messages(self) -> None:
        while True:
            try:
                kind, payload = self.controller.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "preflight":
                code, document, merged = payload
                self.preflight_code = code
                self.preflight = document
                self.preflight_time = time.monotonic()
                blockers = preflight_blockers(document)
                if code == 0 and document and document.get("ready"):
                    self.notice = "预检通过：两只脚将分别使用 hci0 / hci1，可开始所选阶段。"
                    self.notice_color = self.GREEN
                else:
                    self.notice = "预检阻止采集：" + "；".join(blockers)
                    self.notice_color = self.RED
                if not document:
                    self.logs.extend(merged.strip().splitlines()[-4:])
            elif kind == "preflight_error":
                self.preflight = None
                self.preflight_code = 2
                self.notice = f"预检执行失败：{payload}"
                self.notice_color = self.RED
            elif kind == "log":
                self.logs.append(str(payload))
                self.logs = self.logs[-12:]
            elif kind == "capture_done":
                return_code, session = payload
                self.controller.clear_finished_capture()
                self.live_sample = None
                self._record_operator_event(
                    "capture_done", session=Path(session), details={"return_code": return_code}
                )
                manifest_path = Path(session) / "manifest.json"
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    self.last_result = assess_capture_quality(manifest)
                except (OSError, json.JSONDecodeError) as error:
                    self.last_result = ("FAIL", [f"无法读取 manifest：{error}"])
                grade, reasons = self.last_result
                self.phase_status[self.phase_index] = grade
                if return_code == 0 and grade in {"PASS", "REVIEW"}:
                    self.notice = (
                        f"本阶段已封存（{grade}）。现在可以卸力/调整；开始下一阶段前重新确认安全项。"
                    )
                    self.notice_color = self.GREEN if grade == "PASS" else self.YELLOW
                else:
                    self.notice = (
                        f"本阶段未通过（进程码 {return_code}, {grade}），数据保留供排查；"
                        "确认人员和设备安全后再卸力。"
                    )
                    self.notice_color = self.RED
                self.checks = {key: False for key in SAFETY_LABELS}
                self.controller.request_preflight()
                if self.close_requested:
                    self.running = False

    def _read_health(self) -> None:
        now = time.monotonic()
        if now - self.last_health_read < 0.2:
            return
        self.last_health_read = now
        session = self.controller.capture_session
        if not session:
            return
        path = session / "health.json"
        try:
            self.health = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    def _read_live_hall(self) -> None:
        if not self._capture_running():
            return
        try:
            sample = read_capture_packet(CAPTURE_IPC)
        except (OSError, ValueError):
            return
        if sample.sequence == self.last_capture_sequence:
            return
        self.last_capture_sequence = sample.sequence
        self.live_sample = sample
        self.hall_display.update(
            sample.magnetic,
            sample.valid,
            sample_time_s=sample.publish_monotonic_ns * 1.0e-9,
        )

    def _record_operator_event(
        self,
        event: str,
        *,
        session: Path | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not self.batch_id:
            return
        path = self.args.output_root.resolve() / self.batch_id / "ui_operator_events.jsonl"
        document = {
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "event": event,
            "phase_index": self.phase_index,
            "phase_key": PHASES[self.phase_index].key,
            "phase_title": PHASES[self.phase_index].title,
            "session": None if session is None else str(session),
            "details": details or {},
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(document, ensure_ascii=False) + "\n")
                stream.flush()
        except OSError as error:
            self.logs.append(f"[UI] operator event log failed: {error}")

    def _capture_running(self) -> bool:
        return self.controller.capture is not None and self.controller.capture.poll() is None

    def _ready_to_start(self) -> bool:
        phase = PHASES[self.phase_index]
        fresh_preflight = time.monotonic() - self.preflight_time < 15.0
        return bool(
            not self._capture_running()
            and self.preflight
            and self.preflight.get("ready")
            and self.preflight_code == 0
            and fresh_preflight
            and required_checks_complete(phase, self.checks)
        )

    def _start_capture(self) -> None:
        if not self._ready_to_start():
            self.notice = "尚不能开始：请刷新预检并完成当前阶段全部安全确认。"
            self.notice_color = self.RED
            return
        if not self.batch_id:
            self.batch_id = "batch_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        phase = PHASES[self.phase_index]
        try:
            session = self.controller.start_capture(phase, self.note, self.batch_id)
        except Exception as error:
            self.notice = f"启动失败：{error}"
            self.notice_color = self.RED
            return
        self.health = None
        self.live_sample = None
        self.last_capture_sequence = -1
        self.last_result = None
        self.phase_status[self.phase_index] = "active"
        if phase.key == "suspended_unloaded" or (
            not self.hall_display.baseline_ready and "unloaded" in phase.safety
        ):
            self.hall_display.begin_unloaded_baseline()
        self.notice = f"正在采集 {phase.title}；结束前请保持当前操作，不要提前卸力。"
        self.notice_color = self.CYAN
        self.logs.append(f"[UI] session={session}")
        self._record_operator_event("capture_start", session=session)

    def _change_phase(self, delta: int) -> None:
        if self._capture_running():
            return
        self.phase_index = max(0, min(len(PHASES) - 1, self.phase_index + delta))
        self.checks = {key: False for key in SAFETY_LABELS}
        self.last_result = None
        self.notice = "阶段已切换；请阅读操作说明并重新完成安全确认。"
        self.notice_color = self.YELLOW

    def _handle_action(self, action: tuple[str, Any]) -> None:
        name, value = action
        if name == "preflight":
            if self.controller.request_preflight():
                self.notice = "正在重新执行安全预检……"
                self.notice_color = self.YELLOW
        elif name == "phase":
            self._change_phase(int(value))
        elif name == "check":
            key = str(value)
            self.checks[key] = not self.checks[key]
        elif name == "start":
            self._start_capture()
        elif name == "stop":
            if self.controller.stop_capture():
                self._record_operator_event(
                    "safe_stop_requested", session=self.controller.capture_session
                )
                self.notice = "已请求安全停止，正在封存文件；未看到完成通知前不要卸力。"
                self.notice_color = self.YELLOW
        elif name == "viz_baseline":
            phase = PHASES[self.phase_index]
            allowed = (
                self._capture_running()
                and "unloaded" in phase.safety
                and bool(self.checks.get("unloaded", False))
            )
            if allowed:
                self.hall_display.begin_unloaded_baseline()
                self._record_operator_event(
                    "display_baseline_reset", session=self.controller.capture_session
                )
                self.notice = "正在重新采集显示专用无载基线；请继续保持完全卸力。"
                self.notice_color = self.YELLOW
            else:
                self.notice = "显示基线只能在采集中、且当前阶段已明确确认完全卸力时重置。"
                self.notice_color = self.RED
        elif name == "skip":
            if self.phase_status[self.phase_index] == "pending":
                self.phase_status[self.phase_index] = "skipped"
            self._change_phase(1)

    def _events(self) -> None:
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                if self._capture_running():
                    self.close_requested = True
                    self._record_operator_event(
                        "window_close_stop_requested",
                        session=self.controller.capture_session,
                    )
                    self.controller.stop_capture()
                    self.notice = "正在安全停止并封存，完成前窗口不会退出。"
                    self.notice_color = self.YELLOW
                else:
                    self.running = False
            elif event.type == self.pg.MOUSEBUTTONDOWN and event.button == 1:
                self.note_active = False
                for rect, action in reversed(self.click_targets):
                    if rect.collidepoint(event.pos):
                        if action[0] == "note":
                            self.note_active = True
                        else:
                            self._handle_action(action)
                        break
            elif event.type == self.pg.KEYDOWN and self.note_active:
                if event.key == self.pg.K_BACKSPACE:
                    self.note = self.note[:-1]
                elif event.key in (self.pg.K_RETURN, self.pg.K_ESCAPE):
                    self.note_active = False
                elif event.unicode and event.unicode.isprintable() and len(self.note) < 140:
                    self.note += event.unicode

    def _draw_header(self, width: int) -> None:
        self._text("G1 双足 Hall 数据采集控制台", (24, 17), "xl")
        self._text(
            f"远端主机 {socket.gethostname()}  ·  {self.project_dir}",
            (25, 59),
            "xs",
            self.MUTED,
        )
        boundary = "严格测量边界：仅 Bx / By / Bz / 温度与时间信息；不显示、不反演力/压力/接触力/摩擦系数"
        box = self.pg.Rect(24, 87, width - 48, 38)
        self.pg.draw.rect(self.screen, (50, 34, 19), box, border_radius=8)
        self._text(boundary, (box.x + 14, box.y + 8), "sm", self.YELLOW)

    def _draw_preflight(self, rect: Any) -> None:
        self._card(rect, "连接与采集门禁")
        document = self.preflight or {}
        adapters = document.get("adapters", {})
        y = rect.y + 48
        for side, label, address in (
            ("left", "LEFT", LEFT_ADDRESS),
            ("right", "RIGHT", RIGHT_ADDRESS),
        ):
            adapter = REQUIRED_ADAPTERS[side]
            exists = adapter in adapters
            color = self.GREEN if exists else self.RED
            self._text(f"{label}  P00–P14", (rect.x + 16, y), "sm", color)
            self._text(f"{address}  →  {adapter}", (rect.x + 160, y), "xs", self.TEXT)
            self._text(
                adapters.get(adapter, "MISSING"), (rect.x + 390, y), "xs", color
            )
            y += 29
        y += 3
        blockers = preflight_blockers(self.preflight)
        if document.get("ready"):
            self._text("READY：适配器独立且无 BLE 抢占", (rect.x + 16, y), "sm", self.GREEN)
            y += 28
        else:
            for blocker in blockers[:3]:
                for line in self._wrap("阻塞：" + blocker, rect.w - 32, "xs"):
                    self._text(line, (rect.x + 16, y), "xs", self.RED)
                    y += 21
        self._button(
            self.pg.Rect(rect.x + 16, rect.bottom - 48, 150, 34),
            "刷新安全预检",
            ("preflight", None),
            not self.controller.preflight_running and not self._capture_running(),
        )
        status = "检查中…" if self.controller.preflight_running else (
            "通过" if document.get("ready") else "未通过"
        )
        self._text(f"状态：{status}", (rect.x + 180, rect.bottom - 42), "xs", self.MUTED)

    def _draw_phase(self, rect: Any) -> None:
        phase = PHASES[self.phase_index]
        self._card(rect, "采集阶段与操作")
        self._text(
            f"{phase.title}  ·  {phase.group}  ·  {phase.duration_s} 秒（双脚 fresh 后计时）",
            (rect.x + 16, rect.y + 50),
            "md",
            self.CYAN,
        )
        y = rect.y + 84
        for line in self._wrap(phase.instruction, rect.w - 32, "sm"):
            self._text(line, (rect.x + 16, y), "sm")
            y += 27
        y += 8
        self._text("本阶段必须逐项确认：", (rect.x + 16, y), "sm", self.MUTED)
        y += 30
        for key in phase.safety:
            checked = self.checks[key]
            check_rect = self.pg.Rect(rect.x + 18, y, 22, 22)
            self.pg.draw.rect(
                self.screen, self.GREEN if checked else self.PANEL_2, check_rect, border_radius=4
            )
            self.pg.draw.rect(self.screen, self.LINE, check_rect, width=1, border_radius=4)
            if checked:
                self._text("✓", (check_rect.x + 2, check_rect.y - 4), "md", self.BG)
            self._text(SAFETY_LABELS[key], (rect.x + 50, y - 1), "sm")
            self.click_targets.append((self.pg.Rect(rect.x + 12, y - 3, rect.w - 24, 28), ("check", key)))
            y += 30

        note_label_y = rect.bottom - 115
        self._text("本批次备注：", (rect.x + 16, note_label_y), "xs", self.MUTED)
        note_rect = self.pg.Rect(rect.x + 16, note_label_y + 23, rect.w - 32, 38)
        self.pg.draw.rect(self.screen, (13, 21, 36), note_rect, border_radius=7)
        self.pg.draw.rect(
            self.screen, self.CYAN if self.note_active else self.LINE, note_rect, width=1, border_radius=7
        )
        shown = self.note
        while self.fonts["xs"].size(shown)[0] > note_rect.w - 18 and shown:
            shown = shown[1:]
        self._text(shown, (note_rect.x + 9, note_rect.y + 8), "xs")
        self.click_targets.append((note_rect, ("note", None)))

        buttons_y = rect.bottom - 44
        self._button(self.pg.Rect(rect.x + 16, buttons_y, 90, 32), "上一阶段", ("phase", -1), self.phase_index > 0 and not self._capture_running())
        self._button(self.pg.Rect(rect.x + 116, buttons_y, 90, 32), "跳过/下一项", ("skip", None), not self._capture_running())
        start_color = (13, 112, 87) if self._ready_to_start() else None
        self._button(self.pg.Rect(rect.right - 226, buttons_y, 100, 32), "开始采集", ("start", None), self._ready_to_start(), start_color)
        self._button(self.pg.Rect(rect.right - 116, buttons_y, 100, 32), "安全停止", ("stop", None), self._capture_running(), (153, 55, 55))

    def _viz_geometry(self, side: str, width: int, height: int) -> tuple[Any, ...]:
        key = (side, width, height)
        cached = self.viz_geometry_cache.get(key)
        if cached is not None:
            return cached
        sensor_uv = self.hall_layout.sensor_uv.astype(np.float32).copy()
        outline_uv = self.hall_layout.outline_uv.astype(np.float32).copy()
        if side == "left":
            sensor_uv[:, 0] = 1.0 - sensor_uv[:, 0]
            outline_uv[:, 0] = 1.0 - outline_uv[:, 0]
        yy, xx = np.mgrid[0:height, 0:width]
        grid_u = xx.astype(np.float32) / max(width - 1, 1)
        grid_v = yy.astype(np.float32) / max(height - 1, 1)
        weights = []
        sigma = 0.135
        for u, v in sensor_uv:
            distance2 = (grid_u - u) ** 2 + (grid_v - v) ** 2
            weights.append(np.exp(-distance2 / (2.0 * sigma * sigma)))
        weight_stack = np.stack(weights, axis=-1)
        weight_stack /= np.maximum(np.sum(weight_stack, axis=-1, keepdims=True), 1.0e-7)

        mask_surface = self.pg.Surface((width, height))
        mask_surface.fill((0, 0, 0))
        outline_px = [
            (round(float(u) * (width - 1)), round(float(v) * (height - 1)))
            for u, v in outline_uv
        ]
        self.pg.draw.polygon(mask_surface, (255, 255, 255), outline_px)
        mask = self.pg.surfarray.array2d(mask_surface).T != 0
        sensor_px = [
            (round(float(u) * (width - 1)), round(float(v) * (height - 1)))
            for u, v in sensor_uv
        ]
        cached = (weight_stack, mask, outline_px, sensor_px)
        self.viz_geometry_cache[key] = cached
        return cached

    def _hall_heat_surface(self, side: str, width: int, height: int) -> Any:
        weights, mask, _outline_px, _sensor_px = self._viz_geometry(side, width, height)
        values = self.hall_display.feet[side].intensity
        field = np.sum(weights * values.reshape(1, 1, -1), axis=-1)
        field = np.clip(field / max(self.hall_display.shared_scale_counts, 1.0), 0.0, 1.0) ** 0.72
        field *= mask
        stops = np.asarray([0.0, 0.12, 0.34, 0.58, 0.80, 1.0])
        colors = np.asarray(
            [
                (28, 39, 57),
                (22, 79, 108),
                (25, 153, 164),
                (242, 203, 73),
                (239, 118, 55),
                (220, 45, 67),
            ],
            dtype=np.float32,
        )
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        for channel in range(3):
            rgb[:, :, channel] = np.interp(field, stops, colors[:, channel]).astype(np.uint8)
        rgb[~mask] = self.PANEL
        surface = self.pg.Surface((width, height))
        self.pg.surfarray.pixels3d(surface)[:, :, :] = np.transpose(rgb, (1, 0, 2))
        return surface

    def _draw_foot_viz(self, side: str, rect: Any, foot_health: dict[str, Any]) -> None:
        filter_state = self.hall_display.feet[side]
        surface = self._hall_heat_surface(side, rect.w, rect.h)
        self.screen.blit(surface, rect.topleft)
        _weights, _mask, outline, sensors = self._viz_geometry(side, rect.w, rect.h)
        outline_screen = [(rect.x + x, rect.y + y) for x, y in outline]
        self.pg.draw.lines(self.screen, self.CYAN, True, outline_screen, 2)

        for index, ((px, py), vector) in enumerate(zip(sensors, filter_state.filtered)):
            x, y = rect.x + px, rect.y + py
            magnitude = float(np.linalg.norm(vector))
            strength = float(np.clip(magnitude / max(self.hall_display.shared_scale_counts, 1.0), 0.0, 1.0))
            color = (
                round(70 + 185 * strength),
                round(225 - 115 * strength),
                round(210 - 155 * strength),
            )
            self.pg.draw.circle(self.screen, (240, 245, 249), (x, y), 4)
            self.pg.draw.circle(self.screen, color, (x, y), 3)
            # Bx/By are shown as magnetic-axis change only, never mechanical direction.
            dx = float(np.clip(vector[0] / max(self.hall_display.shared_scale_counts, 1.0), -1.0, 1.0)) * 11.0
            dy = -float(np.clip(vector[1] / max(self.hall_display.shared_scale_counts, 1.0), -1.0, 1.0)) * 11.0
            if abs(dx) + abs(dy) > 2.0:
                self.pg.draw.line(self.screen, color, (x, y), (round(x + dx), round(y + dy)), 1)
            self._text(f"{index:02d}", (x + 4, y - 7), "tiny", self.TEXT)

        connected = bool(foot_health.get("connected", False))
        fresh = bool(foot_health.get("fresh", False))
        if not (self._capture_running() and self.live_sample is not None and connected and fresh):
            overlay = self.pg.Surface(rect.size, self.pg.SRCALPHA)
            overlay.fill((7, 11, 19, 155))
            self.screen.blit(overlay, rect.topleft)
            label = "等待 F0R1" if self._capture_running() else "未采集"
            text_surface = self.fonts["xs"].render(label, True, self.MUTED)
            self.screen.blit(text_surface, text_surface.get_rect(center=rect.center))
        elif filter_state.calibrating:
            progress = filter_state.calibration_progress
            box = self.pg.Rect(rect.x + 8, rect.centery - 23, rect.w - 16, 46)
            self.pg.draw.rect(self.screen, (8, 14, 23), box, border_radius=6)
            if filter_state.calibration_status == "unstable":
                baseline_label = (
                    f"漂移 {filter_state.calibration_drift_p95:.1f}/"
                    f"{filter_state.calibration_drift_max:.1f} c/s"
                )
            else:
                baseline_label = "等待无载稳定"
            self._text(baseline_label, (box.x + 8, box.y + 4), "tiny", self.YELLOW)
            track = self.pg.Rect(box.x + 8, box.y + 26, box.w - 16, 8)
            self.pg.draw.rect(self.screen, self.LINE, track, border_radius=4)
            fill = track.copy()
            fill.w = round(track.w * progress)
            self.pg.draw.rect(self.screen, self.YELLOW, fill, border_radius=4)
        elif filter_state.baseline is None:
            overlay = self.pg.Surface(rect.size, self.pg.SRCALPHA)
            overlay.fill((7, 11, 19, 170))
            self.screen.blit(overlay, rect.topleft)
            for line_index, text in enumerate(("无显示基线", "先采无载阶段")):
                text_surface = self.fonts["tiny"].render(text, True, self.YELLOW)
                center = (rect.centerx, rect.centery - 9 + line_index * 18)
                self.screen.blit(text_surface, text_surface.get_rect(center=center))

    def _draw_live(self, rect: Any) -> None:
        self._card(rect, "实时双足 Hall 变化可视化（同一色标，不是压力图）")
        phase = PHASES[self.phase_index]
        baseline_allowed = bool(
            self._capture_running()
            and "unloaded" in phase.safety
            and self.checks.get("unloaded", False)
        )
        self._button(
            self.pg.Rect(rect.right - 175, rect.y + 11, 158, 30),
            "无载重置显示基线",
            ("viz_baseline", None),
            baseline_allowed,
        )
        health_feet = (self.health or {}).get("feet", {})
        col_w = (rect.w - 48) // 2
        map_h = min(238, rect.h - 101)
        map_w = max(78, round(map_h * 0.38))
        y0 = rect.y + 53
        for index, side in enumerate(("left", "right")):
            col_x = rect.x + 16 + index * (col_w + 16)
            foot_health = health_feet.get(side, {})
            map_rect = self.pg.Rect(col_x + 4, y0, map_w, map_h)
            self._draw_foot_viz(side, map_rect, foot_health)
            info_x = map_rect.right + 14
            connected = bool(foot_health.get("connected", False))
            fresh = bool(foot_health.get("fresh", False))
            active = connected and fresh
            self._text(side.upper(), (info_x, y0), "md", self.GREEN if active else self.RED)
            peak = float(np.max(self.hall_display.feet[side].intensity))
            values = (
                f"{foot_health.get('adapter', REQUIRED_ADAPTERS[side])}  BLE/fresh {int(connected)}/{int(fresh)}",
                f"rate {float(foot_health.get('sample_rate_hz', 0) or 0):.1f} Hz",
                f"age {foot_health.get('age_s', '--')} s",
                f"frames {int(foot_health.get('frames', 0) or 0)}",
                f"rejected {int(foot_health.get('rejected_frames', 0) or 0)}",
                f"ΔB peak {peak:.0f} counts",
                f"temp min {foot_health.get('temperature_c_min', '--')} °C",
                "P00–P14 独立顺序",
            )
            for line_index, line in enumerate(values):
                self._text(line, (info_x, y0 + 34 + line_index * 22), "xs", self.TEXT)

        footer_y = rect.bottom - 32
        if self.hall_display.baseline_ready:
            baseline_text = "固定无载显示基线 LOCKED（承载期间不漂移）"
        else:
            details = []
            for side in ("left", "right"):
                foot = self.hall_display.feet[side]
                if foot.calibration_status == "unstable":
                    details.append(
                        f"{side[0].upper()}漂移P95/max "
                        f"{foot.calibration_drift_p95:.1f}/{foot.calibration_drift_max:.1f} c/s"
                    )
            baseline_text = (
                "；".join(details)
                if details
                else f"等待无载稳定 {self.hall_display.calibration_progress * 100:.0f}%"
            )
        self._text(baseline_text, (rect.x + 16, footer_y), "xs", self.YELLOW if not self.hall_display.baseline_ready else self.GREEN)
        self._text(
            f"共享色标上限 {self.hall_display.shared_scale_counts:.0f} ΔB counts",
            (rect.centerx + 25, footer_y),
            "xs",
            self.MUTED,
        )

    def _draw_notice_logs(self, rect: Any) -> None:
        self._card(rect, "操作状态与封存通知")
        y = rect.y + 48
        for line in self._wrap(self.notice, rect.w - 32, "sm")[:3]:
            self._text(line, (rect.x + 16, y), "sm", self.notice_color)
            y += 27
        y += 5
        if self.logs:
            self._text("最近日志：", (rect.x + 16, y), "xs", self.MUTED)
            y += 22
            for line in self.logs[-4:]:
                clean = line if len(line) <= 105 else line[:102] + "..."
                self._text(clean, (rect.x + 16, y), "xs", self.MUTED)
                y += 19
        else:
            self._text(
                "采集结束必须看到“本阶段已封存，现在可以卸力/调整”后再改变载荷。",
                (rect.x + 16, y),
                "xs",
                self.MUTED,
            )
            y += 24

        list_top = max(y + 10, rect.y + 185)
        self._text("完整采集清单：", (rect.x + 16, list_top), "xs", self.MUTED)
        list_top += 24
        column_width = (rect.w - 48) // 2
        rows_per_column = (len(PHASES) + 1) // 2
        status_labels = {
            "pending": "○",
            "active": "●",
            "skipped": "—",
            "PASS": "✓",
            "REVIEW": "!",
            "FAIL": "×",
        }
        status_colors = {
            "pending": self.MUTED,
            "active": self.CYAN,
            "skipped": self.MUTED,
            "PASS": self.GREEN,
            "REVIEW": self.YELLOW,
            "FAIL": self.RED,
        }
        for index, phase in enumerate(PHASES):
            column = index // rows_per_column
            row = index % rows_per_column
            x = rect.x + 16 + column * (column_width + 16)
            item_y = list_top + row * 21
            status = self.phase_status[index]
            color = status_colors[status]
            current = " ▶" if index == self.phase_index else ""
            label = f"{status_labels[status]} {phase.title}{current}"
            self._text(label, (x, item_y), "xs", color)

    def draw(self) -> None:
        self.screen.fill(self.BG)
        self.click_targets = []
        width, height = self.screen.get_size()
        self._draw_header(width)
        gap = 14
        left_w = max(430, int(width * 0.38))
        right_x = 24 + left_w + gap
        right_w = width - right_x - 24
        top = 139
        self._draw_preflight(self.pg.Rect(24, top, left_w, 235))
        self._draw_phase(self.pg.Rect(24, top + 235 + gap, left_w, height - top - 235 - gap - 18))
        live_h = 360
        self._draw_live(self.pg.Rect(right_x, top, right_w, live_h))
        self._draw_notice_logs(self.pg.Rect(right_x, top + live_h + gap, right_w, height - top - live_h - gap - 18))
        phase_counter = f"阶段 {self.phase_index + 1}/{len(PHASES)}"
        self._text(phase_counter, (width - 115, 62), "xs", self.MUTED)
        self.pg.display.flip()

    def run(self) -> int:
        while self.running:
            self._events()
            self._poll_messages()
            self._read_health()
            self._read_live_hall()
            self.draw()
            self.clock.tick(30)
        self.pg.key.stop_text_input()
        self.pg.quit()
        return 0


def main() -> int:
    args = parse_args()
    args.config = args.config.resolve()
    args.output_root = args.output_root.resolve()
    try:
        return HallCaptureUI(args).run()
    except ImportError as error:
        print(f"[ERROR] UI dependency missing: {error}", file=sys.stderr)
        print("Install project requirements in .venv before launching the UI.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
