"""Remove common LLM meta / closing phrases from model output."""
from __future__ import annotations

import re
from typing import Optional

# Trailing blocks (English + Chinese) often added despite instructions
_PATTERNS = [
    re.compile(r"(?is)\n{1,2}let me know[\s\S]{0,800}\Z"),
    re.compile(r"(?is)\n{1,2}feel free to[\s\S]{0,800}\Z"),
    re.compile(r"(?is)\n{1,2}if you (?:need|have|would like)[\s\S]{0,800}\Z"),
    re.compile(r"(?is)\n{1,2}(?:happy to|please let me know|don't hesitate)[\s\S]{0,800}\Z"),
    re.compile(r"(?is)\n{1,2}(?:hope this helps|thanks for reading)[\s\S]{0,400}\Z"),
    re.compile(r"(?is)\n{1,2}(?:请告诉|如有需要|如需|欢迎反馈|希望对你|以上(?:内容)?)[\s\S]{0,800}\Z"),
    # LLM rejection / no-op responses — match anywhere in text, not just at start
    re.compile(r"(?is)(?:没有提供|未提供|抱歉.{0,10}无法|您提供的.{0,30}仅包含).*(?:转录文本|原始文本|音频.{0,5}内容|音频转写|有效内容).*(?:无法|不能|请提供).*"),
    re.compile(r"(?is)(?:请提供|请上传|请检查|请您提供).*(?:转录文本|原始文本|音频文件|完整的).*(?:优化|格式化|整理).*"),
    re.compile(r"(?is)(?:I cannot|I can't|There is no|No audio|No transcript).*[\s\S]{0,500}\Z"),
    # Catch: the output is entirely meta-talk (LLM explaining what it needs instead of processing)
    re.compile(r"(?is)^.*(?:似乎仅包含|仅包含.{0,10}元数据|未包含实际|不包含.{0,5}内容).*(?:请提供|无法|不能).*$"),
    # Catch: "您好，您提供的...只包含了...并没有...请将...粘贴出来"
    re.compile(r"(?is)(?:您好.{0,20})?(?:您提供的.{0,50})(?:只包含|仅包含|并没有).*(?:转录文[本件]|音频).*(?:粘贴出来|提供|无法|不能).*(?:优化|格式化|整理).*"),
]


def strip_llm_artifacts(text: Optional[str]) -> str:
    if not text or not isinstance(text, str):
        return (text or "").strip()
    t = text.strip()
    for _ in range(6):
        before = t
        for pat in _PATTERNS:
            t = pat.sub("", t).strip()
        if t == before:
            break
    lines = t.split("\n")
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        low = last.lower()
        if len(last) < 200 and any(
            x in low
            for x in (
                "let me know",
                "further adjustments",
                "feel free",
                "hope this helps",
                "请告诉我",
                "如需调整",
                "欢迎反馈",
            )
        ):
            lines.pop()
            continue
        break
    return "\n".join(lines).strip()
