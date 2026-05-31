import os
import re
import shutil
import uuid
import asyncio
import subprocess
import json
import urllib.request
import yt_dlp
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class VideoProcessor:
    """视频处理器，使用yt-dlp下载和转换视频"""
    
    def __init__(self):
        self.ydl_opts = {
            'format': 'bestaudio/best',  # 优先下载最佳音频源
            'outtmpl': '%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                # 直接在提取阶段转换为单声道 16k（空间小且稳定）
                'preferredcodec': 'm4a',
                'preferredquality': '192'
            }],
            # 全局FFmpeg参数：单声道 + 16k 采样率 + faststart
            'postprocessor_args': ['-ac', '1', '-ar', '16000', '-movflags', '+faststart'],
            'prefer_ffmpeg': True,
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,  # 强制只下载单个视频，不下载播放列表
        }

    async def normalize_local_media_to_m4a(self, input_path: Path, output_dir: Path) -> str:
        """
        将本地上传的音视频转为单声道 16kHz AAC m4a，供 Faster-Whisper 使用（与 yt-dlp 后处理参数对齐）。
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        unique_id = str(uuid.uuid4())[:8]
        out_path = output_dir / f"upload_norm_{unique_id}.m4a"

        cmd = [
            "ffmpeg", "-y", "-nostdin", "-i", str(input_path.resolve()),
            "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            str(out_path.resolve()),
        ]

        def _run():
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "").strip()
                raise Exception(f"FFmpeg 转换失败: {err[:800]}")
            if not out_path.exists():
                raise Exception("FFmpeg 未生成输出文件")

        await asyncio.to_thread(_run)
        return str(out_path)
    
    async def fetch_subtitles(self, url: str, output_dir: Path, cookies_file: Optional[str] = None) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        先尝试从平台获取字幕文本，比下载音频快得多。

        Returns:
            (subtitle_markdown, video_title, language_code)
            subtitle_markdown 为 None 表示无可用字幕。
        """
        import asyncio

        output_dir.mkdir(exist_ok=True)
        unique_id = str(uuid.uuid4())[:8]
        sub_dir = output_dir / f"subs_{unique_id}"

        try:
            # 1. 快速探测：获取视频信息和字幕可用性，不下载任何内容
            check_opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
            if cookies_file:
                check_opts["cookiefile"] = cookies_file
            with yt_dlp.YoutubeDL(check_opts) as ydl:
                info = await asyncio.to_thread(ydl.extract_info, url, False)

            video_title = info.get("title", "unknown")
            manual_subs: dict = info.get("subtitles") or {}
            auto_caps: dict = info.get("automatic_captions") or {}

            # 过滤掉 live_chat、danmaku 等非语音轨道
            _non_speech = ("live_chat", "danmaku")
            manual_langs = [k for k in manual_subs if not k.startswith(_non_speech)]
            auto_langs = [k for k in auto_caps if not k.startswith(_non_speech)]

            if not manual_langs and not auto_langs:
                # Bilibili fallback: try direct API for AI subtitles
                if "bilibili.com" in url:
                    logger.info(f"yt-dlp 无字幕，尝试 Bilibili AI 字幕 API")
                    bili_result = await self._fetch_bilibili_ai_subtitles(url, output_dir, cookies_file)
                    if bili_result[0]:
                        return bili_result
                logger.info(f"视频无可用字幕: {url}")
                return None, video_title, None

            # 优先手动字幕，其次自动字幕
            prefer_manual = bool(manual_langs)
            candidate_langs = manual_langs if prefer_manual else auto_langs

            # 按优先级选语言，子串匹配兼容 Bilibili 的 zh-CN、ai-zh 等代码
            _priority = ["en", "en-orig", "zh-Hans", "zh-Hant", "zh", "ja", "ko", "fr", "de", "es"]
            prefer_lang = next(
                (c for p in _priority for c in candidate_langs if p.lower() in c.lower()),
                candidate_langs[0],
            )
            logger.info(
                f"发现{'手动' if prefer_manual else '自动'}字幕，选用语言: {prefer_lang}"
                f"（候选 {len(candidate_langs)} 种）"
            )

            # 2. 仅下载字幕，跳过音视频
            sub_dir.mkdir(exist_ok=True)
            dl_opts = {
                "writesubtitles": prefer_manual,
                "writeautomaticsub": not prefer_manual,
                "subtitlesformat": "vtt/srt/best",
                "subtitleslangs": [prefer_lang],
                "skip_download": True,
                "outtmpl": str(sub_dir / "sub.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
            }
            if cookies_file:
                dl_opts["cookiefile"] = cookies_file
            with yt_dlp.YoutubeDL(dl_opts) as ydl:
                await asyncio.to_thread(ydl.download, [url])

            # 3. 查找下载的字幕文件
            sub_files = list(sub_dir.glob("*.vtt")) + list(sub_dir.glob("*.srt"))
            if not sub_files:
                if "bilibili.com" in url:
                    logger.info("yt-dlp 字幕文件未找到，尝试 Bilibili AI 字幕 API")
                    bili_result = await self._fetch_bilibili_ai_subtitles(url, output_dir, cookies_file)
                    if bili_result[0]:
                        return bili_result
                logger.warning("字幕下载后未找到文件，回退音频模式")
                return None, video_title, None

            sub_file = sub_files[0]

            # 从文件名提取语言代码 (e.g. sub.en.vtt → en)
            stem_parts = sub_file.stem.split(".")
            file_lang = stem_parts[-1] if len(stem_parts) > 1 else prefer_lang

            # 4. 解析字幕文件
            if sub_file.suffix == ".vtt":
                entries = self._parse_vtt(str(sub_file))
            else:
                entries = self._parse_srt(str(sub_file))

            if not entries:
                logger.warning("字幕解析结果为空，回退音频模式")
                return None, video_title, None

            # 5. 格式化为与 Whisper 输出兼容的 Markdown
            formatted = self._format_subtitle_entries(entries, file_lang)
            logger.info(f"字幕获取成功: lang={file_lang}, {len(entries)} 条目")
            return formatted, video_title, file_lang

        except Exception as e:
            logger.warning(f"字幕获取失败（将回退至音频下载）: {e}")
            return None, None, None
        finally:
            if sub_dir.exists():
                try:
                    shutil.rmtree(str(sub_dir))
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Bilibili AI 字幕直接 API 获取
    # ------------------------------------------------------------------

    async def _fetch_bilibili_ai_subtitles(self, url: str, output_dir: Path, cookies_file: Optional[str] = None) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """绕过 yt-dlp，直接调用 B站 API 获取 AI 字幕（需要登录 cookie）。"""
        import re as _re
        try:
            # 提取 BV 号
            bv_match = _re.search(r"BV[a-zA-Z0-9]+", url)
            if not bv_match:
                return None, None, None
            bvid = bv_match.group(0)

            # 先用 yt-dlp 获取 cid（不需要 cookie）
            check_opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
            with yt_dlp.YoutubeDL(check_opts) as ydl:
                info = await asyncio.to_thread(ydl.extract_info, url, False)
            video_title = info.get("title", bvid)
            cid = None
            for fmt in info.get("formats", []):
                url_part = fmt.get("url", "")
                cid_match = _re.search(r"/(\d{10,})/", url_part)
                if cid_match:
                    cid = cid_match.group(1)
                    break
            if not cid:
                logger.info("无法获取 cid，跳过 Bilibili AI 字幕")
                return None, video_title, None

            # 加载 cookies
            cookie_jar = None
            if cookies_file:
                try:
                    import http.cookiejar
                    cookie_jar = http.cookiejar.MozillaCookieJar()
                    cookie_jar.load(cookies_file, ignore_discard=True, ignore_expires=True)
                except Exception as e:
                    logger.warning(f"加载 Cookie 文件失败: {e}")

            if not cookie_jar:
                # 尝试从浏览器自动提取
                cookie_jar = self._extract_browser_cookies()

            if not cookie_jar:
                logger.info("无可用的 Bilibili Cookie，跳过 AI 字幕")
                return None, video_title, None

            # 调用 B站播放器 API
            import urllib.request as _urllib
            from http.cookiejar import CookieJar

            api_url = f"https://api.bilibili.com/x/player/wbi/v2?bvid={bvid}&cid={cid}"
            req = _urllib.Request(api_url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Referer": f"https://www.bilibili.com/video/{bvid}/",
            })
            opener = _urllib.build_opener(_urllib.HTTPCookieProcessor(cookie_jar))
            raw = await asyncio.to_thread(opener.open, req)
            data = json.loads(raw.read())

            subs = data.get("data", {}).get("subtitle", {}).get("subtitles", [])
            if not subs:
                logger.info("B站 API 返回空字幕列表")
                return None, video_title, None

            # 按优先级选语言：ai-zh > zh-CN > ai-en > en-US
            _lang_priority = ["ai-zh", "zh-CN", "zh", "ai-en", "en-US", "en"]
            selected = next(
                (s for lang in _lang_priority for s in subs if s.get("lan") == lang),
                subs[0],
            )
            selected_lang = selected["lan"]
            logger.info(f"Bilibili AI 字幕选中: {selected_lang}")

            # 下载字幕 JSON
            sub_url = selected.get("subtitle_url", "")
            if sub_url.startswith("//"):
                sub_url = "https:" + sub_url
            sub_req = _urllib.Request(sub_url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Referer": f"https://www.bilibili.com/video/{bvid}/",
            })
            sub_raw = await asyncio.to_thread(opener.open, sub_req)
            sub_data = json.loads(sub_raw.read())

            # 转为本项目格式
            entries = []
            seen = set()
            for item in sub_data.get("body", []):
                start_sec = float(item.get("from", 0))
                end_sec = float(item.get("to", 0))
                text = item.get("content", "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                start_str = f"{int(start_sec // 60):02d}:{int(start_sec % 60):02d}"
                end_str = f"{int(end_sec // 60):02d}:{int(end_sec % 60):02d}"
                entries.append({"start": start_str, "end": end_str, "text": text})

            if not entries:
                logger.warning("Bilibili AI 字幕解析为空")
                return None, video_title, None

            # 映射语言代码为友好名称
            lang_map = {
                "ai-zh": "zh", "zh-CN": "zh", "zh": "zh",
                "ai-en": "en", "en-US": "en", "en": "en",
            }
            file_lang = lang_map.get(selected_lang, selected_lang)

            formatted = self._format_subtitle_entries(entries, file_lang)
            logger.info(f"Bilibili AI 字幕获取成功: lang={selected_lang}, {len(entries)} 条目")
            return formatted, video_title, file_lang

        except Exception as e:
            logger.warning(f"Bilibili AI 字幕获取失败: {e}")
            return None, None, None

    @staticmethod
    def _extract_browser_cookies():
        """从浏览器提取 Bilibili cookies。"""
        try:
            from yt_dlp.cookies import extract_cookies_from_browser
            browsers = ["chrome", "safari", "firefox", "edge"]
            for browser in browsers:
                try:
                    cj = extract_cookies_from_browser(browser)
                    # 检查是否有 Bilibili cookie
                    has_sessdata = any(
                        "bilibili" in c.domain and c.name == "SESSDATA"
                        for c in cj
                    )
                    if has_sessdata:
                        logger.info(f"从 {browser} 提取到 Bilibili cookies")
                        return cj
                except Exception:
                    continue
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # 字幕解析辅助方法
    # ------------------------------------------------------------------

    def _parse_vtt(self, filepath: str) -> list:
        """解析 WebVTT 字幕文件，返回去重后的条目列表。

        特别处理 YouTube 自动字幕的「滚动追加」格式：
        同一句话会被分成多个 cue 逐字追加，只保留每组的「最终版本」。
        """
        raw_entries = []
        seen_texts: set = set()

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"读取 VTT 文件失败: {e}")
            return []

        # 移除 WEBVTT 文件头，按空行分割 cue 块
        content = re.sub(r"^WEBVTT[^\n]*\n", "", content)
        blocks = re.split(r"\n{2,}", content.strip())

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            lines = block.split("\n")
            timing_idx = next((i for i, l in enumerate(lines) if "-->" in l), -1)
            if timing_idx < 0:
                continue

            timing_line = lines[timing_idx]
            text_lines = lines[timing_idx + 1:]

            match = re.match(
                r"(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\s*-->\s*"
                r"(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)",
                timing_line,
            )
            if not match:
                continue

            start_str = self._normalize_time(match.group(1))
            end_str = self._normalize_time(match.group(2))

            raw_text = " ".join(text_lines)
            # 去除 HTML / VTT 内联标签（包括 YouTube 逐字时间码标签）
            text = re.sub(r"<[^>]+>", "", raw_text)
            text = (
                text.replace("&amp;", "&")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                    .replace("&nbsp;", " ")
                    .replace("&#39;", "'")
                    .replace("&quot;", '"')
                    .strip()
            )
            # 合并行内多余空白
            text = re.sub(r"\s+", " ", text).strip()

            if not text or text in seen_texts:
                continue

            seen_texts.add(text)
            raw_entries.append({"start": start_str, "end": end_str, "text": text})

        # ── 二次去重：过滤 YouTube「滚动追加」的中间状态 ──────────────────
        # 若条目 i 的文本是条目 i+1 文本的起始子串，则条目 i 是中间状态，丢弃。
        # 同时丢弃纯空白/单字符的噪音条目。
        if not raw_entries:
            return []

        entries = []
        for i, entry in enumerate(raw_entries):
            text = entry["text"]
            if len(text) < 2:
                continue
            # 检查后续若干条是否以当前文本开头（滚动追加的特征）
            is_intermediate = False
            for j in range(i + 1, min(i + 4, len(raw_entries))):
                next_text = raw_entries[j]["text"]
                if next_text.startswith(text) and len(next_text) > len(text):
                    is_intermediate = True
                    break
            if not is_intermediate:
                entries.append(entry)

        return entries

    def _parse_srt(self, filepath: str) -> list:
        """解析 SRT 字幕文件，返回去重后的条目列表。"""
        entries = []
        seen_texts: set = set()

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"读取 SRT 文件失败: {e}")
            return []

        blocks = re.split(r"\n{2,}", content.strip())

        for block in blocks:
            lines = block.strip().split("\n")
            timing_idx = next((i for i, l in enumerate(lines) if "-->" in l), -1)
            if timing_idx < 0:
                continue

            timing_line = lines[timing_idx]
            text_lines = lines[timing_idx + 1:]

            match = re.match(
                r"(\d{1,2}:\d{2}:\d{2}[.,]\d+)\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d+)",
                timing_line,
            )
            if not match:
                continue

            start_str = self._normalize_time(match.group(1))
            end_str = self._normalize_time(match.group(2))

            text = " ".join(text_lines)
            text = re.sub(r"<[^>]+>", "", text).strip()

            if not text or text in seen_texts:
                continue

            seen_texts.add(text)
            entries.append({"start": start_str, "end": end_str, "text": text})

        return entries

    def _normalize_time(self, time_str: str) -> str:
        """将 HH:MM:SS.mmm 或 MM:SS.mmm 统一转为 MM:SS 格式。"""
        time_str = re.sub(r"[.,]\d+$", "", time_str)
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{h * 60 + m:02d}:{s:02d}"
        elif len(parts) == 2:
            m, s = int(parts[0]), int(parts[1])
            return f"{m:02d}:{s:02d}"
        return time_str

    def _format_subtitle_entries(self, entries: list, language: str) -> str:
        """将字幕条目格式化为与 Whisper 输出兼容的 Markdown，供下游管道直接使用。"""
        lines = [
            "# Video Transcription",
            "",
            f"**Detected Language:** {language}",
            "**Language Probability:** 1.00",
            "",
            "## Transcription Content",
            "",
        ]
        for entry in entries:
            lines.append(f"**[{entry['start']} - {entry['end']}]**")
            lines.append("")
            lines.append(entry["text"])
            lines.append("")
        return "\n".join(lines)

    async def download_and_convert(
        self,
        url: str,
        output_dir: Path,
        prefetched_title: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        下载视频并转换为m4a格式。

        prefetched_title: 若调用方已通过 fetch_subtitles 探测过视频信息，
        可直接传入视频标题，跳过重复的 extract_info 网络请求。
        """
        try:
            # 创建输出目录
            output_dir.mkdir(exist_ok=True)
            
            # 生成唯一的文件名
            unique_id = str(uuid.uuid4())[:8]
            output_template = str(output_dir / f"audio_{unique_id}.%(ext)s")
            
            # 更新yt-dlp选项
            ydl_opts = self.ydl_opts.copy()
            ydl_opts['outtmpl'] = output_template
            
            logger.info(f"开始下载视频: {url}")
            
            import asyncio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                if prefetched_title:
                    # 标题和时长已在 fetch_subtitles 中获取，直接下载，跳过重复探测
                    video_title = prefetched_title
                    expected_duration = 0
                    logger.info(f"复用预取标题，跳过 extract_info: {video_title}")
                else:
                    # 获取视频信息（放到线程池避免阻塞事件循环）
                    info = await asyncio.to_thread(ydl.extract_info, url, False)
                    video_title = info.get('title', 'unknown')
                    expected_duration = info.get('duration') or 0
                    logger.info(f"视频标题: {video_title}")
                
                # 下载视频（放到线程池避免阻塞事件循环）
                await asyncio.to_thread(ydl.download, [url])
            
            # 查找生成的m4a文件
            audio_file = str(output_dir / f"audio_{unique_id}.m4a")
            
            if not os.path.exists(audio_file):
                # 如果m4a文件不存在，查找其他音频格式
                for ext in ['webm', 'mp4', 'mp3', 'wav']:
                    potential_file = str(output_dir / f"audio_{unique_id}.{ext}")
                    if os.path.exists(potential_file):
                        audio_file = potential_file
                        break
                else:
                    raise Exception("未找到下载的音频文件")
            
            # 校验时长，如果和源视频差异较大，尝试一次ffmpeg规范化重封装
            try:
                import subprocess, shlex
                probe_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(audio_file)}"
                out = subprocess.check_output(probe_cmd, shell=True).decode().strip()
                actual_duration = float(out) if out else 0.0
            except Exception as _:
                actual_duration = 0.0
            
            if expected_duration and actual_duration and abs(actual_duration - expected_duration) / expected_duration > 0.1:
                logger.warning(
                    f"音频时长异常，期望{expected_duration}s，实际{actual_duration}s，尝试重封装修复…"
                )
                try:
                    fixed_path = str(output_dir / f"audio_{unique_id}_fixed.m4a")
                    fix_cmd = f"ffmpeg -y -i {shlex.quote(audio_file)} -vn -c:a aac -b:a 160k -movflags +faststart {shlex.quote(fixed_path)}"
                    subprocess.check_call(fix_cmd, shell=True)
                    # 用修复后的文件替换
                    audio_file = fixed_path
                    # 重新探测
                    out2 = subprocess.check_output(probe_cmd.replace(shlex.quote(audio_file.rsplit('.',1)[0]+'.m4a'), shlex.quote(audio_file)), shell=True).decode().strip()
                    actual_duration2 = float(out2) if out2 else 0.0
                    logger.info(f"重封装完成，新时长≈{actual_duration2:.2f}s")
                except Exception as e:
                    logger.error(f"重封装失败：{e}")
            
            logger.info(f"音频文件已保存: {audio_file}")
            return audio_file, video_title
            
        except Exception as e:
            logger.error(f"下载视频失败: {str(e)}")
            raise Exception(f"下载视频失败: {str(e)}")
    
    def get_video_info(self, url: str) -> dict:
        """
        获取视频信息
        
        Args:
            url: 视频链接
            
        Returns:
            视频信息字典
        """
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', ''),
                    'upload_date': info.get('upload_date', ''),
                    'description': info.get('description', ''),
                    'view_count': info.get('view_count', 0),
                }
        except Exception as e:
            logger.error(f"获取视频信息失败: {str(e)}")
            raise Exception(f"获取视频信息失败: {str(e)}")
