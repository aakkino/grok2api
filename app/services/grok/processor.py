"""
OpenAI 响应格式处理器
"""
import time
import uuid
import random
import html
import re
import orjson
from typing import Any, AsyncGenerator, Optional, AsyncIterable, List

from app.core.config import get_config
from app.core.logger import logger
from app.services.grok.assets import DownloadService


ASSET_URL = "https://assets.grok.com/"
_GROK_RENDER_END = "</grok:render>"
_GROK_RENDER_TAG_RE = re.compile(r"<grok:render\b(?P<attrs>[^>]*)>(?P<body>.*?)</grok:render>", re.DOTALL)
_GROK_ATTR_RE = re.compile(r'([A-Za-z_:][A-Za-z0-9_:\.\-]*)\s*=\s*("([^"]*)"|\'([^\']*)\')')
_URL_TRAILING_RE = re.compile(r"(?:https?://|www\.)[^\s<>'\"`]+$", re.IGNORECASE)


def _build_video_poster_preview(video_url: str, thumbnail_url: str = "") -> str:
    """将 <video> 替换为可点击的 Poster 预览图（用于前端展示）"""
    safe_video = html.escape(video_url or "", quote=True)
    safe_thumb = html.escape(thumbnail_url or "", quote=True)

    if not safe_video:
        return ""

    if not safe_thumb:
        return f'<a href="{safe_video}" target="_blank" rel="noopener noreferrer">{safe_video}</a>'

    return f'''<a href="{safe_video}" target="_blank" rel="noopener noreferrer" style="display:inline-block;position:relative;max-width:100%;text-decoration:none;">
  <img src="{safe_thumb}" alt="video" style="max-width:100%;height:auto;border-radius:12px;display:block;" />
  <span style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;">
    <span style="width:64px;height:64px;border-radius:9999px;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;">
      <span style="width:0;height:0;border-top:12px solid transparent;border-bottom:12px solid transparent;border-left:18px solid #fff;margin-left:4px;"></span>
    </span>
  </span>
</a>'''


class BaseProcessor:
    """基础处理器"""
    
    def __init__(self, model: str, token: str = ""):
        self.model = model
        self.token = token
        self.created = int(time.time())
        self.app_url = get_config("app.app_url", "")
        self._dl_service: Optional[DownloadService] = None

    def _get_dl(self) -> DownloadService:
        """获取下载服务实例（复用）"""
        if self._dl_service is None:
            self._dl_service = DownloadService()
        return self._dl_service

    async def close(self):
        """释放下载服务资源"""
        if self._dl_service:
            await self._dl_service.close()
            self._dl_service = None

    async def process_url(self, path: str, media_type: str = "image") -> str:
        """处理资产 URL"""
        # 处理可能的绝对路径
        if path.startswith("http"):
            from urllib.parse import urlparse
            path = urlparse(path).path
            
        if not path.startswith("/"):
            path = f"/{path}"

        # Invalid root path is not a displayable image URL.
        if path in {"", "/"}:
            return ""

        # Always materialize to local cache endpoint so callers don't rely on
        # direct assets.grok.com access (often blocked without upstream cookies).
        dl_service = self._get_dl()
        await dl_service.download(path, self.token, media_type)
        local_path = f"/v1/files/{media_type}{path}"
        if self.app_url:
            return f"{self.app_url.rstrip('/')}{local_path}"
        return local_path

    @staticmethod
    def _parse_attrs(attr_text: str) -> dict[str, str]:
        attrs: dict[str, str] = {}
        for m in _GROK_ATTR_RE.finditer(attr_text or ""):
            key = (m.group(1) or "").strip()
            val = m.group(3) if m.group(3) is not None else (m.group(4) or "")
            if key:
                attrs[key] = val
        return attrs

    def _render_tag_to_citation(
        self,
        tag: str,
        citation_map: dict[str, int],
        next_index_holder: list[int],
    ) -> str:
        m = _GROK_RENDER_TAG_RE.fullmatch(tag.strip())
        if not m:
            return tag

        attrs = self._parse_attrs(m.group("attrs") or "")
        if attrs.get("card_type") != "citation_card":
            return tag
        if attrs.get("type") != "render_inline_citation":
            return tag

        body = (m.group("body") or "").strip()
        key = attrs.get("card_id") or body or f"inline-{next_index_holder[0]}"

        idx = citation_map.get(key)
        if idx is None:
            idx = next_index_holder[0]
            citation_map[key] = idx
            next_index_holder[0] += 1

        return f"[{idx}]"

    def _replace_inline_citations(
        self,
        text: str,
        citation_map: Optional[dict[str, int]] = None,
        next_index_holder: Optional[list[int]] = None,
    ) -> str:
        if not text:
            return text

        local_map = citation_map if citation_map is not None else {}
        local_next = next_index_holder if next_index_holder is not None else [1]

        out: list[str] = []
        last = 0
        for m in _GROK_RENDER_TAG_RE.finditer(text):
            out.append(text[last:m.start()])
            repl = self._render_tag_to_citation(m.group(0), local_map, local_next)
            prefix = "".join(out)
            out.append(self._separate_citation_from_url(prefix, repl))
            last = m.end()
        out.append(text[last:])
        return "".join(out)

    @staticmethod
    def _looks_like_url_suffix(prefix: str) -> bool:
        if not prefix:
            return False
        tail = prefix[-2048:]
        return bool(_URL_TRAILING_RE.search(tail))

    def _separate_citation_from_url(self, prefix: str, citation: str) -> str:
        if not citation or not citation.startswith("["):
            return citation
        if self._looks_like_url_suffix(prefix):
            return f" {citation}"
        return citation
            
    def _sse(self, content: str = "", role: str = None, finish: str = None) -> str:
        """构建 SSE 响应 (StreamProcessor 通用)"""
        if not hasattr(self, 'response_id'):
            self.response_id = None
        if not hasattr(self, 'fingerprint'):
            self.fingerprint = ""
            
        delta = {}
        if role:
            delta["role"] = role
            delta["content"] = ""
        elif content:
            delta["content"] = content
        
        chunk = {
            "id": self.response_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.fingerprint if hasattr(self, 'fingerprint') else "",
            "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]
        }
        return f"data: {orjson.dumps(chunk).decode()}\n\n"


class StreamProcessor(BaseProcessor):
    """流式响应处理器"""
    
    def __init__(self, model: str, token: str = "", think: bool = None):
        super().__init__(model, token)
        self.response_id: Optional[str] = None
        self.fingerprint: str = ""
        self.think_opened: bool = False
        self.role_sent: bool = False
        all_filter_tags = get_config("grok.filter_tags", [])
        if isinstance(all_filter_tags, str):
            all_filter_tags = [x.strip() for x in all_filter_tags.split(",") if x.strip()]
        if not isinstance(all_filter_tags, list):
            all_filter_tags = []
        self.filter_tags = [str(t) for t in all_filter_tags if str(t) and str(t) != "grok:render"]
        self.image_format = get_config("app.image_format", "url")
        self._render_pending: str = ""
        self._citation_map: dict[str, int] = {}
        self._citation_next: list[int] = [1]
        
        if think is None:
            self.show_think = get_config("grok.thinking", False)
        else:
            self.show_think = think

    def _consume_token_with_citations(self, token: str) -> str:
        if not token:
            return token

        text = f"{self._render_pending}{token}" if self._render_pending else token
        self._render_pending = ""

        out: list[str] = []
        cursor = 0
        while True:
            start = text.find("<grok:render", cursor)
            if start < 0:
                out.append(text[cursor:])
                break

            out.append(text[cursor:start])
            end = text.find(_GROK_RENDER_END, start)
            if end < 0:
                self._render_pending = text[start:]
                break

            end += len(_GROK_RENDER_END)
            tag = text[start:end]
            citation = self._render_tag_to_citation(tag, self._citation_map, self._citation_next)
            prefix = "".join(out)
            out.append(self._separate_citation_from_url(prefix, citation))
            cursor = end

        return "".join(out)
    
    async def process(self, response: AsyncIterable[bytes]) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                resp = data.get("result", {}).get("response", {})
                
                # 元数据
                if (llm := resp.get("llmInfo")) and not self.fingerprint:
                    self.fingerprint = llm.get("modelHash", "")
                if rid := resp.get("responseId"):
                    self.response_id = rid
                
                # 首次发送 role
                if not self.role_sent:
                    yield self._sse(role="assistant")
                    self.role_sent = True
                
                # 图像生成进度
                if img := resp.get("streamingImageGenerationResponse"):
                    if self.show_think:
                        if not self.think_opened:
                            yield self._sse("<think>\n")
                            self.think_opened = True
                        idx = img.get('imageIndex', 0) + 1
                        progress = img.get('progress', 0)
                        yield self._sse(f"正在生成第{idx}张图片中，当前进度{progress}%\n")
                    continue
                
                # modelResponse
                if mr := resp.get("modelResponse"):
                    if self.think_opened and self.show_think:
                        if msg := mr.get("message"):
                            yield self._sse(msg + "\n")
                        yield self._sse("</think>\n")
                        self.think_opened = False
                    
                    # 处理生成的图片
                    for url in mr.get("generatedImageUrls", []):
                        parts = url.split("/")
                        img_id = parts[-2] if len(parts) >= 2 else "image"
                        
                        if self.image_format == "base64":
                            dl_service = self._get_dl()
                            base64_data = await dl_service.to_base64(url, self.token, "image")
                            if base64_data:
                                yield self._sse(f"![{img_id}]({base64_data})\n")
                            else:
                                final_url = await self.process_url(url, "image")
                                yield self._sse(f"![{img_id}]({final_url})\n")
                        else:
                            final_url = await self.process_url(url, "image")
                            yield self._sse(f"![{img_id}]({final_url})\n")
                    
                    if (meta := mr.get("metadata", {})).get("llm_info", {}).get("modelHash"):
                        self.fingerprint = meta["llm_info"]["modelHash"]
                    continue
                
                # 普通 token
                if (token := resp.get("token")) is not None:
                    if isinstance(token, str):
                        token = self._consume_token_with_citations(token)
                    if token and not (self.filter_tags and any(t in token for t in self.filter_tags)):
                        yield self._sse(token)
                        
            if self._render_pending:
                pending = self._replace_inline_citations(
                    self._render_pending,
                    citation_map=self._citation_map,
                    next_index_holder=self._citation_next,
                )
                if pending and not (self.filter_tags and any(t in pending for t in self.filter_tags)):
                    yield self._sse(pending)
                self._render_pending = ""

            if self.think_opened:
                yield self._sse("</think>\n")
            yield self._sse(finish="stop")
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream processing error: {e}", extra={"model": self.model})
            raise
        finally:
            await self.close()


class CollectProcessor(BaseProcessor):
    """非流式响应处理器"""
    
    def __init__(self, model: str, token: str = ""):
        super().__init__(model, token)
        self.image_format = get_config("app.image_format", "url")
        all_filter_tags = get_config("grok.filter_tags", [])
        if isinstance(all_filter_tags, str):
            all_filter_tags = [x.strip() for x in all_filter_tags.split(",") if x.strip()]
        if not isinstance(all_filter_tags, list):
            all_filter_tags = []
        self.filter_tags = [str(t) for t in all_filter_tags if str(t) and str(t) != "grok:render"]
    
    async def process(self, response: AsyncIterable[bytes]) -> dict[str, Any]:
        """处理并收集完整响应"""
        response_id = ""
        fingerprint = ""
        content = ""
        
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                resp = data.get("result", {}).get("response", {})
                
                if (llm := resp.get("llmInfo")) and not fingerprint:
                    fingerprint = llm.get("modelHash", "")
                
                if mr := resp.get("modelResponse"):
                    response_id = mr.get("responseId", "")
                    content = mr.get("message", "")
                    if isinstance(content, str):
                        content = self._replace_inline_citations(content)
                        if self.filter_tags:
                            for t in self.filter_tags:
                                if t:
                                    content = content.replace(t, "")
                    
                    if urls := mr.get("generatedImageUrls"):
                        content += "\n"
                        for url in urls:
                            parts = url.split("/")
                            img_id = parts[-2] if len(parts) >= 2 else "image"
                            
                            if self.image_format == "base64":
                                dl_service = self._get_dl()
                                base64_data = await dl_service.to_base64(url, self.token, "image")
                                if base64_data:
                                    content += f"![{img_id}]({base64_data})\n"
                                else:
                                    final_url = await self.process_url(url, "image")
                                    content += f"![{img_id}]({final_url})\n"
                            else:
                                final_url = await self.process_url(url, "image")
                                content += f"![{img_id}]({final_url})\n"
                    
                    if (meta := mr.get("metadata", {})).get("llm_info", {}).get("modelHash"):
                        fingerprint = meta["llm_info"]["modelHash"]
                            
        except Exception as e:
            logger.error(f"Collect processing error: {e}", extra={"model": self.model})
        finally:
            await self.close()
        
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": fingerprint,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": None, "annotations": []},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 0, "audio_tokens": 0, "image_tokens": 0},
                "completion_tokens_details": {"text_tokens": 0, "audio_tokens": 0, "reasoning_tokens": 0}
            }
        }


class VideoStreamProcessor(BaseProcessor):
    """视频流式响应处理器"""
    
    def __init__(self, model: str, token: str = "", think: bool = None):
        super().__init__(model, token)
        self.response_id: Optional[str] = None
        self.think_opened: bool = False
        self.role_sent: bool = False
        self.video_format = get_config("app.video_format", "url")
        
        if think is None:
            self.show_think = get_config("grok.thinking", False)
        else:
            self.show_think = think
    
    def _build_video_html(self, video_url: str, thumbnail_url: str = "") -> str:
        """构建视频 HTML 标签"""
        if get_config("grok.video_poster_preview", False):
            return _build_video_poster_preview(video_url, thumbnail_url)
        poster_attr = f' poster="{thumbnail_url}"' if thumbnail_url else ""
        return f'''<video id="video" controls="" preload="none"{poster_attr}>
  <source id="mp4" src="{video_url}" type="video/mp4">
</video>'''
    
    async def process(self, response: AsyncIterable[bytes]) -> AsyncGenerator[str, None]:
        """处理视频流式响应"""
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                resp = data.get("result", {}).get("response", {})
                
                if rid := resp.get("responseId"):
                    self.response_id = rid
                
                # 首次发送 role
                if not self.role_sent:
                    yield self._sse(role="assistant")
                    self.role_sent = True
                
                # 视频生成进度
                if video_resp := resp.get("streamingVideoGenerationResponse"):
                    progress = video_resp.get("progress", 0)
                    
                    if self.show_think:
                        if not self.think_opened:
                            yield self._sse("<think>\n")
                            self.think_opened = True
                        yield self._sse(f"正在生成视频中，当前进度{progress}%\n")
                    
                    if progress == 100:
                        video_url = video_resp.get("videoUrl", "")
                        thumbnail_url = video_resp.get("thumbnailImageUrl", "")
                        
                        if self.think_opened and self.show_think:
                            yield self._sse("</think>\n")
                            self.think_opened = False
                        
                        if video_url:
                            final_video_url = await self.process_url(video_url, "video")
                            final_thumbnail_url = ""
                            if thumbnail_url:
                                final_thumbnail_url = await self.process_url(thumbnail_url, "image")
                            
                            video_html = self._build_video_html(final_video_url, final_thumbnail_url)
                            yield self._sse(video_html)
                            
                            logger.info(f"Video generated: {video_url}")
                    continue
                        
            if self.think_opened:
                yield self._sse("</think>\n")
            yield self._sse(finish="stop")
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Video stream processing error: {e}", extra={"model": self.model})
        finally:
            await self.close()


class VideoCollectProcessor(BaseProcessor):
    """视频非流式响应处理器"""
    
    def __init__(self, model: str, token: str = ""):
        super().__init__(model, token)
        self.video_format = get_config("app.video_format", "url")
    
    def _build_video_html(self, video_url: str, thumbnail_url: str = "") -> str:
        if get_config("grok.video_poster_preview", False):
            return _build_video_poster_preview(video_url, thumbnail_url)
        poster_attr = f' poster="{thumbnail_url}"' if thumbnail_url else ""
        return f'''<video id="video" controls="" preload="none"{poster_attr}>
  <source id="mp4" src="{video_url}" type="video/mp4">
</video>'''
    
    async def process(self, response: AsyncIterable[bytes]) -> dict[str, Any]:
        """处理并收集视频响应"""
        response_id = ""
        content = ""
        
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                resp = data.get("result", {}).get("response", {})
                
                if video_resp := resp.get("streamingVideoGenerationResponse"):
                    if video_resp.get("progress") == 100:
                        response_id = resp.get("responseId", "")
                        video_url = video_resp.get("videoUrl", "")
                        thumbnail_url = video_resp.get("thumbnailImageUrl", "")
                        
                        if video_url:
                            final_video_url = await self.process_url(video_url, "video")
                            final_thumbnail_url = ""
                            if thumbnail_url:
                                final_thumbnail_url = await self.process_url(thumbnail_url, "image")
                            
                            content = self._build_video_html(final_video_url, final_thumbnail_url)
                            logger.info(f"Video generated: {video_url}")
                            
        except Exception as e:
            logger.error(f"Video collect processing error: {e}", extra={"model": self.model})
        finally:
            await self.close()
        
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": None},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }


class ImageStreamProcessor(BaseProcessor):
    """图片生成流式响应处理器"""
    
    def __init__(
        self,
        model: str,
        token: str = "",
        n: int = 1,
        response_format: str = "b64_json",
    ):
        super().__init__(model, token)
        self.partial_index = 0
        self.n = n
        self.target_index = random.randint(0, 1) if n == 1 else None
        self.response_format = (response_format or "b64_json").lower()
        if self.response_format == "url":
            self.response_field = "url"
        elif self.response_format == "base64":
            self.response_field = "base64"
        else:
            self.response_field = "b64_json"
    
    def _sse(self, event: str, data: dict) -> str:
        """构建 SSE 响应 (覆盖基类)"""
        return f"event: {event}\ndata: {orjson.dumps(data).decode()}\n\n"
    
    async def process(self, response: AsyncIterable[bytes]) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        final_images = []
        
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                resp = data.get("result", {}).get("response", {})
                
                # 图片生成进度
                if img := resp.get("streamingImageGenerationResponse"):
                    image_index = img.get("imageIndex", 0)
                    progress = img.get("progress", 0)
                    
                    if self.n == 1 and image_index != self.target_index:
                        continue
                    
                    out_index = 0 if self.n == 1 else image_index
                    
                    yield self._sse("image_generation.partial_image", {
                        "type": "image_generation.partial_image",
                        self.response_field: "",
                        "index": out_index,
                        "progress": progress
                    })
                    continue
                
                # modelResponse
                if mr := resp.get("modelResponse"):
                    if urls := mr.get("generatedImageUrls"):
                        for url in urls:
                            if self.response_format == "url":
                                processed = await self.process_url(url, "image")
                                if processed:
                                    final_images.append(processed)
                                continue
                            dl_service = self._get_dl()
                            base64_data = await dl_service.to_base64(url, self.token, "image")
                            if base64_data:
                                if "," in base64_data:
                                    b64 = base64_data.split(",", 1)[1]
                                else:
                                    b64 = base64_data
                                final_images.append(b64)
                    continue
                    
            for index, b64 in enumerate(final_images):
                if self.n == 1:
                    if index != self.target_index:
                        continue
                    out_index = 0
                else:
                    out_index = index
                
                yield self._sse("image_generation.completed", {
                    "type": "image_generation.completed",
                    self.response_field: b64,
                    "index": out_index,
                    "usage": {
                        "total_tokens": 50,
                        "input_tokens": 25,
                        "output_tokens": 25,
                        "input_tokens_details": {"text_tokens": 5, "image_tokens": 20}
                    }
                })
        except Exception as e:
            logger.error(f"Image stream processing error: {e}")
            raise
        finally:
            await self.close()


class ImageCollectProcessor(BaseProcessor):
    """图片生成非流式响应处理器"""
    
    def __init__(
        self,
        model: str,
        token: str = "",
        response_format: str = "b64_json",
    ):
        super().__init__(model, token)
        self.response_format = (response_format or "b64_json").lower()
    
    async def process(self, response: AsyncIterable[bytes]) -> List[str]:
        """处理并收集图片"""
        images = []
        
        try:
            async for line in response:
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                
                resp = data.get("result", {}).get("response", {})
                
                if mr := resp.get("modelResponse"):
                    if urls := mr.get("generatedImageUrls"):
                        for url in urls:
                            if self.response_format == "url":
                                processed = await self.process_url(url, "image")
                                if processed:
                                    images.append(processed)
                                continue
                            dl_service = self._get_dl()
                            base64_data = await dl_service.to_base64(url, self.token, "image")
                            if base64_data:
                                if "," in base64_data:
                                    b64 = base64_data.split(",", 1)[1]
                                else:
                                    b64 = base64_data
                                images.append(b64)
                                
        except Exception as e:
            logger.error(f"Image collect processing error: {e}")
        finally:
            await self.close()
        
        return images


__all__ = [
    "StreamProcessor",
    "CollectProcessor",
    "VideoStreamProcessor",
    "VideoCollectProcessor",
    "ImageStreamProcessor",
    "ImageCollectProcessor",
]
