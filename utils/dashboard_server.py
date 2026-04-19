"""Tiny stdlib-only HTTP server exposing the DashboardBus over SSE.

Usage (from main.py when --dashboard is passed):
    from utils.dashboard_server import start
    start(host="127.0.0.1", port=8765, project_root=Path(...))
"""
from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

from utils.dashboard_bus import bus

logger = logging.getLogger(__name__)

_PROJECT_ROOT: Path = Path(".").resolve()


INDEX_HTML = """<!doctype html>
<meta charset="utf-8">
<title>endless_winter live</title>
<style>
 body{margin:0;font:13px/1.4 -apple-system,Menlo,monospace;background:#111;color:#ddd}
 header{padding:8px 12px;background:#222;border-bottom:1px solid #333;display:flex;gap:12px;align-items:center}
 header h1{font-size:14px;margin:0;font-weight:600}
 header .stat{color:#888}
 main{display:grid;grid-template-columns:420px 1fr;height:calc(100vh - 40px)}
 #rounds{overflow-y:auto;border-right:1px solid #333}
 .card{padding:10px 12px;border-bottom:1px solid #222;cursor:pointer}
 .card:hover{background:#1a1a1a}
 .card.active{background:#1e2a3a}
 .card .hd{display:flex;justify-content:space-between;color:#9af}
 .card .line{color:#bbb;margin-top:2px}
 .card .tgt{color:#ffd270}
 .card .ok{color:#7fe58c}.card .fail{color:#ff6a6a}
 #detail{overflow-y:auto;padding:12px}
 #detail .imgwrap{position:relative;display:inline-block;margin-bottom:10px}
 #detail .imgwrap img{max-width:100%;border:1px solid #333;display:block}
 #detail .marker{position:absolute;width:24px;height:24px;margin:-12px 0 0 -12px;pointer-events:none;
   border:2px solid var(--mc, #ff3b6b);border-radius:50%;box-shadow:0 0 0 1px #000,0 0 8px var(--mc, #ff3b6b)}
 #detail .marker::before,#detail .marker::after{content:'';position:absolute;background:var(--mc, #ff3b6b)}
 #detail .marker::before{left:10px;top:-6px;width:2px;height:34px}
 #detail .marker::after{top:10px;left:-6px;height:2px;width:34px}
 #detail .marker.exec{--mc:#43f27b}
 #detail .marker.model{--mc:#ff3b6b}
 #detail .mlabel{position:absolute;transform:translate(-50%,-160%);background:rgba(0,0,0,.75);
   color:var(--mc,#ffb3c6);padding:1px 5px;border-radius:3px;font-size:11px;white-space:nowrap;pointer-events:none}
 #detail .legend{font-size:11px;color:#888;margin-bottom:6px}
 #detail .legend .dot{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle;margin:0 4px 0 10px}
 #detail .legend .dot.model{background:#ff3b6b}
 #detail .legend .dot.exec{background:#43f27b}
 #detail .sec{margin-top:10px}
 #detail h3{font-size:12px;color:#9af;margin:0 0 4px 0;text-transform:uppercase;letter-spacing:.5px}
 #detail pre{background:#0a0a0a;padding:8px;border:1px solid #222;white-space:pre-wrap;word-break:break-word;max-height:300px;overflow:auto}
 .pill{display:inline-block;padding:1px 6px;border-radius:3px;background:#333;margin-right:4px;font-size:11px}
 .pill.s1{background:#2a4a6a}.pill.s2{background:#6a3a2a}
</style>
<header>
  <h1>endless_winter · live</h1>
  <span class="stat" id="conn">connecting…</span>
  <span class="stat" id="count">0 events</span>
</header>
<main>
  <div id="rounds"></div>
  <div id="detail"><em>waiting for events…</em></div>
</main>
<script>
const rounds = new Map();  // roundNo -> {screenshot, scene, task, infers:[], action, verify}
let active = null;
const $rounds = document.getElementById('rounds');
const $detail = document.getElementById('detail');
const $count  = document.getElementById('count');
const $conn   = document.getElementById('conn');
let evCount = 0;

function ensureRound(n, withCard){
  if(!rounds.has(n)){
    rounds.set(n, {n, infers:[], action:null, verify:null, screenshot:null,
                   img_w:null, img_h:null, scene:null, task:null, hasCard:false});
  }
  const r = rounds.get(n);
  if(withCard && !r.hasCard){
    const card = document.createElement('div');
    card.className='card'; card.id='card-'+n;
    card.onclick = () => select(n);
    $rounds.prepend(card);
    r.hasCard = true;
  }
  return r;
}

function renderCard(n){
  const r = rounds.get(n); if(!r) return;
  const el = document.getElementById('card-'+n); if(!el) return;
  const last = r.infers[r.infers.length-1];
  const a = r.action;
  const v = r.verify;
  el.innerHTML = `
    <div class="hd"><span>#${n}</span><span>${r.scene||''} ${r.task?'· '+r.task:''}</span></div>
    ${last?`<div class="line"><span class="pill ${last.system}">${last.system}</span>${(last.latency_ms||0)|0}ms · ${(last.response||'').slice(0,60).replace(/\\n/g,' ')}</div>`:''}
    ${a?`<div class="line"><span class="tgt">→ ${a.target||''}</span> (${a.x||'?'},${a.y||'?'}) ${a.success===false?'<span class="fail">skip</span>':'<span class="ok">ok</span>'}</div>`:''}
    ${v?`<div class="line">verify: ${v.ok?'<span class="ok">ok</span>':'<span class="fail">'+(v.reason||'fail')+'</span>'} scene_match=${v.scene_matched}</div>`:''}
  `;
}

function select(n){
  active = n;
  for(const el of document.querySelectorAll('.card.active')) el.classList.remove('active');
  const card = document.getElementById('card-'+n); if(card) card.classList.add('active');
  const r = rounds.get(n); if(!r){$detail.innerText='no round'; return;}
  const legend = r.screenshot
    ? `<div class="legend"><span class="dot model"></span>模型目标 <span class="dot exec"></span>实际点击</div>`
    : '';
  const imgBlock = r.screenshot
    ? `${legend}<div class="imgwrap" id="imgwrap-${n}"><img id="img-${n}" src="/img?path=${encodeURIComponent(r.screenshot)}"></div>`
    : '';
  const infers = r.infers.map(i => `
    <div class="sec">
      <h3><span class="pill ${i.system}">${i.system}</span> ${(i.latency_ms||0)|0}ms · image ${(i.image_kb||0)|0}KB</h3>
      <h3>prompt</h3><pre>${escape(i.prompt||'')}</pre>
      <h3>response</h3><pre>${escape(i.response||'')}</pre>
    </div>
  `).join('');
  const action = r.action ? `<div class="sec"><h3>action</h3><pre>${escape(JSON.stringify(r.action,null,2))}</pre></div>` : '';
  const verify = r.verify ? `<div class="sec"><h3>verify</h3><pre>${escape(JSON.stringify(r.verify,null,2))}</pre></div>` : '';
  $detail.innerHTML = `<h3>round #${n} · ${r.scene||''} ${r.task?'· '+r.task:''}</h3>${imgBlock}${infers}${action}${verify}`;
  if(r.screenshot){
    const el = document.getElementById('img-'+n);
    if(el){
      const draw = () => drawMarkers(n);
      if(el.complete && el.naturalWidth){ draw(); }
      else { el.addEventListener('load', draw, {once:true}); }
    }
  }
}

function drawMarkers(n){
  const r = rounds.get(n); if(!r || !r.screenshot) return;
  const wrap = document.getElementById('imgwrap-'+n);
  const img  = document.getElementById('img-'+n);
  if(!wrap || !img || !img.naturalWidth) return;
  // Prefer image_w/h from infer event; fall back to natural.
  const iw = r.img_w || img.naturalWidth;
  const ih = r.img_h || img.naturalHeight;
  // Clear any stale markers.
  wrap.querySelectorAll('.marker,.mlabel').forEach(e=>e.remove());
  const a = r.action;
  if(!a) return;
  // kind='model' = coords the LLM returned, 'exec' = post-snap coords actually clicked
  const points = [];
  if(a.type === 'drag'){
    if(a.x1!=null && a.y1!=null) points.push({x:a.x1, y:a.y1, kind:'model', label:'模型 start'});
    if(a.x2!=null && a.y2!=null) points.push({x:a.x2, y:a.y2, kind:'model', label:'模型 end'});
    if(a.executed_x1!=null && a.executed_y1!=null && (a.executed_x1!==a.x1 || a.executed_y1!==a.y1))
      points.push({x:a.executed_x1, y:a.executed_y1, kind:'exec', label:'实际 start'});
    if(a.executed_x2!=null && a.executed_y2!=null && (a.executed_x2!==a.x2 || a.executed_y2!==a.y2))
      points.push({x:a.executed_x2, y:a.executed_y2, kind:'exec', label:'实际 end'});
  } else if(a.x!=null && a.y!=null){
    points.push({x:a.x, y:a.y, kind:'model', label:`模型 (${a.x},${a.y})`});
    if(a.executed_x!=null && a.executed_y!=null && (a.executed_x!==a.x || a.executed_y!==a.y)){
      points.push({x:a.executed_x, y:a.executed_y, kind:'exec', label:`实际 (${a.executed_x},${a.executed_y})`});
    }
  }
  const rect = img.getBoundingClientRect();
  const sx = rect.width / iw;
  const sy = rect.height / ih;
  for(const p of points){
    const m = document.createElement('div'); m.className='marker '+p.kind;
    m.style.left = (p.x*sx)+'px'; m.style.top = (p.y*sy)+'px';
    wrap.appendChild(m);
    const lb = document.createElement('div'); lb.className='mlabel '+p.kind;
    lb.textContent = p.label;
    // Stack labels: exec labels render below the marker so they don't collide.
    lb.style.left = (p.x*sx)+'px';
    lb.style.top  = ((p.y*sy) + (p.kind==='exec' ? 28 : 0))+'px';
    if(p.kind==='exec') lb.style.transform = 'translate(-50%, 20%)';
    wrap.appendChild(lb);
  }
}
window.addEventListener('resize', () => { if(active!=null) drawMarkers(active); });
function escape(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

function handle(evt){
  evCount++; $count.textContent = evCount+' events';
  const {type, data} = evt;
  const n = data.round;
  // round_start alone doesn't create a card — skip rounds (no capture/infer/action)
  // would otherwise show as empty placeholders.
  if(type==='round_start'){ ensureRound(n, false); return; }
  if(type==='screenshot_captured'){
    const r = ensureRound(n, true);
    r.screenshot=data.path; r.scene=data.scene; r.task=data.task;
    if(data.width)  r.img_w=data.width;
    if(data.height) r.img_h=data.height;
    renderCard(n); if(active==null || active===n) select(n);
  }
  else if(type==='infer'){
    const r = ensureRound(n, true); r.infers.push(data);
    if(data.image_w) r.img_w=data.image_w;
    if(data.image_h) r.img_h=data.image_h;
    renderCard(n); if(active==null || active===n) select(n);
  }
  else if(type==='action'){
    const r = ensureRound(n, true); r.action=data;
    renderCard(n); if(active==null || active===n) select(n);
  }
  else if(type==='verify'){
    const r = ensureRound(n, true); r.verify=data;
    renderCard(n); if(active===n) select(n);
  }
}

const es = new EventSource('/events');
es.onopen = () => { $conn.textContent='connected'; $conn.style.color='#7fe58c'; };
es.onerror = () => { $conn.textContent='reconnecting…'; $conn.style.color='#ff6a6a'; };
es.onmessage = (e) => { try { handle(JSON.parse(e.data)); } catch(err) { console.error(err); } };
</script>
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return  # silence

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/index.html":
            self._reply(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
        elif path == "/events":
            self._stream_events()
        elif path == "/img":
            self._serve_image(parsed.query)
        else:
            self._reply(404, "text/plain", b"not found")

    def _reply(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_image(self, query: str):
        parts = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
        rel = unquote(parts.get("path", ""))
        if not rel:
            self._reply(400, "text/plain", b"missing path")
            return
        full = (_PROJECT_ROOT / rel).resolve()
        if not str(full).startswith(str(_PROJECT_ROOT)):
            self._reply(403, "text/plain", b"forbidden")
            return
        if not full.exists() or not full.is_file():
            self._reply(404, "text/plain", b"not found")
            return
        data = full.read_bytes()
        ctype = "image/png" if full.suffix.lower() == ".png" else "application/octet-stream"
        self._reply(200, ctype, data)

    def _stream_events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        q = bus.subscribe(include_backlog=True)
        try:
            while True:
                try:
                    evt = q.get(timeout=15)
                except Exception:
                    try:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    except Exception:
                        break
                    continue
                try:
                    line = b"data: " + json.dumps(evt, ensure_ascii=False).encode("utf-8") + b"\n\n"
                    self.wfile.write(line)
                    self.wfile.flush()
                except Exception:
                    break
        finally:
            bus.unsubscribe(q)


def start(host: str = "127.0.0.1", port: int = 8765, project_root: Optional[Path] = None) -> ThreadingHTTPServer:
    """Enable the bus and spawn the HTTP server in a background thread."""
    global _PROJECT_ROOT
    if project_root is not None:
        _PROJECT_ROOT = Path(project_root).resolve()
    # Persist every event to debug/dashboard/ so a past session can be replayed
    # or grep-inspected even after the game process exits.
    import time as _time
    ts = _time.strftime("%Y%m%d_%H%M%S", _time.localtime())
    log_path = _PROJECT_ROOT / "debug" / "dashboard" / f"sess-{ts}.jsonl"
    bus.enable(log_path=log_path)
    server = ThreadingHTTPServer((host, port), _Handler)
    t = threading.Thread(target=server.serve_forever, name="dashboard-http", daemon=True)
    t.start()
    logger.info(f"[DASH] live dashboard at http://{host}:{port}/")
    return server
