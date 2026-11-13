#!/usr/bin/env python3
"""滚动快门校正 Web 界面（仅标准库 + 本项目的 rs 包）。

  python3 ui/server.py --port 8000
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rs import imaging, pipeline
from rs.exporter import Exporter

OUT_DIR = os.path.join(ROOT, "out")

PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>滚动快门校正器</title>
<style>
 body{font-family:system-ui,sans-serif;margin:16px;background:#f5f6f7;color:#222}
 h1{font-size:20px} h2{font-size:15px;margin:12px 0 6px}
 .panel{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin-bottom:12px}
 .ctrl{display:inline-block;margin:6px 14px 6px 0;font-size:13px;vertical-align:top}
 .ctrl label{display:block;color:#555;margin-bottom:2px}
 .ctrl output{color:#06c;font-weight:600}
 button{padding:8px 18px;border:0;border-radius:6px;background:#0b6bcb;color:#fff;font-size:14px;cursor:pointer}
 button.sec{background:#5a6472}
 .imgs{display:grid;grid-template-columns:1fr 1fr;gap:10px}
 .imgs figure{margin:0;background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px}
 .imgs img{width:100%;image-rendering:pixelated;display:block}
 figcaption{font-size:13px;color:#444;margin-bottom:6px}
 pre{background:#1e2430;color:#d7e0ee;padding:10px;border-radius:8px;font-size:12px;overflow:auto;max-height:340px}
 .legend span{display:inline-block;margin-right:12px;font-size:12px}
 .sw{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;vertical-align:-2px}
 #ver{font-family:monospace;color:#0a7}
 a{color:#0b6bcb}
</style>
</head>
<body>
<h1>逐行曝光滚动快门校正器 <small style="font-size:12px;color:#777">纯旋转 / 已知平面假设 · 逐像素曝光时刻求解</small></h1>

<div class="panel">
 <h2>参数（与导出文件共用同一参数版本）</h2>
 <div class="ctrl"><label>场景</label>
  <select id="scene">
   <option value="grid">网格控制点</option>
   <option value="lines">斜线/竖线</option>
   <option value="plane">平面场景（含平移）</option>
  </select></div>
 <div class="ctrl"><label>校正模型</label>
  <select id="model">
   <option value="rotation">纯旋转</option>
   <option value="plane">已知平面</option>
  </select></div>
 <div class="ctrl"><label>行读出方向</label>
  <select id="direction">
   <option value="top_to_bottom">自上而下</option>
   <option value="bottom_to_top">自下而上</option>
   <option value="left_to_right">自左而右</option>
   <option value="right_to_left">自右而左</option>
  </select></div>
 <div class="ctrl"><label>运动剖面</label>
  <select id="motion">
   <option value="mixed">混合</option>
   <option value="yaw">偏航</option>
   <option value="roll">横滚</option>
   <option value="pitch">俯仰</option>
   <option value="extreme">剧烈（诱发折叠）</option>
  </select></div>
 <div class="ctrl"><label>行读出总时长 <output id="o_readout">32</output> ms</label>
  <input type="range" id="readout_ms" min="5" max="60" step="1" value="32"></div>
 <div class="ctrl"><label>时间偏移 <output id="o_off">+2.0</output> ms（真实值 +2.0）</label>
  <input type="range" id="t_offset_ms" min="-6" max="6" step="0.1" value="2"></div>
 <div class="ctrl"><label>参考时刻 <output id="o_tref">16</output> ms</label>
  <input type="range" id="t_ref_ms" min="0" max="60" step="0.5" value="16"></div>
 <div class="ctrl"><label>陀螺采样率 <output id="o_gyro">500</output> Hz</label>
  <input type="range" id="gyro_hz" min="5" max="1000" step="5" value="500"></div>
 <div class="ctrl"><label>独立运动物体</label>
  <input type="checkbox" id="moving"> 加入（超出模型支持，仅标记）</div>
 <div style="margin-top:8px">
  <button id="run">运行校正</button>
  <button id="export" class="sec">导出图像+映射（版本 <span id="ver">—</span>）</button>
 </div>
</div>

<div class="imgs">
 <figure><figcaption>参考图（全局快门真值）</figcaption><img id="im_ref"></figure>
 <figure><figcaption>滚动快门输入（前向成像畸变）</figcaption><img id="im_in"></figure>
 <figure><figcaption>校正结果（无效区域留空，不做纹理补全）</figcaption><img id="im_out"></figure>
 <figure><figcaption>有效性掩码</figcaption><img id="im_val">
  <div class="legend">
   <span><span class="sw" style="background:#262626"></span>有效</span>
   <span><span class="sw" style="background:#e63c3c"></span>越界</span>
   <span><span class="sw" style="background:#ffa028"></span>姿态覆盖外</span>
   <span><span class="sw" style="background:#b450dc"></span>相机背面</span>
   <span><span class="sw" style="background:#3cc8dc"></span>局部折叠</span>
   <span><span class="sw" style="background:#f0dc3c"></span>独立运动</span>
  </div></figure>
</div>

<div class="panel"><h2>指标与有效性说明</h2><pre id="stats">尚未运行</pre></div>
<div class="panel"><h2>导出记录（图像与映射共享参数版本）</h2><div id="exports">—</div></div>

<script>
const $ = id => document.getElementById(id);
function cfg(){
  return {scene:$('scene').value, model:$('model').value, direction:$('direction').value,
    motion:$('motion').value, readout_ms:+$('readout_ms').value,
    t_offset_ms:+$('t_offset_ms').value, t_ref_ms:+$('t_ref_ms').value,
    gyro_hz:+$('gyro_hz').value, moving:$('moving').checked};
}
function show(o){
  $('o_readout').textContent = $('readout_ms').value;
  $('o_off').textContent = (+$('t_offset_ms').value).toFixed(1);
  $('o_tref').textContent = $('t_ref_ms').value;
  $('o_gyro').textContent = $('gyro_hz').value;
}
async function post(url, body){
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)});
  if(!r.ok) throw new Error(await r.text());
  return await r.json();
}
async function run(){
  show();
  $('stats').textContent = '计算中…';
  try{
    const d = await post('/api/run', cfg());
    $('im_ref').src = d.images.reference;
    $('im_in').src = d.images.input;
    $('im_out').src = d.images.corrected;
    $('im_val').src = d.images.validity;
    $('ver').textContent = d.stats.version;
    $('stats').textContent = JSON.stringify(d.stats, null, 2);
  }catch(e){ $('stats').textContent = '出错：' + e.message; }
}
async function doExport(){
  try{
    const d = await post('/api/export', cfg());
    $('exports').innerHTML = '版本 <b>' + d.version + '</b>：<br>' +
      d.files.map(f => '<a href="/out/' + f + '">' + f + '</a>').join('<br>');
  }catch(e){ $('exports').textContent = '导出失败：' + e.message; }
}
for(const id of ['readout_ms','t_offset_ms','t_ref_ms','gyro_hz']) $(id).oninput = show;
$('run').onclick = run; $('export').onclick = doExport;
run();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path.startswith("/out/"):
            name = os.path.normpath(self.path[len("/out/"):]).lstrip(os.sep)
            full = os.path.join(OUT_DIR, name)
            if not os.path.abspath(full).startswith(os.path.abspath(OUT_DIR)) or not os.path.isfile(full):
                self._send(404, json.dumps({"error": "not found"}))
                return
            ctype = ("image/png" if name.endswith(".png")
                     else "application/octet-stream" if name.endswith(".npz")
                     else "application/json")
            with open(full, "rb") as f:
                self._send(200, f.read(), ctype)
        elif self.path == "/api/exports":
            mf = os.path.join(OUT_DIR, "manifest.json")
            data = json.load(open(mf)) if os.path.exists(mf) else []
            self._send(200, json.dumps(data, ensure_ascii=False))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            cfg = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/run":
                res = pipeline.run_synthetic(cfg)
                imgs = res["images"]
                body = {
                    "stats": res["stats"],
                    "images": {
                        "reference": imaging.png_data_uri(imgs["reference"]),
                        "input": imaging.png_data_uri(imgs["input"]),
                        "corrected": imaging.png_data_uri(imgs["corrected"]),
                        "validity": imaging.png_data_uri(imaging.validity_image(res["inv"].reasons)),
                    },
                }
                self._send(200, json.dumps(body, ensure_ascii=False))
            elif self.path == "/api/export":
                res = pipeline.run_synthetic(cfg)
                exp = Exporter(OUT_DIR)
                out = exp.export(
                    name="ui_%s" % cfg.get("scene", "scene"),
                    params=res["params"],
                    images={
                        "reference": res["images"]["reference"],
                        "input": res["images"]["input"],
                        "corrected": res["images"]["corrected"],
                        "validity": imaging.validity_image(res["inv"].reasons).astype(float) / 255.0,
                    },
                    inv=res["inv"], track=res["track_corr"], K=res["K"])
                out["files"] = [os.path.basename(f) for f in out["files"]]
                self._send(200, json.dumps(out, ensure_ascii=False))
            else:
                self._send(404, json.dumps({"error": "not found"}))
        except Exception as e:  # surface pipeline errors to the UI
            self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False))

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("滚动快门校正器 UI: http://%s:%d/" % (a.host, a.port))
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
