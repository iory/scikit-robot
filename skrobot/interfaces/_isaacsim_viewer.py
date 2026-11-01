"""Browser view of an IsaacSimRobotInterface, rendered by Isaac Sim's RTX.

The browser opens a WebSocket and sends where the camera should be; the
simulation pushes back JPEG frames drawn from there. Frames are pipelined:
each :meth:`RtxWebViewer.service` call renders once and sends what the
annotator holds, which is the frame of a few renders before (the RTX
pipeline hands its output to an annotator three renders late). Waiting for
each frame to come out instead -- as ``get_image`` does -- costs those
renders on every frame.

Isaac Sim's API is single-threaded. The HTTP/WebSocket server runs in a
thread of its own and only moves bytes; rendering happens in
:meth:`RtxWebViewer.service`, which the interface calls from ``step``.

Through Colab's port proxy (``google.colab.output.serve_kernel_port_as_iframe``)
a WebSocket round trip measured 86-91 ms, against 683-735 ms for an HTTP
request (T4 runtime, 2026-10-31), which is why frames go over a WebSocket.
"""

import io
import json
import logging
import queue
import struct
import threading
import time

import numpy as np

from skrobot.coordinates.math import matrix2quaternion


logger = logging.getLogger(__name__)

_CAMERA_PRIM = '/World/skrobot_viewer'

# Renders between a camera move and the annotator showing it, measured on an
# RTX 3090 with Isaac Sim 6.2 (it then blends in over a few more frames).
_LAG = 3

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>scikit-robot Isaac Sim view</title>
<style>body{margin:0;background:#111;color:#ccc;font:13px sans-serif}
#v{display:block;width:100%;max-width:1280px;cursor:grab;user-select:none}
#s{padding:4px 8px}</style></head><body>
<img id="v" draggable="false"><div id="s">connecting</div>
<script>
let az=-0.6, el=0.35, dist=3.0, target=[0,0,0.6], viewId=0;
const viewTime={}; let shown=-1, lat=0, n=0, t0=performance.now(), ping=0;
const v=document.getElementById('v'), s=document.getElementById('s');
const ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host
  +location.pathname.replace(/[^/]*$/,'')+'ws');
ws.binaryType='arraybuffer';
function send(o){ if(ws.readyState===1) ws.send(JSON.stringify(o)); }
function changed(){ viewId++; viewTime[viewId]=performance.now();
  send({view:{az:az,el:el,dist:dist,target:target,id:viewId}}); }
let drag=null;
v.oncontextmenu=e=>e.preventDefault();
v.onmousedown=e=>{drag={x:e.clientX,y:e.clientY,az:az,el:el,t:target.slice(),pan:e.button===2||e.shiftKey}};
window.onmouseup=()=>{drag=null};
window.onmousemove=e=>{ if(!drag) return;
  const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
  if(drag.pan){ const k=dist*0.0015, r=[-Math.sin(az),Math.cos(az),0],
      u=[-Math.sin(el)*Math.cos(az),-Math.sin(el)*Math.sin(az),Math.cos(el)];
    target=[0,1,2].map(i=>drag.t[i]-k*dx*r[i]+k*dy*u[i]);
  } else { az=drag.az-dx*0.01; el=Math.max(-0.2,Math.min(1.5,drag.el+dy*0.01)); }
  changed(); };
v.onwheel=e=>{ e.preventDefault(); dist=Math.max(0.3,Math.min(30,dist*Math.exp(e.deltaY*0.001))); changed(); };
ws.onopen=()=>{ s.textContent='connected'; changed(); setInterval(()=>send({ping:performance.now()}),1000); };
ws.onclose=e=>{ s.textContent='disconnected ('+e.code+')'; };
ws.onmessage=e=>{
  if(typeof e.data==='string'){ const m=JSON.parse(e.data);
    if(m.pong) ping=0.7*ping+0.3*(performance.now()-m.pong); return; }
  const h=new DataView(e.data,0,12), id=h.getInt32(0,true), simT=h.getFloat32(8,true);
  send({ack:1});
  if(id>shown && viewTime[id]!==undefined) lat=0.7*lat+0.3*(performance.now()-viewTime[id]);
  if(id>shown) shown=id;
  const old=v.src; v.src=URL.createObjectURL(new Blob([e.data.slice(12)],{type:'image/jpeg'}));
  if(old) URL.revokeObjectURL(old);
  n++; const dt=(performance.now()-t0)/1000;
  if(dt>1){ s.textContent=`${(n/dt).toFixed(1)} fps, view latency ${lat.toFixed(0)} ms, `
    +`ping ${ping.toFixed(0)} ms, sim ${simT.toFixed(1)} s   `
    +'drag: orbit / right-drag or shift-drag: pan / wheel: zoom';
    n=0; t0=performance.now(); }
};
</script></body></html>"""


def _look_at(eye, target):
    """USD camera orientation (w, x, y, z) at ``eye`` looking at ``target``.

    A USD camera looks along its -z with +y up; world +z is up.
    """
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    if np.linalg.norm(x) < 1e-9:  # looking straight down or up
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return matrix2quaternion(np.stack([x, y, z], axis=1))


class _Client(object):
    """One browser: its unacknowledged frames and its send queue."""

    def __init__(self, ws):
        self.ws = ws
        self.inflight = 0
        self.frames = queue.Queue()
        self.closed = threading.Event()


class RtxWebViewer(object):
    """Serve a browser page showing the simulation through an RTX camera.

    Parameters
    ----------
    ri : skrobot.interfaces._isaacsim.IsaacSimRobotInterface
        The interface whose stage to draw.
    port : int
        Port of the page (``/``) and its WebSocket (``/ws``).
    width, height : int
        Frame size [px].
    max_fps : float
        Most frames a second :meth:`service` renders; each render holds up
        the physics for its duration.
    max_inflight : int
        Frames sent to a browser and not yet acknowledged. More keeps the
        GPU busy across a slow link; fewer shows a moved view sooner.
    jpeg_quality : int
        JPEG quality of the frames.
    """

    def __init__(self, ri, port=8765, width=960, height=540, max_fps=30.0,
                 max_inflight=2, jpeg_quality=80):
        try:
            from websockets.datastructures import Headers
            from websockets.http11 import Response
            from websockets.sync.server import serve
        except ImportError:
            raise ImportError(
                'the Isaac Sim viewer needs the websockets package:\n\n'
                '  $ pip install websockets\n')
        import omni.replicator.core as rep
        from pxr import Gf
        from pxr import UsdGeom

        self.ri = ri
        self.port = port
        self.min_interval = 1.0 / max_fps
        self.max_inflight = max_inflight
        self.jpeg_quality = jpeg_quality
        self._last_render = 0.0
        self._Gf = Gf

        camera = UsdGeom.Camera.Define(ri._stage, _CAMERA_PRIM)
        camera.CreateFocalLengthAttr(18.0)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(20.955 * height / float(width))
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 1000.0))
        self._translate = camera.AddTranslateOp()
        self._orient = camera.AddOrientOp()
        product = rep.create.render_product(_CAMERA_PRIM, (width, height))
        self._rgb = rep.AnnotatorRegistry.get_annotator('rgb')
        self._rgb.attach([product])

        # The view the browsers asked for last, and the view ids of the
        # renders in the RTX pipeline, so each frame carries the id of the
        # view it was drawn from.
        self._view = {'az': -0.6, 'el': 0.35, 'dist': 3.0,
                      'target': [0.0, 0.0, 0.6], 'id': 0}
        self._pipeline = []
        self._clients = []
        self._lock = threading.Lock()

        def process_request(connection, request):
            if request.path.split('?')[0].endswith('/ws'):
                return None
            body = PAGE.encode()
            return Response(200, 'OK', Headers([
                ('Content-Type', 'text/html; charset=utf-8'),
                ('Content-Length', str(len(body)))]), body)

        self._server = serve(self._handle, '0.0.0.0', port,
                             process_request=process_request,
                             max_size=None, compression=None)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        logger.info('Isaac Sim viewer on port %d', port)

    def _handle(self, ws):
        """One browser connection, in the server's thread."""
        client = _Client(ws)
        with self._lock:
            self._clients.append(client)

        def receive():
            try:
                for message in ws:
                    m = json.loads(message)
                    if 'view' in m:
                        self._view = m['view']
                    if 'ack' in m:
                        client.inflight -= 1
                    if 'ping' in m:
                        ws.send(json.dumps({'pong': m['ping']}))
            except Exception:  # NOQA: the browser went away
                pass
            client.closed.set()

        threading.Thread(target=receive, daemon=True).start()
        try:
            while not client.closed.is_set():
                try:
                    frame = client.frames.get(timeout=0.5)
                except queue.Empty:
                    continue
                ws.send(frame)
        except Exception:  # NOQA: the browser went away
            pass
        finally:
            client.closed.set()
            with self._lock:
                self._clients.remove(client)

    def _set_camera(self, view):
        target = np.asarray(view['target'], dtype=np.float64)
        az, el, dist = view['az'], view['el'], view['dist']
        eye = target + dist * np.array([np.cos(el) * np.cos(az),
                                        np.cos(el) * np.sin(az),
                                        np.sin(el)])
        Gf = self._Gf
        self._translate.Set(Gf.Vec3d(*[float(v) for v in eye]))
        self._orient.Set(Gf.Quatf(*[float(v) for v in
                                    _look_at(eye, target)]))

    def service(self):
        """Render and send a frame, if a browser is waiting for one.

        Call from the main thread; ``IsaacSimRobotInterface.step`` does.
        """
        now = time.time()
        if now - self._last_render < self.min_interval:
            return
        with self._lock:
            ready = [c for c in self._clients
                     if not c.closed.is_set()
                     and c.inflight < self.max_inflight]
        if not ready:
            return
        self._last_render = now
        view = dict(self._view)
        self._set_camera(view)
        self.ri.world.render()
        self._pipeline.append(view.get('id', 0))
        # The annotator holds the frame of _LAG renders ago.
        if len(self._pipeline) > _LAG:
            self._pipeline.pop(0)
        image = np.asarray(self._rgb.get_data())
        if image.size == 0:
            return  # shaders still compiling
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(image[..., :3]).save(buf, 'JPEG',
                                             quality=self.jpeg_quality)
        frame = struct.pack('<iff', int(self._pipeline[0]),
                            (time.time() - now) * 1000.0,
                            self.ri.time) + buf.getvalue()
        for client in ready:
            client.inflight += 1
            client.frames.put(frame)

    def close(self):
        """Stop serving."""
        self._server.shutdown()
