(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const API = { status: '/api/status', tracks: '/api/tracks', events: '/api/events', stream: '/video.mjpg' };
  const video = $('video');
  let streamTimer;
  const value = (obj, ...keys) => { for (const k of keys) if (obj?.[k] !== undefined && obj[k] !== null) return obj[k]; return null; };
  const listFrom = data => Array.isArray(data) ? data : (data?.items || data?.tracks || data?.events || data?.detections || []);
  const fmt = (v, digits=2) => Number.isFinite(Number(v)) ? Number(v).toFixed(digits) : '—';
  const safe = v => String(v ?? '—');
  async function get(path) {
    const response = await fetch(path, { cache: 'no-store', headers: { Accept: 'application/json' } });
    if (!response.ok) throw new Error(`${path} returned HTTP ${response.status}`);
    return response.json();
  }
  function renderStatus(s) {
    const cam = value(s, 'camera_status', 'cameraState', 'camera');
    const cameraOnline = typeof cam === 'object' ? !!value(cam, 'online', 'connected', 'available') : ['online','connected','ready'].includes(String(cam).toLowerCase());
    $('camera-status').textContent = cameraOnline ? 'Online' : (cam ? 'Unavailable' : 'Unknown');
    $('camera-dot').className = `status-dot ${cameraOnline ? 'online' : 'offline'}`;
    $('camera-detail').textContent = typeof cam === 'object' ? safe(value(cam,'name','source','message') || (cameraOnline ? 'Frame source connected' : 'No frame source')) : (safe(value(s,'camera_message','camera_detail') || (cameraOnline ? 'Frame source connected' : 'Camera unavailable')));
    $('fps').textContent = fmt(value(s,'fps','processing_fps'),1);
    $('latency').textContent = fmt(value(s,'latency_ms','processing_latency_ms','latency'),0);
    $('active-count').textContent = safe(value(s,'active_tracks','track_count') ?? '—');
    $('detection-count').textContent = safe(value(s,'detection_count','detections_count') ?? '—');
    const mode = String(value(s,'mode','application_mode') || '').toLowerCase();
    document.querySelectorAll('.mode-button').forEach(b => b.classList.toggle('active', mode && ((mode.includes('industrial') && b.dataset.mode==='industrial') || (!mode.includes('industrial') && b.dataset.mode==='human'))));
    const width=value(s,'frame_width','width'), height=value(s,'frame_height','height');
    if(width && height) $('stream-resolution').textContent=`${width} × ${height}`;
    $('stream-caption').textContent = cameraOnline ? 'Live feed with backend overlays' : 'Camera unavailable';
    $('map-caption').textContent = value(s,'localization_status','calibration_status') || 'Calibrated camera plane';
    const banner=$('connection-banner'); banner.className='connection-banner online'; banner.textContent=`Local service connected${mode ? ` · ${mode.toUpperCase()} MODE` : ''}`;
  }
  function renderTracks(payload) {
    const tracks=listFrom(payload); const tbody=$('tracks'); tbody.replaceChildren();
    $('track-updated').textContent=`Updated ${new Date().toLocaleTimeString()}`;
    if(!tracks.length){tbody.innerHTML='<tr><td colspan="5" class="empty-cell">No active tracks reported</td></tr>';}
    tracks.forEach(t=>{
      const id=value(t,'id','track_id'), cls=value(t,'class_name','class','label'), state=value(t,'state','status')||'TRACKING';
      const x=value(t,'world_x','x'), y=value(t,'world_y','y'), speed=value(t,'speed','speed_mps');
      const pos=x!==null&&y!==null?`${fmt(x)} m, ${fmt(y)} m`:'—';
      const direction=value(t,'direction','heading');
      const tr=document.createElement('tr'); tr.innerHTML=`<td><div class="object-id"></div><div class="object-class"></div></td><td></td><td></td><td></td><td><span class="state-pill"></span></td>`;
      tr.children[0].querySelector('.object-id').textContent=safe(id); tr.children[0].querySelector('.object-class').textContent=safe(cls);
      const pixelSpeed=value(t,'speed_px_s');
      tr.children[1].textContent=pos; tr.children[2].textContent=speed!==null?`${fmt(speed,1)} m/s${direction?` · ${direction}`:''}`:(pixelSpeed!==null?`${fmt(pixelSpeed,0)} px/s${direction?` · ${direction}`:''}`:'—'); tr.children[3].textContent=safe(value(t,'zone','zone_name'));
      const pill=tr.children[4].querySelector('.state-pill'); pill.textContent=String(state).toUpperCase(); if(/lost|removed|missing/i.test(state))pill.classList.add('lost'); tbody.append(tr);
    });
    renderMap(tracks);
  }
  function renderMap(tracks) {
    const points=$('map-points'); points.replaceChildren(); let located=0;
    const statusWindow=window.__vxStatus||{}; const width=Number(value(statusWindow,'map_width_m','room_width_m'))||10; const height=Number(value(statusWindow,'map_height_m','room_height_m'))||8;
    tracks.forEach(t=>{const x=Number(value(t,'world_x','x')), y=Number(value(t,'world_y','y')); if(!Number.isFinite(x)||!Number.isFinite(y))return; located++;
      const dot=document.createElement('div'); dot.className='map-dotpoint'+(/component|object|product|part/i.test(String(value(t,'class_name','class','label')||''))?' component':''); dot.style.left=`${Math.max(2,Math.min(98,x/width*100))}%`; dot.style.top=`${100-Math.max(2,Math.min(98,y/height*100))}%`;
      const label=document.createElement('span'); label.textContent=safe(value(t,'id','track_id')); dot.append(label); points.append(dot);
    }); $('map-empty').hidden=located>0; $('map-unit').textContent=String(value(statusWindow,'coordinate_unit')||'METERS').toUpperCase();
  }
  function renderEvents(payload) {
    const events=listFrom(payload); $('event-count').textContent=String(events.length); const box=$('events'); box.replaceChildren();
    if(!events.length){box.innerHTML='<div class="empty-cell">No events reported</div>';return;}
    events.slice(0,40).forEach(e=>{const row=document.createElement('div');row.className='event-row';const time=value(e,'timestamp','time','created_at');let display='—';if(time){const d=new Date(time);display=Number.isNaN(d.getTime())?String(time):d.toLocaleTimeString();}const level=String(value(e,'severity','level','type')||'info').toLowerCase();row.innerHTML='<div class="event-time"></div><div class="event-mark"></div><div><div class="event-text"></div><div class="event-meta"></div></div>';row.children[0].textContent=display;row.children[1].classList.toggle('warn',/warn|anomaly/i.test(level));row.children[1].classList.toggle('error',/error|critical/i.test(level));row.querySelector('.event-text').textContent=safe(value(e,'message','description','name'));row.querySelector('.event-meta').textContent=[value(e,'object_id','track_id'),value(e,'zone','zone_name'),level.toUpperCase()].filter(Boolean).join(' · ');box.append(row);});
  }
  async function poll() {
    const results=await Promise.allSettled([get(API.status),get(API.tracks),get(API.events)]);
    if(results[0].status==='fulfilled'){window.__vxStatus=results[0].value;renderStatus(results[0].value);} else {const b=$('connection-banner');b.className='connection-banner error';b.textContent=`Local service unavailable · ${results[0].reason.message}`;$('camera-status').textContent='Unavailable';$('camera-dot').className='status-dot offline';$('camera-detail').textContent='Could not read /api/status';}
    if(results[1].status==='fulfilled')renderTracks(results[1].value); else {$('track-updated').textContent='Track feed unavailable';}
    if(results[2].status==='fulfilled')renderEvents(results[2].value);
  }
  video.addEventListener('load',()=>{video.classList.add('loaded');$('video-message').hidden=true;clearTimeout(streamTimer);});
  video.addEventListener('error',()=>{video.classList.remove('loaded');$('video-message').hidden=false;$('video-message').querySelector('strong').textContent='CAMERA UNAVAILABLE';$('video-message').querySelector('small').textContent='No live stream at /video.mjpg';clearTimeout(streamTimer);streamTimer=setTimeout(()=>{video.src=`${API.stream}?t=${Date.now()}`;},5000);});
  $('refresh').addEventListener('click',()=>{poll();video.src=`${API.stream}?t=${Date.now()}`;});
  document.querySelectorAll('.mode-button').forEach(button=>button.addEventListener('click',async()=>{try{await fetch('/api/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:button.dataset.mode})});poll();}catch(e){const b=$('connection-banner');b.className='connection-banner error';b.textContent=`Could not change mode · ${e.message}`;}}));
  setInterval(()=>{$('clock').textContent=new Date().toLocaleTimeString();},1000); $('clock').textContent=new Date().toLocaleTimeString(); poll();setInterval(poll,1500);
})();
