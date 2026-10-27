let data, selected = 1, filter = 'all';
const $ = id => document.getElementById(id);
const band = value => value <= 30 ? 'critical' : value <= 60 ? 'watch' : 'healthy';
const labels = {critical:'Low RUL',watch:'Monitor',healthy:'Higher RUL'};
const colors = {healthy:'#45896b',watch:'#dbb665',critical:'#cf8d75'};
const fmt = value => Number(value).toFixed(2);
const svg = (body, viewBox, label) => `<svg viewBox="${viewBox}" role="img" aria-label="${label}">${body}</svg>`;

function renderOverview() {
 const m = data.metrics;
 $('metrics').innerHTML = [
  ['Test engines','100','engines','NASA C-MAPSS · FD001'],
  ['Mean absolute error',fmt(m.capped_target_MAE),'cycles','Capped target · held-out test'],
  ['Root mean squared error',fmt(m.capped_target_RMSE),'cycles','Capped target · held-out test'],
  ['R² score',m.capped_target_R2.toFixed(3),'','Capped target · held-out test']
 ].map(([label,value,unit,note]) => `<article class="metric"><div class="metric-label">${label}<span class="badge">↗</span></div><strong>${value}<small>${unit}</small></strong><p>${note}</p></article>`).join('');
 let chart = '';
 for(let n=0;n<=125;n+=25){let x=45+n*3.35,y=210-n*1.45;chart+=`<line x1="45" y1="${y}" x2="464" y2="${y}" stroke="#edf1ed"/><text x="32" y="${y+3}" text-anchor="end" fill="#94a096" font-size="9">${n}</text><text x="${x}" y="230" text-anchor="middle" fill="#94a096" font-size="9">${n}</text>`;}
 chart += '<path d="M45 210 L464 28.75" stroke="#a6b3a6" stroke-dasharray="5 4" fill="none"/>';
 chart += data.engines.map(e=>`<circle cx="${45+e.actual_RUL_capped*3.35}" cy="${210-e.predicted_RUL_capped*1.45}" r="3.3" fill="#438768" opacity=".72"><title>Engine ${e.engine_id}: actual ${e.actual_RUL_capped}, predicted ${fmt(e.predicted_RUL_capped)} cycles</title></circle>`).join('');
 chart += '<text x="250" y="250" text-anchor="middle" fill="#8b988e" font-size="9">Actual remaining useful life (cycles)</text><text transform="translate(11 120) rotate(-90)" text-anchor="middle" fill="#8b988e" font-size="9">Predicted RUL (cycles)</text>';
 $('scatter').innerHTML = svg(chart,'0 0 490 260','Scatter chart of actual capped RUL against predicted RUL for 100 engines');
 const counts={healthy:0,watch:0,critical:0};data.engines.forEach(e=>counts[band(e.predicted_RUL_capped)]++);
 $('donut').innerHTML=`<div class="ring" style="background:conic-gradient(${colors.healthy} 0 ${counts.healthy}%,${colors.watch} ${counts.healthy}% ${counts.healthy+counts.watch}%,${colors.critical} ${counts.healthy+counts.watch}% 100%)"><div><strong>100</strong><small>TEST ENGINES</small></div></div>`;
 $('health-legend').innerHTML=Object.entries(counts).map(([key,value])=>`<div class="health-row"><span class="key" style="background:${colors[key]}"></span>${labels[key]} <span style="color:#a0a89f">${key==='healthy'?' > 60':key==='watch'?'>30–60':'≤ 30'} cycles</span><b>${value}</b></div>`).join('');
 $('model-rows').innerHTML=data.models.map(m=>`<tr><td>${m.model}</td><td>${fmt(m.mean_CV_MAE)}</td><td>± ${fmt(m.std_fold_MAE)}</td><td>${fmt(m.mean_CV_RMSE)}</td><td>${m.mean_CV_R2.toFixed(3)}</td></tr>`).join('');
}
function renderFleet(){
 const query=$('search').value.trim();
 const engines=data.engines.filter(e=>(filter==='all'||band(e.predicted_RUL_capped)===filter)&&String(e.engine_id).includes(query));
 $('engine-rows').innerHTML=engines.length?engines.map(e=>`<tr class="${e.engine_id===selected?'chosen':''}"><td><button data-engine="${e.engine_id}" aria-label="Select engine ${e.engine_id}" aria-pressed="${selected===e.engine_id}">ENG-${String(e.engine_id).padStart(3,'0')}</button></td><td>${e.last_observed_cycle} cycles</td><td>${e.predicted_RUL_capped.toFixed(1)} cycles</td><td><span class="status ${band(e.predicted_RUL_capped)}">${labels[band(e.predicted_RUL_capped)]}</span></td></tr>`).join(''):'<tr><td colspan="4">No engines match your search.</td></tr>';
 $('count').textContent=`Showing ${engines.length} of 100 engines · Select an ID for details`;
}
function renderDetail(){
 const e=data.engines.find(e=>e.engine_id===selected);
 $('detail').innerHTML=`<div class="detail-top"><h3>ENG-${String(selected).padStart(3,'0')}</h3><span class="status ${band(e.predicted_RUL_capped)}">${labels[band(e.predicted_RUL_capped)]}</span></div><div class="rul">${e.predicted_RUL_capped.toFixed(1)} <small>cycles remaining</small></div><p>Saved XGBoost prediction · capped at 125 cycles</p><div class="facts"><span>Last observed cycle<b>${e.last_observed_cycle}</b></span><span>Actual RUL<b>${e.actual_RUL} cycles</b></span><span>Capped error<b>${Math.abs(e.predicted_RUL_capped-e.actual_RUL_capped).toFixed(1)}</b></span></div><div class="sensor-label"><span>Observed sensor history</span><select id="sensor" aria-label="Choose sensor">${[2,4,7,11,12,14].map(n=>`<option value="${n}">Sensor ${n}</option>`).join('')}</select></div><div class="sensor-chart" id="sensor-chart"></div><p class="sensor-caption">Raw sensor values across observed cycles. No future readings.</p>`;
 $('sensor').addEventListener('change',renderSensor);renderSensor();
}
function renderSensor(){
 const index=[2,4,7,11,12,14].indexOf(Number($('sensor').value))+1;
 const history=data.histories[String(selected)], values=history.map(r=>r[index]);
 const min=Math.min(...values),max=Math.max(...values),range=max-min||1;
 const x=r=>35+(r[0]-1)/Math.max(1,history.at(-1)[0]-1)*295;
 const y=r=>64-(r[index]-min)/range*50;
 const points=history.map(r=>`${x(r)},${y(r)}`).join(' ');
 $('sensor-chart').innerHTML=svg(`<path d="M35 70 H330" stroke="#e5ece5"/><polyline points="${points}" fill="none" stroke="#50866a" stroke-width="1.4"/><text x="0" y="18" font-size="8" fill="#8b998c">${max.toFixed(1)}</text><text x="0" y="66" font-size="8" fill="#8b998c">${min.toFixed(1)}</text><text x="35" y="85" font-size="8" fill="#8b998c">Cycle 1</text><text x="330" y="85" text-anchor="end" font-size="8" fill="#8b998c">Cycle ${history.at(-1)[0]}</text>`,'0 0 340 90',`Sensor ${$('sensor').value} for engine ${selected}, range ${min} to ${max}`);
}
$('search').addEventListener('input',()=>data&&renderFleet());
document.querySelector('.filters').addEventListener('click',event=>{const button=event.target.closest('button');if(!button||!data)return;filter=button.dataset.filter;document.querySelectorAll('.filters button').forEach(b=>b.classList.toggle('selected',b===button));renderFleet();});
$('engine-rows').addEventListener('click',event=>{const button=event.target.closest('[data-engine]');if(!button)return;selected=Number(button.dataset.engine);renderFleet();renderDetail();});
$('export').addEventListener('click',()=>{if(!data)return;const keys=Object.keys(data.engines[0]);const csv=[keys.join(','),...data.engines.map(e=>keys.map(k=>e[k]).join(','))].join('\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));const link=document.createElement('a');link.href=url;link.download='enginemind-fd001-predictions.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);$('message').textContent='Exported predictions for 100 engines';$('message').style.display='block';setTimeout(()=>$('message').style.display='none',3000);});
const observer=new IntersectionObserver(entries=>{entries.forEach(entry=>{if(entry.isIntersecting){document.querySelectorAll('nav a').forEach(a=>{const active=a.hash===`#${entry.target.id}`;a.classList.toggle('active',active);if(active){a.setAttribute('aria-current','location');$('breadcrumb').textContent=a.textContent.trim();}else a.removeAttribute('aria-current');});}});},{rootMargin:'-10% 0px -60% 0px'});document.querySelectorAll('main>section').forEach(s=>observer.observe(s));
fetch('data.json').then(r=>{if(!r.ok)throw Error('Data unavailable');return r.json();}).then(result=>{data=result;renderOverview();renderFleet();renderDetail();}).catch(()=>{ $('metrics').innerHTML='<p role="alert">Could not load evaluation data. Run python3 app/prepare_data.py and serve the app over HTTP, then reload.</p>'; $('export').disabled=true;});
