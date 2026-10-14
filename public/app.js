const $ = id => document.getElementById(id);
const titles = {analyze:'Text analysis',batch:'Batch analysis',preprocess:'Preprocessing',metrics:'Evaluation',about:'Models & limitations'};
function el(tag, text, cls){const e=document.createElement(tag); if(text!==undefined)e.textContent=String(text);if(cls)e.className=cls;return e;}
function notice(target,text,error=false){target.replaceChildren(el('div',text,'notice'+(error?' error':'')));}
async function request(path,options){const r=await fetch(path,options);if(!r.ok){let detail='Request failed';try{const d=await r.json();detail=typeof d.detail==='string'?d.detail:'Please check the input.';}catch{}throw new Error(detail);}return r.json();}
function post(path,body){return request(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}
function showView(name){if(!titles[name])name='analyze';document.querySelectorAll('.view').forEach(v=>v.hidden=v.id!==`view-${name}`);document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active',b.dataset.view===name));$('breadcrumb').textContent=`Workspace / ${titles[name]}`;if(name==='metrics')loadMetrics();if(name==='about')request('/models').then(d=>$('model-config').textContent=JSON.stringify(d,null,2)).catch(e=>$('model-config').textContent=e.message);}
document.querySelectorAll('nav button').forEach(b=>b.addEventListener('click',()=>{location.hash=b.dataset.view;}));window.addEventListener('hashchange',()=>showView(location.hash.slice(1)));showView(location.hash.slice(1));
function count(){$('char-count').textContent=`${$('text-input').value.length.toLocaleString()} / 5,000`;}$('text-input').addEventListener('input',count);count();
document.querySelectorAll('[data-example]').forEach(b=>b.addEventListener('click',()=>{$('text-input').value=b.dataset.example;count();}));
function prediction(title,result){const card=el('div',undefined,'prediction');card.append(el('h3',title),el('div',result.label,'label'),el('span',result.uncertain?'Uncertain':'Above threshold','badge'+(result.uncertain?' uncertain':'')));for(const [name,score] of Object.entries(result.scores)){const row=el('div',undefined,'score-row'),track=el('div',undefined,'bar-track'),fill=el('div',undefined,'bar-fill');fill.style.width=`${Math.max(0,Math.min(1,score))*100}%`;track.append(fill);row.append(el('span',name),track,el('span',`${Math.round(score*100)}%`));card.append(row);}card.append(el('p',result.confidence_kind,'note'));return card;}
function renderAnalysis(result){const panel=el('article',undefined,'panel');const head=el('div',undefined,'result-head');head.append(el('h2',result.engine==='rules'?'Transparent rule baseline':result.engine==='browser'?'Browser transformers · sentiment + emotion':'Python transformers'),el('span',`${result.latency_ms} ms`));panel.append(head);const grid=el('div',undefined,'prediction-grid');grid.append(prediction('Sentiment',result.sentiment),prediction('Emotion',result.emotion));panel.append(grid);const emojis=el('div',undefined,'emoji-row');emojis.append(el('span','Emoji suggestions'));result.emojis.forEach(e=>emojis.append(el('span',e,'emoji')));panel.append(emojis);for(const [task,res]of Object.entries({sentiment:result.sentiment,emotion:result.emotion})){const detail=el('details');detail.append(el('summary',`${task} evidence`));const tokens=el('div',undefined,'evidence');res.evidence.forEach(t=>{const node=el('span',`${t.token} ${t.contribution>0?'+':''}${t.contribution}${t.negated?' · negated':''}`,'token'+(t.contribution<0?' negative':''));node.title=t.rule;tokens.append(node);});if(!res.evidence.length)tokens.append(el('span','No matching evidence.','note'));detail.append(tokens,el('p',res.explanation_note,'note'));panel.append(detail);}panel.append(el('p',`Model: ${result.model}`,'note'));result.warnings.forEach(w=>panel.append(el('p',w,'note')));return panel;}
let worker;let sequence=0;let modelBusy=false;
const serverConfig=request('/models').catch(()=>({transformers_enabled:false,threshold:.60}));
function browserInference(payload,onProgress){
  if(!worker)worker=new Worker('/static/sentiment-worker.js',{type:'module'});
  const activeWorker=worker,id=++sequence;
  return new Promise((resolve,reject)=>{
    let timer;
    const cleanup=()=>{clearTimeout(timer);activeWorker.removeEventListener('message',handler);activeWorker.removeEventListener('error',fail);};
    const arm=()=>{clearTimeout(timer);timer=setTimeout(()=>{cleanup();activeWorker.terminate();if(worker===activeWorker)worker=null;reject(new Error('Model loading or inference timed out. Check your connection and retry.'));},600000);};
    const fail=()=>{cleanup();activeWorker.terminate();if(worker===activeWorker)worker=null;reject(new Error('Browser model could not run. Check network access and device memory, then retry.'));};
    const handler=event=>{if(event.data.id!==id)return;if(event.data.status==='progress'){arm();onProgress(event.data.message);return;}cleanup();event.data.error?reject(new Error(event.data.error)):resolve(event.data);};
    activeWorker.addEventListener('message',handler);activeWorker.addEventListener('error',fail);arm();activeWorker.postMessage({id,...payload});
  });
}
function setModelBusy(value){modelBusy=value;$('analyze-btn').disabled=value;$('batch-btn').disabled=value;}
$('analyze-btn').addEventListener('click',async()=>{
  if(modelBusy)return;
  const text=$('text-input').value;if(!text.trim()){notice($('status'),'Enter some text to analyze.',true);return;}
  const engine=$('engine').value;setModelBusy(true);notice($('status'),'Analyzing…');$('result-area').replaceChildren();
  try{
    const config=await serverConfig;
    const useBrowser=engine==='browser'||((engine==='auto'||engine==='compare')&&!config.transformers_enabled);
    const data=await post('/analyze',{text,engine:useBrowser||engine==='rules'?'rules':engine==='auto'?'transformer':engine,explain:$('explain').checked});
    if(data.rules&&(engine==='rules'||engine==='compare'))$('result-area').append(renderAnalysis(data.rules));
    if(data.transformer)$('result-area').append(renderAnalysis(data.transformer));
    if(useBrowser){
      const response=await browserInference({text,explain:$('explain').checked,precision:$('precision').value,threshold:config.threshold},message=>notice($('status'),message));
      const panel=renderAnalysis(response.result);if(engine==='compare')panel.style.marginTop='20px';$('result-area').append(panel);
      const disagreement=engine==='compare'&&(data.rules.sentiment.label!==response.result.sentiment.label||data.rules.emotion.label!==response.result.emotion.label);
      notice($('status'),disagreement?'Analysis complete. Rules and transformers disagree; inspect both results.':'Analysis complete. Both sentiment and emotion transformers ran on your device.');
    }else if(data.transformer_error)notice($('status'),data.transformer_error,true);
    else notice($('status'),data.disagreement?'The models disagree. Inspect their evidence.':'Analysis complete.');
  }catch(e){notice($('status'),e.message,true);}finally{setModelBusy(false);}
});
let batchRows=[];
$('batch-btn').addEventListener('click',async()=>{
  if(modelBusy)return;
  const file=$('batch-file').files[0];if(!file){$('batch-status').textContent='Choose a file first.';return;}
  if(file.size>1000000){$('batch-status').textContent='File exceeds 1 MB.';return;}
  setModelBusy(true);$('download-btn').disabled=true;batchRows=[];$('batch-summary').replaceChildren();$('batch-table').querySelector('tbody').replaceChildren();$('batch-status').textContent='Validating file…';
  const body=new FormData();body.append('file',file);body.append('column',$('column').value);
  try{
    const data=await request('/batch',{method:'POST',body});
    let rows=data.rows;
    const transformers=$('batch-engine').value==='browser';
    if(transformers){
      const config=await serverConfig;
      const response=await browserInference({texts:rows.map(row=>row.text),precision:$('precision').value,threshold:config.threshold},message=>$('batch-status').textContent=message);
      rows=rows.map((row,i)=>({...row,sentiment:response.results[i].sentiment.label,emotion:response.results[i].emotion.label,confidence:response.results[i].sentiment.confidence,uncertain:response.results[i].sentiment.uncertain||response.results[i].emotion.uncertain}));
    }
    batchRows=rows;
    const distribution={};rows.forEach(row=>distribution[row.sentiment]=(distribution[row.sentiment]||0)+1);
    Object.entries(distribution).forEach(([k,v])=>{const item=el('div',undefined,'kpi');item.append(el('b',v),el('span',k));$('batch-summary').append(item);});
    const tbody=$('batch-table').querySelector('tbody');rows.forEach(row=>{const tr=el('tr');[row.row,row.text,row.sentiment,row.emotion,`${Math.round(row.confidence*100)}%`,row.uncertain?'Yes':'No'].forEach(value=>tr.append(el('td',value)));tbody.append(tr);});
    $('batch-status').textContent=`Analyzed ${rows.length} rows using ${transformers?'sentiment and emotion transformers':'rules'}.`;$('download-btn').disabled=false;
  }catch(e){$('batch-status').textContent=e.message;}finally{setModelBusy(false);}
});
function safeCell(value){let s=String(value);if(/^[\s]*[=+\-@]|^[\t\r\n]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';}
$('download-btn').addEventListener('click',()=>{const fields=['row','text','sentiment','emotion','confidence','uncertain'];const csv=[fields.join(','),...batchRows.map(r=>fields.map(f=>safeCell(r[f])).join(','))].join('\r\n');const link=el('a');link.href=URL.createObjectURL(new Blob(['\uFEFF'+csv],{type:'text/csv;charset=utf-8'}));link.download='predictions.csv';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);});
$('pre-btn').addEventListener('click',async()=>{const params=new URLSearchParams({remove_stopwords:$('stopwords').checked,stemming:$('stemming').checked,n:$('ngram').value});try{const data=await post('/preprocess?'+params,{text:$('pre-text').value});$('pre-result').replaceChildren();for(const name of ['original','normalized','tokens','processed','ngrams'])$('pre-result').append(el('h3',name.replaceAll('_',' ')),el('pre',typeof data[name]==='string'?data[name]:JSON.stringify(data[name],null,2)));$('pre-result').append(el('p',data.note,'note'));}catch(e){notice($('pre-result'),e.message,true);}});
let metricsData;
async function loadMetrics(){try{metricsData=await request('/metrics');renderMetrics();}catch(e){notice($('metrics-status'),e.message,true);}}
function renderMetrics(){if(!metricsData)return;$('metrics-content').replaceChildren();notice($('metrics-status'),metricsData.runs.length?`Generated ${metricsData.generated_at}. Scope and dataset are listed for each run.`:'Not yet measured. Run python -m scripts.evaluate to generate benchmark results.');for(const r of metricsData.runs){if($('metric-task').value!=='all'&&r.task!==$('metric-task').value)continue;if($('metric-engine').value!=='all'&&r.engine!==$('metric-engine').value)continue;const panel=el('article',undefined,'panel metric-panel');panel.append(el('h2',`${r.task} · ${r.engine}`),el('p',`${r.n} examples · ${r.scope} · ${r.dataset.id}`,'note'));const kpis=el('div',undefined,'kpis');for(const [label,value]of Object.entries({Accuracy:r.accuracy,'Macro F1':r.macro_f1,'Weighted F1':r.weighted_f1,'ECE':r.calibration.ece})){const item=el('div',undefined,'kpi');item.append(el('b',value.toFixed(3)),el('span',label));kpis.append(item);}panel.append(kpis);const charts=el('div',undefined,'metric-charts');for(const kind of ['confusion','reliability']){const img=el('img');img.src=`/figures/${r.task}-${r.engine}-${kind}.png`;img.alt=`${r.task} ${r.engine} ${kind}`;charts.append(img);}panel.append(charts);const table=el('table'),head=el('tr');['Class','Precision','Recall','F1','Count'].forEach(s=>head.append(el('th',s)));table.append(head);Object.entries(r.per_class).forEach(([label,d])=>{const tr=el('tr');[label,d.precision.toFixed(3),d.recall.toFixed(3),d['f1-score'].toFixed(3),d.support].forEach(s=>tr.append(el('td',s)));table.append(tr);});const scroll=el('div',undefined,'table-scroll');scroll.append(table);panel.append(scroll,el('p',r.metric_note,'note'),el('p',`Mean inference ${r.latency_ms.mean.toFixed(2)} ms · P95 ${r.latency_ms.p95.toFixed(2)} ms · ${r.calibration.note}`,'note'));const detail=el('details');detail.append(el('summary',`Inspect ${r.errors.length} errors (indices, without benchmark text)`));const controls=el('div',undefined,'controls'),sel=el('select');sel.setAttribute('aria-label','Filter actual class');['all',...Object.keys(r.per_class)].forEach(v=>{const o=el('option',v);o.value=v;sel.append(o);});const conf=el('input');conf.type='number';conf.min='0';conf.max='1';conf.step='.05';conf.value='1';conf.setAttribute('aria-label','Maximum error confidence');controls.append(el('span','Actual class'),sel,el('span','Maximum confidence'),conf);const errors=el('pre');function update(){const filtered=r.errors.filter(e=>(sel.value==='all'||e.actual===sel.value)&&e.confidence<=Number(conf.value));errors.textContent=JSON.stringify(filtered.slice(0,50),null,2)+`\nShowing up to 50 of ${filtered.length} matching errors.`;}sel.addEventListener('change',update);conf.addEventListener('input',update);update();detail.append(controls,errors);panel.append(detail);$('metrics-content').append(panel);}}
$('metric-task').addEventListener('change',renderMetrics);$('metric-engine').addEventListener('change',renderMetrics);
